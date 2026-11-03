"""Shared SpikeGPT backbone + per-source input stems / output heads.

    x [B, T, C_src] (20 ms bins)
      -> per-session scale/bias (C_src)               "per-day scale/bias" in the pipeline diagram
      -> w_in: C_src -> d per bin                      shared across sessions if they share channels (human, 512->384),
                                                       else one per session (mouse units differ per session)
      -> patching + patch_proj: patch_size*d -> d per patch (per source), as a Conv1d(d, d, patch_size, patch_stride)
      -> ATan spike (SpikeGPT's binary input embedding, optional)
      -> SpikeGPT backbone (shared, causal)  -> z [B, T', d]
    heads: ar_head (per source, Stage 1) predicts the next `ar_horizon` raw bins after each patch;
           ctc_head (d -> n_phonemes, Stage 2) lives in NeuralSpikeGPT.

Token t sees bins [t*stride, t*stride + patch_size), same frame rate as the B2T GRU baseline (14 / 4).
"""
import torch
import torch.nn as nn

from spikegpt import SpikeGPTBackbone, spike


def n_tokens(n_bins, patch_size, patch_stride):
    """Number of patches for sequences of n_bins (tensor or int), as in the GRU baseline."""
    if patch_size <= 0:
        return n_bins
    return (n_bins - patch_size) // patch_stride + 1


class SourceAdapter(nn.Module):
    def __init__(self, channels_per_session, d_model, patch_size=14, patch_stride=4, ar_horizon=4,
                 input_dropout=0.2, binary_input=True, ar_output='linear', mean_rate=None):
        super().__init__()
        self.channels = [int(c) for c in channels_per_session]
        self.shared_io = len(set(self.channels)) == 1
        self.d_model, self.patch_size, self.patch_stride = d_model, patch_size, patch_stride
        self.ar_horizon, self.binary_input = ar_horizon, binary_input
        self.ar_output = ar_output  # 'linear' (MSE on z-scored features) | 'log_rate' (Poisson on counts)

        self.day_scale = nn.ParameterList([nn.Parameter(torch.ones(c)) for c in self.channels])
        self.day_bias = nn.ParameterList([nn.Parameter(torch.zeros(c)) for c in self.channels])
        self.dropout = nn.Dropout(input_dropout)
        io_channels = self.channels[:1] if self.shared_io else self.channels
        self.w_in = nn.ModuleList([nn.Linear(c, d_model) for c in io_channels])
        # same linear map as unfold + Linear(patch_size * d -> d), without materialising the [B, T', patch_size * d] copy
        self.patch_proj = nn.Conv1d(d_model, d_model, max(patch_size, 1), max(patch_stride, 1))
        self.ar_head = nn.ModuleList([nn.Linear(d_model, ar_horizon * c) for c in io_channels])
        if ar_output == 'log_rate' and mean_rate is not None:  # start at the per-unit mean rate, not exp(0) = 1/bin
            with torch.no_grad():
                for head, rate in zip(self.ar_head, mean_rate):
                    head.bias.copy_(torch.log(torch.tensor(rate).clamp(min=1e-3)).repeat(ar_horizon))

    def _io_index(self, session_idx):
        if self.shared_io:
            return 0
        s = session_idx.unique()
        assert len(s) == 1, 'sessions with different channel counts must be batched separately'
        return int(s[0])

    def embed(self, x, session_idx):
        """x [B, T, C] -> tokens [B, T', d]."""
        io = self._io_index(session_idx)
        if self.shared_io:  # per-sample session parameters
            scale, bias = torch.stack(list(self.day_scale))[session_idx], torch.stack(list(self.day_bias))[session_idx]
        else:  # whole batch is one session
            scale, bias = self.day_scale[io].unsqueeze(0), self.day_bias[io].unsqueeze(0)
        x = x * scale.unsqueeze(1) + bias.unsqueeze(1)
        x = self.dropout(x)
        x = self.w_in[io](x)  # [B, T, d]
        x = self.patch_proj(x.transpose(1, 2)).transpose(1, 2)  # [B, T', d]
        return spike(x) if self.binary_input else x

    def predict_next(self, z, session_idx):
        """z [B, T', d] -> [B, T', ar_horizon, C]: the ar_horizon bins right after each token's patch."""
        out = self.ar_head[self._io_index(session_idx)](z)
        return out.view(z.shape[0], z.shape[1], self.ar_horizon, -1)


class NeuralSpikeGPT(nn.Module):
    def __init__(self, adapters: dict, n_embd=384, n_layer=6, dropout=0.03, n_phonemes=41, use_native_kernels=True,
                 grad_checkpoint=False):
        super().__init__()
        self.adapters = nn.ModuleDict(adapters)
        self.backbone = SpikeGPTBackbone(n_embd, n_layer, dropout, use_native_kernels, grad_checkpoint)
        self.ctc_head = nn.Linear(n_embd, n_phonemes)
        nn.init.xavier_uniform_(self.ctc_head.weight)

    def encode(self, x, source, session_idx):
        """Latents for Stage 2 / Stage 3: x [B, T, C_src] -> z [B, T', n_embd]."""
        return self.backbone(self.adapters[source].embed(x, session_idx))

    def forward_ctc(self, x, source, session_idx):
        return self.ctc_head(self.encode(x, source, session_idx))

    def forward_ar(self, x, source, session_idx):
        z = self.encode(x, source, session_idx)
        return self.adapters[source].predict_next(z, session_idx)


def build_model(model_cfg, sources_meta, use_native_kernels=True, grad_checkpoint=False):
    """sources_meta: {name: {'channels': [C per session], 'ar_output': ...}} (from the sources, or a checkpoint)."""
    adapters = {
        name: SourceAdapter(meta['channels'], model_cfg['n_embd'], model_cfg['patch_size'], model_cfg['patch_stride'],
                            model_cfg['ar_horizon'], model_cfg['input_dropout'], model_cfg['binary_input'],
                            meta.get('ar_output', 'linear'), meta.get('mean_rate'))
        for name, meta in sources_meta.items()
    }
    return NeuralSpikeGPT(adapters, model_cfg['n_embd'], model_cfg['n_layer'], model_cfg['backbone_dropout'],
                          model_cfg.get('n_phonemes', 41), use_native_kernels, grad_checkpoint)
