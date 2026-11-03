"""SpikeGPT backbone (spiking RWKV) for neural time series.

Adapted from SpikeGPT (https://github.com/ridgerchu/SpikeGPT @ 029f86f, src/model.py, BSD-2-Clause, see LICENSE).
Kept as upstream: RWKV_TimeMix, RWKV_ChannelMix, Block (LIF on both residual branches, dropout 0.03),
RWKV_Init's "fancy" init, ATan(alpha=2) surrogate, LIF(tau=2, v_threshold=1, hard reset to 0).
Changed:
- token embedding + LM head removed; inputs/outputs are (B, T, n_embd) and the stems/heads live in neural_encoder.py
- spikingjelly replaced by the minimal equivalents below (MultiStepLIFNode torch backend, decay_input=True,
  detach_reset=False); membrane state starts at 0 every forward, same as upstream's functional.reset_net per batch
- torch.jit.ScriptModule dropped, WKV dispatches to the CUDA kernel or a PyTorch fallback (wkv.py)
"""
import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .wkv import wkv


class _ATanSpike(torch.autograd.Function):
    """Heaviside forward, arctan surrogate backward (spikingjelly surrogate.ATan)."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        a = ctx.alpha
        return grad * (a / 2 / (1 + (math.pi / 2 * a * x) ** 2)), None


def spike(x, alpha=2.0):
    return _ATanSpike.apply(x, alpha)


class MultiStepLIF(nn.Module):
    """x: (B, T, C) -> binary spikes (B, T, C). v <- v + (x - v)/tau; s = H(v - v_th); v <- v * (1 - s)."""

    def __init__(self, tau=2., v_threshold=1., alpha=2.0, use_native_kernel=True):
        super().__init__()
        self.tau, self.v_threshold, self.alpha = tau, v_threshold, alpha
        self.use_native_kernel = use_native_kernel

    def forward(self, x):
        if self.use_native_kernel and x.device.type == 'mps':
            from .metal import LIFMetal
            return LIFMetal.apply(x, self.tau, self.v_threshold, self.alpha).to(x.dtype)
        v = torch.zeros_like(x[:, 0])
        out = []
        for t in range(x.shape[1]):
            v = v + (x[:, t] - v) / self.tau
            s = spike(v - self.v_threshold, self.alpha)
            v = v * (1. - s)
            out.append(s)
        return torch.stack(out, dim=1)


class RWKV_TimeMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, use_cuda_kernel=True):
        super().__init__()
        self.layer_id = layer_id
        self.use_cuda_kernel = use_cuda_kernel
        attn_sz = n_embd

        with torch.no_grad():  # fancy init
            ratio_0_to_1 = (layer_id / (n_layer - 1)) if n_layer > 1 else 0.  # 0 to 1
            ratio_1_to_almost0 = (1.0 - (layer_id / n_layer))  # 1 to ~0

            decay_speed = torch.ones(attn_sz)
            for h in range(attn_sz):
                decay_speed[h] = -5 + 8 * (h / (attn_sz - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)

            zigzag = (torch.tensor([(i + 1) % 3 - 1 for i in range(attn_sz)]) * 0.5)
            self.time_first = nn.Parameter(torch.ones(attn_sz) * math.log(0.3) + zigzag)

            x = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                x[0, 0, i] = i / n_embd
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k, v = self.key(xk), self.value(xv)
        sr = torch.sigmoid(self.receptance(xr))
        rwkv = sr * wkv(self.time_decay, self.time_first, k, v, self.use_cuda_kernel).to(x.dtype)
        return self.output(rwkv)


class RWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = (1.0 - (layer_id / n_layer))  # 1 to ~0
            x = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                x[0, 0, i] = i / n_embd
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))

        hidden_sz = 4 * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = torch.square(torch.relu(self.key(xk)))
        return torch.sigmoid(self.receptance(xr)) * self.value(k)


class Block(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, dropout=0.03, use_cuda_kernel=True):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.lif1 = MultiStepLIF(use_native_kernel=use_cuda_kernel)
        self.lif2 = MultiStepLIF(use_native_kernel=use_cuda_kernel)
        self.dropout = nn.Dropout(dropout)
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)
        self.att = RWKV_TimeMix(n_embd, n_layer, layer_id, use_cuda_kernel)
        self.ffn = RWKV_ChannelMix(n_embd, n_layer, layer_id)

    def forward(self, x):
        if self.layer_id == 0:
            x = self.ln0(x)
        x = x + self.lif1(self.att(self.ln1(x)))
        x = x + self.lif2(self.ffn(self.ln2(x)))
        return self.dropout(x)


def rwkv_init(module):
    """Upstream RWKV_Init for the Linear layers of the blocks (no embedding / vocab head here)."""
    for m in module.modules():
        if not isinstance(m, nn.Linear):
            continue
        ww = m.weight
        shape = ww.shape
        gain, scale = 1.0, 1.0
        if shape[0] > shape[1]:
            gain = math.sqrt(shape[0] / shape[1])
        if hasattr(m, 'scale_init'):
            scale = m.scale_init
        gain *= scale
        with torch.no_grad():
            if scale == -999:
                nn.init.eye_(ww)
            elif gain == 0:
                nn.init.zeros_(ww)  # zero init is great for some RWKV matrices
            elif gain > 0:
                nn.init.orthogonal_(ww, gain=gain)
            else:
                nn.init.normal_(ww, mean=0.0, std=-scale)


class SpikeGPTBackbone(nn.Module):
    """(B, T, n_embd) -> (B, T, n_embd). Causal: output at t depends only on inputs <= t."""

    def __init__(self, n_embd=384, n_layer=6, dropout=0.03, use_cuda_kernel=True, grad_checkpoint=False):
        super().__init__()
        self.n_embd, self.n_layer = n_embd, n_layer
        self.grad_checkpoint = grad_checkpoint  # recompute each block in backward: ~6x less activation memory
        self.blocks = nn.Sequential(*[Block(n_embd, n_layer, i, dropout, use_cuda_kernel) for i in range(n_layer)])
        self.ln_out = nn.LayerNorm(n_embd)
        rwkv_init(self)

    def forward(self, x):
        for block in self.blocks:
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.ln_out(x)
