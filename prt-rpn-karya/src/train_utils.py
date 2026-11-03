"""Shared pieces of the Stage 1 (SSL) and Stage 2 (CTC) trainers, device-agnostic (CUDA / MPS / CPU).

The augmentation, cosine schedule and PER code follow rnn_trainer.py (the B2T GRU baseline) so numbers stay comparable.
"""
import contextlib
import logging
import math
import os
import pathlib
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as taF
from scipy.ndimage import gaussian_filter1d
from torch.optim.lr_scheduler import LambdaLR

from data_augmentations import gauss_smooth


def get_device(pref='auto'):
    if pref != 'auto':
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def autocast(device, enabled):
    """bf16 autocast on CUDA only; MPS / CPU run in fp32 (the native kernels are fp32 anyway)."""
    if enabled and device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


def set_seed(seed):
    if seed != -1:
        np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)


def make_logger(output_dir, name):
    logger = logging.getLogger(name)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter(fmt='%(asctime)s: %(message)s')
    for h in (logging.FileHandler(str(pathlib.Path(output_dir, 'training_log'))), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt); logger.addHandler(h)
    return logger


def param_groups(model, lrs, weight_decay):
    """lrs: {'backbone': lr, 'adapters': lr, 'ctc_head': lr}. No weight decay on biases, norms, day scale/bias and
    the RWKV time_* vectors (upstream SpikeGPT uses no weight decay at all)."""
    groups = []
    for prefix, lr in lrs.items():
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            if not n.startswith(prefix + '.'):
                continue
            (decay if p.dim() >= 2 and 'day_' not in n else no_decay).append(p)
        groups += [{'params': decay, 'lr': lr, 'weight_decay': weight_decay, 'group_type': prefix},
                   {'params': no_decay, 'lr': lr, 'weight_decay': 0., 'group_type': prefix}]
    return [g for g in groups if g['params']]


def make_optimizer(model, lrs, args, device):
    return torch.optim.AdamW(param_groups(model, lrs, args['weight_decay']), betas=(args['beta0'], args['beta1']),
                             eps=args['epsilon'], fused=device.type == 'cuda')


def cosine_scheduler(optim, lr_min_ratio, decay_steps, warmup_steps):
    """Linear warmup then cosine decay to lr_min_ratio (rnn_trainer.create_cosine_lr_scheduler)."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        if step < decay_steps:
            progress = float(step - warmup_steps) / float(max(1, decay_steps - warmup_steps))
            return max(lr_min_ratio, lr_min_ratio + (1 - lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))
        return lr_min_ratio
    return LambdaLR(optim, [lr_lambda] * len(optim.param_groups))


def causal_smooth(x, std=2, size=100):
    """One-sided (past-only) Gaussian smoothing along time: x [B, T, C]. Same kernel width rule as gauss_smooth."""
    inp = np.zeros(size, dtype=np.float32); inp[size // 2] = 1
    k = gaussian_filter1d(inp, std)[size // 2:]  # lags 0, 1, 2, ...
    k = k[k > 0.01]; k = k / k.sum()
    k = torch.tensor(k[::-1].copy(), dtype=x.dtype, device=x.device).view(1, 1, -1).repeat(x.shape[-1], 1, 1)
    xt = F.pad(x.permute(0, 2, 1), (k.shape[-1] - 1, 0))
    return F.conv1d(xt, k, groups=x.shape[-1]).permute(0, 2, 1)


def transform(x, lengths, cfg, train, device):
    """Augment + smooth the model input (rnn_trainer.transform_data). Returns (x, lengths, cut).
    cfg.smooth: 'causal' (default, required for the AR objective), 'acausal' (baseline behaviour) or 'none'."""
    cut = 0
    if train:
        B, T, C = x.shape
        if cfg.get('white_noise_std', 0) > 0:
            x = x + torch.randn_like(x) * cfg['white_noise_std']
        if cfg.get('constant_offset_std', 0) > 0:
            x = x + torch.randn((B, 1, C), device=device) * cfg['constant_offset_std']
        if cfg.get('random_cut', 0) > 0:
            cut = np.random.randint(0, cfg['random_cut'])
            x, lengths = x[:, cut:], lengths - cut
    smooth = cfg.get('smooth', 'causal')
    if smooth == 'causal':
        x = causal_smooth(x, cfg.get('smooth_kernel_std', 2), cfg.get('smooth_kernel_size', 100))
    elif smooth == 'acausal':
        x = gauss_smooth(x, device, cfg.get('smooth_kernel_std', 2), cfg.get('smooth_kernel_size', 100))
    return x, lengths, cut


def pad_time(x, multiple):
    """Right-pad the time axis of [B, T, C] to a multiple of `multiple` bins. MPS builds a graph per input shape, so
    bucketing T avoids recompiling for every batch; padding is at the end and the model is causal, so valid outputs
    are unchanged."""
    if multiple <= 1 or x.shape[1] % multiple == 0:
        return x
    return F.pad(x, (0, 0, 0, multiple - x.shape[1] % multiple))


def random_crop(batch, max_bins, rng):
    """Crop each row to a random window of <= max_bins valid bins (x, target, lengths)."""
    T = batch['x'].shape[1]
    if T <= max_bins:
        return batch
    lengths = batch['lengths']
    starts = [int(rng.integers(0, max(1, int(l) - max_bins + 1))) for l in lengths]
    out = dict(batch)
    for key in ('x', 'target'):
        out[key] = torch.stack([batch[key][i, s:s + max_bins] for i, s in enumerate(starts)])
    out['lengths'] = torch.clamp(lengths - torch.tensor(starts), max=max_bins)
    return out


def ctc_loss(logits, labels, input_lengths, label_lengths):
    """Mean CTC loss (blank = 0). MPS has no CTC kernel, so the loss runs on CPU there (gradients flow back)."""
    log_probs = logits.float().log_softmax(-1).permute(1, 0, 2)
    if log_probs.device.type == 'mps':
        log_probs, labels = log_probs.cpu(), labels.cpu()
        input_lengths, label_lengths = input_lengths.cpu(), label_lengths.cpu()
    loss = F.ctc_loss(log_probs, labels, input_lengths, label_lengths, blank=0, reduction='none', zero_infinity=False)
    return loss.mean()


def greedy_decode(logits, lengths):
    """argmax -> collapse repeats -> drop blanks, per row."""
    out = []
    for i in range(logits.shape[0]):
        seq = torch.argmax(logits[i, :lengths[i]], dim=-1)
        seq = torch.unique_consecutive(seq).cpu().numpy()
        out.append(seq[seq != 0])
    return out


def edit_distance(a, b):
    return taF.edit_distance(a, b)


def save_args(args, path):
    from omegaconf import OmegaConf
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        OmegaConf.save(config=args, f=f)
