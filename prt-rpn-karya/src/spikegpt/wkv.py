"""RWKV-v4 WKV operator: upstream CUDA kernel on GPU, pure-PyTorch recurrence elsewhere (MPS / CPU).

Adapted from SpikeGPT (https://github.com/ridgerchu/SpikeGPT @ 029f86f, BSD-2-Clause, see LICENSE),
which takes it from RWKV-LM. The PyTorch fallback is a line-by-line port of kernel_forward in cuda/wkv_cuda.cu
(running max `o` keeps the exponentials finite); autograd handles its backward.
"""
import os

import torch

T_MAX = 1024  # upstream default; max tokens per sequence for the CUDA kernel
_HERE = os.path.dirname(os.path.abspath(__file__))
_wkv_cuda = None


def _load_cuda():
    global _wkv_cuda
    if _wkv_cuda is None:
        from torch.utils.cpp_extension import load
        _wkv_cuda = load(name='wkv', sources=[os.path.join(_HERE, 'cuda', 'wkv_op.cpp'), os.path.join(_HERE, 'cuda', 'wkv_cuda.cu')],
                         verbose=False,
                         extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3',
                                            f'-DTmax={T_MAX}'])
    return _wkv_cuda


class WKVCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        assert T <= T_MAX, f'T={T} > T_MAX={T_MAX}'
        assert B * C % min(C, 1024) == 0
        ctx.B, ctx.T, ctx.C = B, T, C
        w = -torch.exp(w.float().contiguous())
        u, k, v = u.float().contiguous(), k.float().contiguous(), v.float().contiguous()
        ctx.save_for_backward(w, u, k, v)
        y = torch.empty((B, T, C), device=k.device, memory_format=torch.contiguous_format)
        _load_cuda().forward(B, T, C, w, u, k, v, y)
        return y

    @staticmethod
    def backward(ctx, gy):
        B, T, C = ctx.B, ctx.T, ctx.C
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device=gy.device).contiguous()
        gu = torch.zeros((B, C), device=gy.device).contiguous()
        gk = torch.zeros((B, T, C), device=gy.device).contiguous()
        gv = torch.zeros((B, T, C), device=gy.device).contiguous()
        _load_cuda().backward(B, T, C, w, u, k, v, gy.float().contiguous(), gw, gu, gk, gv)
        return (None, None, None, gw.sum(0), gu.sum(0), gk, gv)


def wkv_torch(time_decay, time_first, k, v):
    """time_decay, time_first: (C,); k, v: (B, T, C) -> (B, T, C), float32."""
    w = -torch.exp(time_decay.float())
    u = time_first.float()
    k, v = k.float(), v.float()
    B, T, C = k.shape
    p = torch.zeros(B, C, device=k.device)
    q = torch.zeros(B, C, device=k.device)
    o = torch.full((B, C), -1e38, device=k.device)
    ys = []
    for t in range(T):
        kt, vt = k[:, t], v[:, t]
        no = torch.maximum(o, u + kt)
        a, b = torch.exp(o - no), torch.exp(u + kt - no)
        ys.append((a * p + b * vt) / (a * q + b))
        no = torch.maximum(w + o, kt)
        a, b = torch.exp(w + o - no), torch.exp(kt - no)
        p, q, o = a * p + b * vt, a * q + b, no
    return torch.stack(ys, dim=1)


def wkv(time_decay, time_first, k, v, use_cuda_kernel=True):
    """use_cuda_kernel: use the native kernel of the device (CUDA kernel on GPU, Metal kernel on MPS)."""
    if use_cuda_kernel and k.is_cuda:
        B, T, C = k.shape
        return WKVCuda.apply(B, T, C, time_decay, time_first, k, v)
    if use_cuda_kernel and k.device.type == 'mps':
        from .metal import WKVMetal
        return WKVMetal.apply(time_decay, time_first, k, v)
    return wkv_torch(time_decay, time_first, k, v)
