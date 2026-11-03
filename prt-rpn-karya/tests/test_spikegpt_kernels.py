"""Native kernels vs reference implementations, and causality of the backbone.

    .venv-train/bin/python tests/test_spikegpt_kernels.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import torch
from spikegpt import SpikeGPTBackbone, MultiStepLIF, wkv, wkv_torch


def closed_form_wkv(td, tf, k, v):
    w = torch.exp(td)
    y = torch.zeros_like(k)
    for t in range(k.shape[1]):
        num, den = torch.exp(tf + k[:, t]) * v[:, t], torch.exp(tf + k[:, t])
        for i in range(t):
            e = torch.exp(-(t - 1 - i) * w + k[:, i])
            num, den = num + e * v[:, i], den + e
        y[:, t] = num / den
    return y


def grads(fn, *xs):
    xs = [x.detach().clone().requires_grad_() for x in xs]
    y = fn(*xs)
    (y * torch.linspace(-1, 1, y.numel(), device=y.device).view_as(y)).sum().backward()
    return y.detach().cpu(), [x.grad.cpu() for x in xs]


def test_wkv_fallback():
    torch.manual_seed(0)
    td, tf, k, v = torch.randn(6), torch.randn(6), torch.randn(2, 20, 6), torch.randn(2, 20, 6)
    assert torch.allclose(wkv_torch(td, tf, k, v), closed_form_wkv(td, tf, k, v), atol=1e-5)


def native_device():
    return 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else None)


def test_native_wkv():
    dev = native_device()
    if dev is None:
        return print('skip native wkv (no GPU)')
    torch.manual_seed(1)
    ins = [torch.randn(16), torch.randn(16), torch.randn(4, 50, 16) * 2, torch.randn(4, 50, 16)]
    y_ref, g_ref = grads(lambda *a: wkv_torch(*a), *ins)
    y_nat, g_nat = grads(lambda *a: wkv(*a, use_cuda_kernel=True), *[x.to(dev) for x in ins])
    assert torch.allclose(y_ref, y_nat, atol=1e-4), (y_ref - y_nat).abs().max()
    for a, b in zip(g_ref, g_nat):
        assert torch.allclose(a, b, atol=1e-3, rtol=1e-3), (a - b).abs().max()


def test_native_lif():
    dev = native_device()
    if dev != 'mps':
        return print('skip native lif (MPS only)')
    torch.manual_seed(2)
    x = torch.randn(4, 60, 32) * 2
    y_ref, (g_ref,) = grads(MultiStepLIF(use_native_kernel=False), x)
    y_nat, (g_nat,) = grads(MultiStepLIF(use_native_kernel=True), x.to(dev))
    assert torch.equal(y_ref, y_nat)
    assert torch.allclose(g_ref, g_nat, atol=1e-5), (g_ref - g_nat).abs().max()


def test_causal():
    torch.manual_seed(3)
    m = SpikeGPTBackbone(64, 2, dropout=0.).eval()
    for p in m.parameters():  # RWKV init zeroes several matrices (blocks start as identity): randomise them
        if p.dim() == 2:
            torch.nn.init.normal_(p, std=0.3)
    x = torch.randn(2, 40, 64)
    y1 = m(x)
    x2 = x.clone(); x2[:, 25:] = torch.randn(2, 15, 64)
    y2 = m(x2)
    assert torch.equal(y1[:, :25], y2[:, :25])
    assert not torch.allclose(y1[:, 25:], y2[:, 25:])


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('ok', name)
