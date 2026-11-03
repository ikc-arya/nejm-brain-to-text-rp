"""Metal (Apple GPU) kernels for the two sequential ops: WKV and the multistep LIF neuron.

The PyTorch loops in wkv.py / model.py launch ~10 tiny ops per time step, which makes MPS ~5 s per step.
These kernels run one thread per (batch, channel) and loop over time, exactly like the upstream CUDA kernel.
WKV: line-by-line port of cuda/wkv_cuda.cu (kernel_forward / kernel_backward); the per-thread y/z/zexp arrays
of the CUDA backward live in global scratch buffers instead.
LIF: tau-decay charge, ATan(alpha) surrogate, hard reset to 0 with the reset kept in the graph (detach_reset=False):
    h_t = v_{t-1} + (x_t - v_{t-1}) / tau,  s_t = H(h_t - v_th),  v_t = h_t (1 - s_t)
backward (g = surrogate'(h_t - v_th)):
    dh_t = gs_t g + gv_t ((1 - s_t) - h_t g),  dx_t = dh_t / tau,  gv_{t-1} = dh_t (1 - 1/tau)
"""
import torch

_SRC = r"""
#include <metal_stdlib>
using namespace metal;
#define MIN_VALUE (-1e38f)

kernel void wkv_forward(constant uint& T [[buffer(0)]], constant uint& C [[buffer(1)]],
                        device const float* _w [[buffer(2)]], device const float* _u [[buffer(3)]],
                        device const float* _k [[buffer(4)]], device const float* _v [[buffer(5)]],
                        device float* _y [[buffer(6)]], uint idx [[thread_position_in_grid]]) {
    const uint _b = idx / C, _c = idx % C;
    const uint _offset = _b * T * C + _c;
    const float u = _u[_c], w = _w[_c];
    device const float* k = _k + _offset; device const float* v = _v + _offset; device float* y = _y + _offset;
    float p = 0, q = 0, o = MIN_VALUE;
    for (uint i = 0; i < T; i++) {
        const uint ii = i * C;
        float no = max(o, u + k[ii]);
        float A = exp(o - no), B = exp(u + k[ii] - no);
        y[ii] = (A * p + B * v[ii]) / (A * q + B);
        no = max(w + o, k[ii]);
        A = exp(w + o - no); B = exp(k[ii] - no);
        p = A * p + B * v[ii]; q = A * q + B; o = no;
    }
}

kernel void wkv_backward(constant uint& T [[buffer(0)]], constant uint& C [[buffer(1)]],
                         device const float* _w [[buffer(2)]], device const float* _u [[buffer(3)]],
                         device const float* _k [[buffer(4)]], device const float* _v [[buffer(5)]],
                         device const float* _gy [[buffer(6)]],
                         device float* _gw [[buffer(7)]], device float* _gu [[buffer(8)]],
                         device float* _gk [[buffer(9)]], device float* _gv [[buffer(10)]],
                         device float* _sy [[buffer(11)]], device float* _sz [[buffer(12)]], device float* _szexp [[buffer(13)]],
                         uint idx [[thread_position_in_grid]]) {
    const uint _b = idx / C, _c = idx % C;
    const uint _offset = _b * T * C + _c;
    const float u = _u[_c], w = _w[_c];
    device const float* k = _k + _offset; device const float* v = _v + _offset; device const float* gy = _gy + _offset;
    device float* gk = _gk + _offset; device float* gv = _gv + _offset;
    device float* y = _sy + _offset; device float* z = _sz + _offset; device float* zexp = _szexp + _offset;

    float gw = 0, gu = 0, p = 0, q = 0, dpdw = 0, dqdw = 0, o = MIN_VALUE;
    for (uint i = 0; i < T; i++) {
        const uint ii = i * C;
        float no = max(o, k[ii] + u);
        float A = exp(o - no), B = exp(k[ii] + u - no);
        float num = A * p + B * v[ii];
        float iden = 1 / (A * q + B);
        y[ii] = num * iden; z[ii] = iden; zexp[ii] = k[ii] + u - no;
        gw += gy[ii] * (dpdw - dqdw * y[ii]) * iden * A;
        gu += gy[ii] * (v[ii] - y[ii]) * B * iden;
        no = max(w + o, k[ii]);
        A = exp(w + o - no); B = exp(k[ii] - no);
        dpdw = A * (p + dpdw); dqdw = A * (q + dqdw);
        p = A * p + B * v[ii]; q = A * q + B; o = no;
    }
    float gp = 0, gq = 0;
    o = MIN_VALUE;
    for (int i = int(T) - 1; i >= 0; i--) {
        const uint ii = uint(i) * C;
        float A = gy[ii] * z[ii] * exp(zexp[ii]);
        float B = exp(k[ii] + o);
        gk[ii] = A * (v[ii] - y[ii]) + B * (gp * v[ii] + gq);
        gv[ii] = A + B * gp;
        float no = max(w + o, zexp[ii] - k[ii] - u);
        A = exp(w + o - no);
        B = gy[ii] * z[ii] * exp(zexp[ii] - k[ii] - u - no);
        gp = A * gp + B; gq = A * gq - B * y[ii]; o = no;
    }
    _gw[_b * C + _c] = gw * _w[_c];
    _gu[_b * C + _c] = gu;
}

kernel void lif_forward(constant uint& T [[buffer(0)]], constant uint& C [[buffer(1)]],
                        constant float& tau [[buffer(2)]], constant float& vth [[buffer(3)]],
                        device const float* _x [[buffer(4)]], device float* _s [[buffer(5)]], device float* _h [[buffer(6)]],
                        uint idx [[thread_position_in_grid]]) {
    const uint _b = idx / C, _c = idx % C;
    const uint _offset = _b * T * C + _c;
    float v = 0;
    for (uint i = 0; i < T; i++) {
        const uint ii = _offset + i * C;
        float h = v + (_x[ii] - v) / tau;
        float s = (h - vth) >= 0 ? 1.0f : 0.0f;
        _h[ii] = h; _s[ii] = s;
        v = h * (1 - s);
    }
}

kernel void lif_backward(constant uint& T [[buffer(0)]], constant uint& C [[buffer(1)]],
                         constant float& tau [[buffer(2)]], constant float& vth [[buffer(3)]], constant float& alpha [[buffer(4)]],
                         device const float* _gs [[buffer(5)]], device const float* _s [[buffer(6)]], device const float* _h [[buffer(7)]],
                         device float* _gx [[buffer(8)]], uint idx [[thread_position_in_grid]]) {
    const uint _b = idx / C, _c = idx % C;
    const uint _offset = _b * T * C + _c;
    const float half_pi_alpha = 1.57079632679489661923f * alpha;
    float gv = 0;
    for (int i = int(T) - 1; i >= 0; i--) {
        const uint ii = _offset + uint(i) * C;
        float h = _h[ii], s = _s[ii];
        float xs = half_pi_alpha * (h - vth);
        float g = alpha / 2.0f / (1.0f + xs * xs);
        float dh = _gs[ii] * g + gv * ((1 - s) - h * g);
        _gx[ii] = dh / tau;
        gv = dh * (1 - 1 / tau);
    }
}
"""

_lib = None


def _kernels():
    global _lib
    if _lib is None:
        _lib = torch.mps.compile_shader(_SRC)
    return _lib


class WKVMetal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, u, k, v):
        B, T, C = k.shape
        w = -torch.exp(w.float().contiguous())
        u, k, v = u.float().contiguous(), k.float().contiguous(), v.float().contiguous()
        ctx.save_for_backward(w, u, k, v)
        y = torch.empty((B, T, C), device=k.device)
        _kernels().wkv_forward(T, C, w, u, k, v, y, threads=B * C)
        return y

    @staticmethod
    def backward(ctx, gy):
        w, u, k, v = ctx.saved_tensors
        B, T, C = k.shape
        gw, gu = torch.empty((B, C), device=k.device), torch.empty((B, C), device=k.device)
        gk, gv = torch.empty_like(k), torch.empty_like(k)
        sy, sz, szexp = torch.empty_like(k), torch.empty_like(k), torch.empty_like(k)
        _kernels().wkv_backward(T, C, w, u, k, v, gy.float().contiguous(), gw, gu, gk, gv, sy, sz, szexp, threads=B * C)
        return gw.sum(0), gu.sum(0), gk, gv


class LIFMetal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tau, v_threshold, alpha):
        B, T, C = x.shape
        x = x.float().contiguous()
        s, h = torch.empty_like(x), torch.empty_like(x)
        _kernels().lif_forward(T, C, float(tau), float(v_threshold), x, s, h, threads=B * C)
        ctx.save_for_backward(s, h)
        ctx.params = (tau, v_threshold, alpha)
        return s

    @staticmethod
    def backward(ctx, gs):
        s, h = ctx.saved_tensors
        tau, vth, alpha = ctx.params
        B, T, C = s.shape
        gx = torch.empty_like(s)
        _kernels().lif_backward(T, C, float(tau), float(vth), float(alpha), gs.float().contiguous(), s, h, gx, threads=B * C)
        return gx, None, None, None
