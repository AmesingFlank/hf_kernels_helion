"""Helion RWKV WKV forward — matches kernels-community/rwkv `forward(w,u,k,v,y)`.

Exact match to the reference's numerically-stabilized RWKV-v4 WKV recurrence
(verified against the ref: `w` is used directly as the decay, not exp-transformed):

  per (b, c), carrying (aa, bb, pp=running max):
    ww = u + k[t];  p = max(pp, ww);   e1=exp(pp-p), e2=exp(ww-p)
    y[t] = (e1*aa + e2*v[t]) / (e1*bb + e2)
    ww = w + pp;    p = max(ww, k[t]);  e1=exp(ww-p), e2=exp(k[t]-p)
    aa = e1*aa + e2*v[t];  bb = e1*bb + e2;  pp = p

w,u: [C]; k,v: [B,T,C]. Sequential over T inside the kernel.
"""
from __future__ import annotations
import torch
import helion
import helion.language as hl

_NEG_INF = -1e38


@helion.aot_kernel(static_shapes=False)
def _rwkv_wkv(w: torch.Tensor, u: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    B, T, C = k.shape
    out = torch.empty_like(v)
    for tb, tc in hl.tile([B, C]):
        w_ = w[tc].to(torch.float32)[None, :]
        u_ = u[tc].to(torch.float32)[None, :]
        aa = hl.zeros([tb, tc], dtype=torch.float32)
        bb = hl.zeros([tb, tc], dtype=torch.float32)
        pp = hl.full([tb, tc], -1e38, dtype=torch.float32)
        for t in range(T):
            kt = k[tb, t, tc].to(torch.float32)
            vt = v[tb, t, tc].to(torch.float32)
            # output uses the bonus u
            ww = u_ + kt
            p = torch.maximum(pp, ww)
            e1 = torch.exp(pp - p)
            e2 = torch.exp(ww - p)
            out[tb, t, tc] = ((e1 * aa + e2 * vt) / (e1 * bb + e2)).to(out.dtype)
            # state update uses the decay w
            ww = w_ + pp
            p = torch.maximum(ww, kt)
            e1 = torch.exp(ww - p)
            e2 = torch.exp(kt - p)
            aa = e1 * aa + e2 * vt
            bb = e1 * bb + e2
            pp = p
    return out
