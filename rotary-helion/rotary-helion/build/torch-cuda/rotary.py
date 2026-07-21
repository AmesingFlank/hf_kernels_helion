"""Helion rotary embedding, matching kernels-community/rotary apply_rotary.

out1 = x1*cos - x2*sin ; out2 = x1*sin + x2*cos  (conj flips sin sign).
x1,x2: [B,S,H,R]; cos,sin: [S,1,R]. Written into out1/out2 in place.
"""
from __future__ import annotations
import torch
import helion
import helion.language as hl


@helion.aot_kernel(static_shapes=False)
def _apply_rotary(out1, out2, x1, x2, cos, sin, conj: hl.constexpr):
    B, S, H, R = x1.shape
    for tb, ts, th in hl.tile([B, S, H]):
        a = x1[tb, ts, th, :].to(torch.float32)
        b = x2[tb, ts, th, :].to(torch.float32)
        c = cos[ts, 0, :].to(torch.float32)[None, :, None, :]
        s = sin[ts, 0, :].to(torch.float32)[None, :, None, :]
        if conj:
            o1 = a * c + b * s
            o2 = -a * s + b * c
        else:
            o1 = a * c - b * s
            o2 = a * s + b * c
        out1[tb, ts, th, :] = o1.to(out1.dtype)
        out2[tb, ts, th, :] = o2.to(out2.dtype)
