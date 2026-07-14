"""Helion segmented gather matrix-vector (SGMV) — punica multi-LoRA serving.

y[s[i]:s[i+1]] += x[s[i]:s[i+1]] @ WA[i].T @ WB[i]  : each token segment applies
its own LoRA (A,B) pair. x:[T,IN], WA:[nprob,rank,IN], WB:[nprob,rank,OUT].
"""
from __future__ import annotations
import torch
import helion
import helion.language as hl


@helion.kernel(static_shapes=False)
def _sgmv(y: torch.Tensor, x: torch.Tensor, WA: torch.Tensor, WB: torch.Tensor,
          s_start: torch.Tensor, s_end: torch.Tensor, rank: hl.constexpr) -> None:
    nprob = WA.shape[0]
    for i in hl.grid(nprob):
        start = s_start[i]
        end = s_end[i]
        seg = end - start
        if seg != 0:
            wa = WA[i, :, :]
            wb = WB[i, :, :]
            for tm in hl.tile(seg):
                xr = x[start + tm.index, :].to(torch.float32)
                proj = hl.dot(xr, wa.to(torch.float32).T)
                delta = hl.dot(proj, wb.to(torch.float32))
                y[start + tm.index, :] = (y[start + tm.index, :].to(torch.float32) + delta).to(y.dtype)
