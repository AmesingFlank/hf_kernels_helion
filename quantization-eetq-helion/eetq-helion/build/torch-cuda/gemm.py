"""Helion int8 weight-only GEMM (EETQ w8_a16_gemm).

out = x @ (W_int8 * scale)ᵀ : fp16 activations, per-output-channel int8 weights.
"""
from __future__ import annotations
import torch
import helion
import helion.language as hl


@helion.kernel(static_shapes=False)
def _w8_a16_gemm(x: torch.Tensor, W: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    N, _ = W.shape
    out = torch.empty([M, N], dtype=torch.float16, device=x.device)
    for tm, tn in hl.tile([M, N]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(K):
            a = x[tm, tk].to(torch.float32)
            w = W[tn, tk].to(torch.float32)
            acc = hl.dot(a, w.T, acc=acc)
        out[tm, tn] = (acc * scale[tn].to(torch.float32)[None, :]).to(torch.float16)
    return out
