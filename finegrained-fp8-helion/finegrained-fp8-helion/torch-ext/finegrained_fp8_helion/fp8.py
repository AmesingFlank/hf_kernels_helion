"""Helion blockwise W8A8 FP8 matmul — matches kernels-community/finegrained-fp8
`w8a8_block_fp8_matmul` EXACTLY (same signature + scale layout).

Computes C = A @ B.T, both fp8_e4m3fn with per-block scales:
  A:  [M, K]  fp8            As: [M, K//block_k]          (per-token-group act scales)
  B:  [N, K]  fp8            Bs: [N//block_n, K//block_k] (per-block weight scales)
  block_size = [block_n, block_k];  out: [M, N], fp32 accumulation.
The ref writes this hot loop as a Triton kernel; here it is the Helion kernel.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.aot_kernel(static_shapes=False)
def _w8a8_block_fp8_matmul(
    A: torch.Tensor, B: torch.Tensor, As: torch.Tensor, Bs: torch.Tensor,
    block_n: hl.constexpr, block_k: hl.constexpr,
) -> torch.Tensor:
    M, K = A.shape
    N, _ = B.shape
    out = torch.empty([M, N], dtype=torch.float32, device=A.device)
    # Tile N by block_n so each N-tile maps to one weight-scale row (nb).
    for tm, tn in hl.tile([M, N], block_size=[None, block_n]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        nb = tn.begin // block_n
        for tk in hl.tile(K, block_size=block_k):
            a = A[tm, tk].to(torch.float32)
            b = B[tn, tk].to(torch.float32)
            kb = tk.begin // block_k
            sa = As[tm, kb]      # [tm] per-token(-group) scale for this K block
            sb = Bs[nb, kb]      # scalar weight scale for this (N,K) block
            acc = acc + hl.dot(a, b.T) * sa[:, None] * sb
        out[tm, tn] = acc
    return out
