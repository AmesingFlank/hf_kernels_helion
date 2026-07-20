"""Helion grouped GEMM — the compute core of block-sparse MoE (megablocks).

Tokens are routed to experts and packed contiguously per expert; group_offsets
delimits each expert's rows. out[i] = A[i] @ W[expert(i)]. This is exactly the
grouped/dropless-MoE expert matmul megablocks accelerates.
"""
from __future__ import annotations
import torch
import helion
import helion.experimental
import helion.language as hl


@helion.experimental.aot_kernel(static_shapes=False)
def _grouped_gemm(A_packed: torch.Tensor, W: torch.Tensor, group_offsets: torch.Tensor) -> torch.Tensor:
    total_M, K = A_packed.shape
    G, _, N = W.shape
    out = torch.empty([total_M, N], dtype=torch.promote_types(A_packed.dtype, W.dtype), device=A_packed.device)
    for g in hl.grid(G):
        start = group_offsets[g]
        end = group_offsets[g + 1]
        M_g = end - start
        if M_g != 0:
            for tile_m, tile_n in hl.tile([M_g, N]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(K):
                    a = A_packed[start + tile_m.index, tile_k]
                    w = W[g, tile_k, tile_n]
                    acc = torch.addmm(acc, a, w)
                out[start + tile_m.index, tile_n] = acc.to(out.dtype)
    return out
