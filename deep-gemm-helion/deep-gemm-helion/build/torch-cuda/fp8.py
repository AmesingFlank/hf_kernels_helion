"""Helion blockwise-scaled FP8 GEMM (DeepSeek / deep-gemm / finegrained-fp8 style).

A:[M,K] fp8e4m3 with per-block row scales sA:[M,K//bk]; B:[N,K] fp8e4m3 with
sB:[N,K//bk]. out = (A·sA) @ (B·sB)ᵀ in bf16. Scales applied per K-block inside
the contraction loop (fine-grained quantization).
"""
from __future__ import annotations
import torch
import helion
import helion.language as hl


@helion.kernel(static_shapes=False)
def _fp8_blockwise_gemm(A: torch.Tensor, B: torch.Tensor, sA: torch.Tensor,
                        sB: torch.Tensor, bk: hl.constexpr) -> torch.Tensor:
    M, K = A.shape
    N, _ = B.shape
    out = torch.empty([M, N], dtype=torch.bfloat16, device=A.device)
    for tm, tn in hl.tile([M, N]):
        acc = hl.zeros([tm, tn], dtype=torch.float32)
        for tk in hl.tile(K, block_size=bk):
            a = A[tm, tk].to(torch.float32)
            b = B[tn, tk].to(torch.float32)
            kb = tk.begin // bk
            sa = sA[tm, kb]
            sb = sB[tn, kb]
            acc = acc + hl.dot(a, b.T) * sa[:, None] * sb[None, :]
        out[tm, tn] = acc.to(torch.bfloat16)
    return out
