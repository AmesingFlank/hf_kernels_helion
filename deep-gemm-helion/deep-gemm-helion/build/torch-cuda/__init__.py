import torch
from ._ops import ops
from .op import gemm_fp8_fp8_bf16_nt as _m


def gemm_fp8_fp8_bf16_nt(A, B, As, Bs, block_size=128):
    """DeepGEMM-style blockwise fp8 GEMM (NT layout): (A*As) @ (B*Bs)^T -> bf16."""
    return ops.gemm_fp8_fp8_bf16_nt(A, B, As, Bs, block_size)


__all__ = ["gemm_fp8_fp8_bf16_nt"]
