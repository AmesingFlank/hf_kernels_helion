from __future__ import annotations
import torch
from ._ops import add_op_namespace_prefix
from .fp8 import _fp8_blockwise_gemm


@torch.library.custom_op(add_op_namespace_prefix("gemm_fp8_fp8_bf16_nt"), mutates_args=())
def gemm_fp8_fp8_bf16_nt(A: torch.Tensor, B: torch.Tensor, As: torch.Tensor,
                         Bs: torch.Tensor, block_k: int) -> torch.Tensor:
    return _fp8_blockwise_gemm(A, B, As, Bs, block_k)


@gemm_fp8_fp8_bf16_nt.register_fake
def _(A, B, As, Bs, block_k):
    return A.new_empty((A.shape[0], B.shape[0]), dtype=torch.bfloat16)
