from typing import List, Optional
import torch
from ._ops import ops
from .op import w8a8_block_fp8_matmul as _m


def w8a8_block_fp8_matmul(A: torch.Tensor, B: torch.Tensor, As: torch.Tensor,
                          Bs: torch.Tensor, block_size, output_dtype=torch.float32):
    """EXACT match to kernels-community/finegrained-fp8 w8a8_block_fp8_matmul.
    C = A @ B.T ; A:[M,K] fp8, B:[N,K] fp8, As:[M,K//bk], Bs:[N//bn,K//bk],
    block_size=[block_n, block_k]. Returns [M,N] in output_dtype."""
    block_n, block_k = int(block_size[0]), int(block_size[1])
    out = ops.w8a8_block_fp8_matmul(A, B, As, Bs, block_n, block_k)
    return out.to(output_dtype)


__all__ = ["w8a8_block_fp8_matmul"]
