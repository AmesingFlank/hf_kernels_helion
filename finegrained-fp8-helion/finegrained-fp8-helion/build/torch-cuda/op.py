from __future__ import annotations
from typing import List
import torch
from ._ops import add_op_namespace_prefix
from .fp8 import _w8a8_block_fp8_matmul


@torch.library.custom_op(add_op_namespace_prefix("w8a8_block_fp8_matmul"), mutates_args=())
def w8a8_block_fp8_matmul(A: torch.Tensor, B: torch.Tensor, As: torch.Tensor,
                          Bs: torch.Tensor, block_n: int, block_k: int) -> torch.Tensor:
    return _w8a8_block_fp8_matmul(A, B, As, Bs, block_n, block_k)


@w8a8_block_fp8_matmul.register_fake
def _(A, B, As, Bs, block_n, block_k):
    return A.new_empty((A.shape[0], B.shape[0]), dtype=torch.float32)
