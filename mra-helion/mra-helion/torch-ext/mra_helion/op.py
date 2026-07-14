from __future__ import annotations
import torch
from ._ops import add_op_namespace_prefix
from .sparse import _mm_to_sparse


@torch.library.custom_op(add_op_namespace_prefix("mm_to_sparse"), mutates_args=())
def mm_to_sparse(dense_A: torch.Tensor, dense_B: torch.Tensor, indices: torch.Tensor,
                 B_num_block: int, blk: int) -> torch.Tensor:
    return _mm_to_sparse(dense_A, dense_B, indices, B_num_block, blk)


@mm_to_sparse.register_fake
def _(dense_A, dense_B, indices, B_num_block, blk):
    return dense_A.new_empty((dense_A.shape[0], indices.shape[0], blk, blk), dtype=torch.float32)
