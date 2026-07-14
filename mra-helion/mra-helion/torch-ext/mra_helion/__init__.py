import torch
from ._ops import ops
from .op import mm_to_sparse as _m


def mm_to_sparse(dense_A, dense_B, indices, B_num_block, blk=32):
    """Block-sparse A@B^T at block positions in `indices` (i*B_num_block+j).
    dense_A:[B,Anb*blk,D], dense_B:[B,Bnb*blk,D] -> [B,nnz,blk,blk]."""
    return ops.mm_to_sparse(dense_A, dense_B, indices, B_num_block, blk)


__all__ = ["mm_to_sparse"]
