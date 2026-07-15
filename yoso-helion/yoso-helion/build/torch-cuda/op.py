from __future__ import annotations
import torch
from ._ops import add_op_namespace_prefix
from .lsh import _lsh_cumulation


@torch.library.custom_op(add_op_namespace_prefix("lsh_cumulation"), mutates_args=())
def lsh_cumulation(hash_codes: torch.Tensor, values: torch.Tensor, n_buckets: int) -> torch.Tensor:
    return _lsh_cumulation(hash_codes, values, n_buckets)


@lsh_cumulation.register_fake
def _(hash_codes, values, n_buckets):
    return values.new_empty((values.shape[0], n_buckets, values.shape[2]), dtype=torch.float32)
