import torch
from ._ops import ops
from .op import lsh_cumulation as _l


def lsh_cumulation(hash_codes, values, n_buckets):
    """YOSO LSH value cumulation: sum values into hash buckets.
    hash_codes:[B,N] int, values:[B,N,D] -> [B,n_buckets,D]."""
    return ops.lsh_cumulation(hash_codes, values, n_buckets)


__all__ = ["lsh_cumulation"]
