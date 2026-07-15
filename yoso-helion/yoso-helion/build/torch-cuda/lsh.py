"""Helion LSH cumulation — the core aggregation of YOSO attention.

Aggregate token values into hash buckets:
  out[b, bucket, :] = Σ_{i : hash_codes[b,i]==bucket} values[b, i, :]
via atomic scatter-add. This is the value-cumulation primitive YOSO uses to
approximate softmax attention over LSH buckets.
"""
from __future__ import annotations
import torch
import helion
import helion.language as hl


@helion.kernel(static_shapes=False)
def _lsh_cumulation(hash_codes: torch.Tensor, values: torch.Tensor, n_buckets: int) -> torch.Tensor:
    B, N, D = values.shape
    out = torch.zeros([B, n_buckets, D], dtype=torch.float32, device=values.device)
    for b in hl.grid(B):
        for tn in hl.tile(N):
            h = hash_codes[b, tn]
            v = values[b, tn, :].to(torch.float32)
            hl.atomic_add(out, [b, h, slice(None)], v)
    return out
