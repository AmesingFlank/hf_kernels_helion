from typing import Optional
import torch
from ._ops import ops
from .op import paged_attention_v1 as _pa


def paged_attention_v1(out, query, key_cache, value_cache, num_kv_heads, scale,
                       block_tables, seq_lens, block_size, max_seq_len,
                       alibi_slopes=None, kv_cache_dtype="auto", k_scale=1.0, v_scale=1.0,
                       *args, **kwargs):
    """EXACT signature match to kernels-community/paged-attention.paged_attention_v1.

    Accepts the vLLM **x-split** key cache `[nb, KVH, D//x, block_size, x]` and
    value cache `[nb, KVH, D, block_size]`. The x-split is a pure memory-layout
    detail, so we un-split the key cache to `[nb, KVH, D, block_size]` in PyTorch
    (reuse) and run the actual flash-decode attention in the Helion kernel (the
    part the reference writes in CUDA).
    """
    if key_cache.dim() == 5:  # vLLM x-split -> logical [nb, KVH, D, block_size]
        nb, kvh, d_over_x, bs, x = key_cache.shape
        key_cache = key_cache.permute(0, 1, 2, 4, 3).reshape(nb, kvh, d_over_x * x, bs).contiguous()
    max_blocks = block_tables.shape[1]
    ops.paged_attention_v1(out, query, key_cache, value_cache, block_tables, seq_lens,
                           float(scale), int(num_kv_heads), int(block_size), int(max_blocks))
    return out


__all__ = ["paged_attention_v1"]
