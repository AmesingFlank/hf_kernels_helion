from __future__ import annotations
import torch
from ._ops import add_op_namespace_prefix
from .paged import _paged_attention_v1


@torch.library.custom_op(add_op_namespace_prefix("paged_attention_v1"), mutates_args={"out"})
def paged_attention_v1(out: torch.Tensor, query: torch.Tensor, key_cache: torch.Tensor,
                       value_cache: torch.Tensor, block_tables: torch.Tensor, seq_lens: torch.Tensor,
                       scale: float, num_kv_heads: int, block_size: int, max_blocks: int) -> None:
    _paged_attention_v1(out, query, key_cache, value_cache, block_tables, seq_lens,
                        scale, num_kv_heads, block_size, max_blocks)


@paged_attention_v1.register_fake
def _(out, query, key_cache, value_cache, block_tables, seq_lens, scale, num_kv_heads, block_size, max_blocks):
    pass
