"""Helion paged attention (decode), matching kernels-community/paged-attention.

Decode-phase attention where each sequence's KV is stored in a paged cache
indexed by a block table. Flash-attention online softmax over the gathered
blocks. Supports GQA (num_heads a multiple of num_kv_heads).

Layout (vLLM-style, simplified single-x key cache):
  query:       [num_seqs, num_heads, head_size]
  key_cache:   [num_blocks, num_kv_heads, head_size, block_size]
  value_cache: [num_blocks, num_kv_heads, head_size, block_size]
  block_tables:[num_seqs, max_num_blocks_per_seq]  int32
  seq_lens:    [num_seqs]                           int32
"""

from __future__ import annotations

import torch

import helion
import helion.experimental
import helion.language as hl

_LOG2E = 1.4426950408889634


@helion.experimental.aot_kernel(static_shapes=False)
def _paged_attention_v1(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    num_kv_heads: int,
    block_size: hl.constexpr,
    max_blocks: hl.constexpr,
) -> None:
    num_seqs, num_heads, head_size = query.shape
    head_size = hl.specialize(head_size)
    q_per_kv = num_heads // num_kv_heads
    for ts, th in hl.tile([num_seqs, num_heads]):
        q = query[ts, th, :].to(torch.float32)
        kv_h = th.index // q_per_kv  # GQA head mapping
        m_i = hl.full([ts, th], float("-inf"), dtype=torch.float32)
        l_i = hl.zeros([ts, th], dtype=torch.float32)
        acc = hl.zeros([ts, th, head_size], dtype=torch.float32)
        slen = seq_lens[ts]
        for blk in range(max_blocks):
            phys = block_tables[ts, blk]
            k = key_cache[phys, kv_h, :, :].to(torch.float32)
            v = value_cache[phys, kv_h, :, :].to(torch.float32)
            scores = torch.sum(q[:, :, :, None] * k, dim=2) * scale
            pos = blk * block_size + torch.arange(0, block_size, device=q.device)
            valid = pos[None, None, :] < slen[:, None, None]
            scores = torch.where(valid, scores, float("-inf"))
            m_new = torch.maximum(m_i, torch.amax(scores, dim=2))
            p = torch.exp2((scores - m_new[:, :, None]) * _LOG2E)
            alpha = torch.exp2((m_i - m_new) * _LOG2E)
            l_i = l_i * alpha + torch.sum(p, dim=2)
            acc = acc * alpha[:, :, None] + torch.sum(p[:, :, None, :] * v, dim=3)
            m_i = m_new
        out[ts, th, :] = (acc / l_i[:, :, None]).to(out.dtype)
