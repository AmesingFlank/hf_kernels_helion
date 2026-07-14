"""Helion SageAttention2 — INT8-quantized flash attention.

Public API mirrors ``thu-ml/SageAttention``'s ``sageattn`` entry point plus a
flash-attn-style ``flash_attn_func`` wrapper. See ``sage_attention.py`` for the
algorithm (smooth-K, per-token INT8 quant of Q/K, INT8 QK^T, fp16 P@V).
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from . import op  # noqa: F401  (registers the custom op)
from .sage_attention import sageattn

__all__ = ["sageattn", "flash_attn_func"]


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    **kwargs,
) -> torch.Tensor:
    """flash-attn-compatible entry point (``(B, S, H, D)`` layout, D == 128).

    Adapts the flash-attn ``(batch, seq, heads, head_dim)`` convention to
    SageAttention2 and returns the output in the same layout.
    """
    return sageattn(
        q, k, v, tensor_layout="NHD", is_causal=causal, sm_scale=softmax_scale
    )
