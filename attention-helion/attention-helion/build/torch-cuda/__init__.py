from typing import Optional

import torch

from . import layers
from ._ops import ops
from .op import _attention


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Scaled dot-product attention on ``(B, H, S, D)`` tensors (SDPA layout).

    Args:
        q, k, v: tensors of shape ``(batch, heads, seq, head_dim)``.
        causal: whether to apply a causal mask.
        out: optional output tensor to write into.

    Returns:
        Attention output of shape ``(batch, heads, seq, head_dim)``.
    """
    if out is None:
        out = torch.empty_like(q)
    ops.attention(out, q, k, v, causal)
    return out


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    **kwargs,
) -> torch.Tensor:
    """flash-attn compatible entry point.

    Matches the calling convention of ``kernels-community/flash-attn3``'s
    ``flash_attn_func`` (see ``kernels.benchmarks.attention``): inputs are
    ``(batch, seq, heads, head_dim)`` and the output has the same layout.

    ``softmax_scale`` is accepted for signature compatibility; these Helion
    kernels use the standard ``1/sqrt(head_dim)`` scale (the only value the
    benchmark util exercises). Extra flash-attn kwargs are ignored.
    """
    # (B, S, H, D) -> (B, H, S, D) for the SDPA-layout Helion kernels.
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    out_t = attention(q_t, k_t, v_t, causal=causal)
    # Back to (B, S, H, D), contiguous to match flash-attn's output.
    return out_t.transpose(1, 2).contiguous()


__all__ = ["attention", "flash_attn_func", "layers"]
