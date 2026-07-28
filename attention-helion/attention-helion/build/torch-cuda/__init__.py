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
    # The kernel writes into ``out`` via ``out.reshape([-1, m, d])``, which is
    # only a view (not a copy) when ``out`` is contiguous — otherwise writes
    # would land in a throwaway buffer. ``torch.empty_like`` preserves the
    # (possibly non-contiguous) strides of ``q``, so allocate an explicitly
    # contiguous buffer here.
    if out is None:
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
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
    # (B, S, H, D) -> (B, H, S, D) for the SDPA-layout Helion kernels (views, no
    # copy). The kernel needs a contiguous output view to write into, so we let
    # attention() allocate a (B, H, S, D) buffer, then transpose back to
    # (B, S, H, D) and make it contiguous to match flash-attn's output layout.
    # (The final .contiguous() is inherent to the layout mismatch; callers who
    # want to avoid it should use attention() directly with (B, H, S, D).)
    out_t = attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                      causal=causal)
    return out_t.transpose(1, 2).contiguous()


__all__ = ["attention", "flash_attn_func", "layers"]
