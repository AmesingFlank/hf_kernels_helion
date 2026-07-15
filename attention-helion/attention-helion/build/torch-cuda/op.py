from __future__ import annotations

import torch

from ._ops import add_op_namespace_prefix
from .attention import attention_output, causal_attention_output


@torch.library.custom_op(
    add_op_namespace_prefix("attention"), mutates_args={"out"}
)
def _attention(
    out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
) -> None:
    """Scaled dot-product attention on ``(B, H, S, D)`` tensors, in place.

    Dispatches to the causal or non-causal Helion kernel and copies the
    result into ``out`` (Helion allocates its own output buffer).
    """
    if causal:
        result = causal_attention_output(q, k, v)
    else:
        result = attention_output(q, k, v)
    out.copy_(result)


@_attention.register_fake
def _(
    out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
) -> None:
    pass
