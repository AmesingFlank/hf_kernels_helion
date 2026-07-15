from __future__ import annotations

import torch

from ._ops import add_op_namespace_prefix
from .sage_attention import sage_attn_fwd, sage_attn_fwd_causal


@torch.library.custom_op(
    add_op_namespace_prefix("sage_attention"), mutates_args={"out"}
)
def _sage_attention(
    out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    km: torch.Tensor,
    sm_scale: float,
    causal: bool,
) -> None:
    """SageAttention2 on ``(BH, S, 128)`` tensors, writing into ``out``.

    ``q/k/v`` are the fp16/bf16 inputs, ``km`` the fp32 per-head smooth-K mean.
    Dispatches to the causal or non-causal Helion kernel (Helion allocates its
    own output buffer, so we copy into ``out``).
    """
    if causal:
        result = sage_attn_fwd_causal(q, k, v, km, sm_scale)
    else:
        result = sage_attn_fwd(q, k, v, km, sm_scale)
    out.copy_(result)


@_sage_attention.register_fake
def _(
    out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    km: torch.Tensor,
    sm_scale: float,
    causal: bool,
) -> None:
    pass
