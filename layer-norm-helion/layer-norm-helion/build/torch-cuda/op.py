from __future__ import annotations
import torch
from ._ops import add_op_namespace_prefix
from .rms import _rms_norm


@torch.library.custom_op(add_op_namespace_prefix("rms_norm_fwd"), mutates_args=())
def rms_norm_fwd(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return _rms_norm(x, weight, eps)


@rms_norm_fwd.register_fake
def _(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.empty_like(x)
