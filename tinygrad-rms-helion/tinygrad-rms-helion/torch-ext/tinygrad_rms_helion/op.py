from __future__ import annotations
import torch
from ._ops import add_op_namespace_prefix
from .rms import _rms_norm


@torch.library.custom_op(add_op_namespace_prefix("tinygrad_rms_norm"), mutates_args=())
def tinygrad_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    orig = x.shape
    out = _rms_norm(x.reshape(-1, orig[-1]), weight, eps)
    return out.reshape(orig)


@tinygrad_rms_norm.register_fake
def _(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.empty_like(x)
