from __future__ import annotations
import torch
from ._ops import add_op_namespace_prefix
from .conv import _causal_conv1d


@torch.library.custom_op(add_op_namespace_prefix("causal_conv1d_fn"), mutates_args=())
def causal_conv1d_fn(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, silu: bool) -> torch.Tensor:
    return _causal_conv1d(x, weight, bias, silu)


@causal_conv1d_fn.register_fake
def _(x, weight, bias, silu):
    return torch.empty_like(x)
