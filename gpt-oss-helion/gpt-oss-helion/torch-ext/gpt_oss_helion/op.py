from __future__ import annotations
import torch
from ._ops import add_op_namespace_prefix
from .moe import _moe_swiglu


@torch.library.custom_op(add_op_namespace_prefix("moe_swiglu"), mutates_args=())
def moe_swiglu(a: torch.Tensor, wg: torch.Tensor, wu: torch.Tensor,
               group_offsets: torch.Tensor) -> torch.Tensor:
    return _moe_swiglu(a, wg, wu, group_offsets)


@moe_swiglu.register_fake
def _(a, wg, wu, group_offsets):
    return a.new_empty((a.shape[0], wg.shape[2]))
