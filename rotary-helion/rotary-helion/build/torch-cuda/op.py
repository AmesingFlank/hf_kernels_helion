from __future__ import annotations
import torch
from ._ops import add_op_namespace_prefix
from .rotary import _apply_rotary


@torch.library.custom_op(add_op_namespace_prefix("apply_rotary"), mutates_args={"out1", "out2"})
def apply_rotary(out1: torch.Tensor, out2: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor,
                 cos: torch.Tensor, sin: torch.Tensor, conj: bool) -> None:
    _apply_rotary(out1, out2, x1, x2, cos, sin, conj)


@apply_rotary.register_fake
def _(out1, out2, x1, x2, cos, sin, conj):
    pass
