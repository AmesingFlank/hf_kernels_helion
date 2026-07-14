from __future__ import annotations
import torch
from ._ops import add_op_namespace_prefix
from .sgmv import _sgmv


@torch.library.custom_op(add_op_namespace_prefix("sgmv"), mutates_args={"y"})
def sgmv(y: torch.Tensor, x: torch.Tensor, wa: torch.Tensor, wb: torch.Tensor,
         s_start: torch.Tensor, s_end: torch.Tensor, rank: int) -> None:
    _sgmv(y, x, wa, wb, s_start, s_end, rank)


@sgmv.register_fake
def _(y, x, wa, wb, s_start, s_end, rank):
    pass
