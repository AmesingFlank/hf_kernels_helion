import torch
from ._ops import ops
from .op import moe_swiglu as _m


def moe_swiglu(a, wg, wu, group_offsets):
    """MoE SwiGLU expert projection: per expert, SiLU(a@Wg)*(a@Wu).
    a:[T,H] packed by expert; wg,wu:[E,H,I]; group_offsets:[E+1] -> [T,I]."""
    return ops.moe_swiglu(a, wg, wu, group_offsets)


__all__ = ["moe_swiglu"]
