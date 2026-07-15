import torch
from ._ops import ops
from .op import apply_rotary as _ar


def apply_rotary(x1, x2, cos, sin, out1=None, out2=None, conj=False):
    """Rotary embedding. x1,x2:[B,S,H,R], cos,sin:[S,1,R]. Writes/returns (out1,out2)."""
    if out1 is None:
        out1 = torch.empty_like(x1)
    if out2 is None:
        out2 = torch.empty_like(x2)
    ops.apply_rotary(out1, out2, x1, x2, cos, sin, conj)
    return out1, out2


__all__ = ["apply_rotary"]
