import torch
from ._ops import ops
from .op import ms_deform_attn as _m


def ms_deform_attn(value, sampling_loc, attn_weight):
    """Single-level multiscale deformable attention.
    value:[B,Nkv=Hg*Wg,H,Dh], sampling_loc:[B,Nq,H,P,2] in [0,1], attn_weight:[B,Nq,H,P].
    Grid is assumed square (Hg=Wg=sqrt(Nkv))."""
    import math
    B, Nkv, H, Dh = value.shape
    P = sampling_loc.shape[3]
    Hg = int(math.isqrt(Nkv)); Wg = Nkv // Hg
    return ops.ms_deform_attn(value, sampling_loc, attn_weight, Hg, Wg, Dh, P, H)


__all__ = ["ms_deform_attn"]
