from __future__ import annotations
import torch
from ._ops import add_op_namespace_prefix
from .deform import _deform_attn


@torch.library.custom_op(add_op_namespace_prefix("ms_deform_attn"), mutates_args=())
def ms_deform_attn(value: torch.Tensor, sampling_loc: torch.Tensor, attn_weight: torch.Tensor,
                   Hgrid: int, Wgrid: int, Dh: int, P: int, H: int) -> torch.Tensor:
    return _deform_attn(value, sampling_loc, attn_weight, Hgrid, Wgrid, Dh, P, H)


@ms_deform_attn.register_fake
def _(value, sampling_loc, attn_weight, Hgrid, Wgrid, Dh, P, H):
    B = value.shape[0]; Nq = sampling_loc.shape[1]
    return value.new_empty((B, Nq, H, Dh), dtype=torch.float32)
