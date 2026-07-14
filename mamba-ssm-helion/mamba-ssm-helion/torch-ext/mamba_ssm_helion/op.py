from __future__ import annotations
import torch
from ._ops import add_op_namespace_prefix
from .scan import _selective_scan


@torch.library.custom_op(add_op_namespace_prefix("selective_scan_fn"), mutates_args=())
def selective_scan_fn(
    u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
    C: torch.Tensor, D_: torch.Tensor, z: torch.Tensor,
    delta_softplus: bool, use_z: bool,
) -> torch.Tensor:
    return _selective_scan(u, delta, A, B, C, D_, z, delta_softplus, use_z)


@selective_scan_fn.register_fake
def _(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
      C: torch.Tensor, D_: torch.Tensor, z: torch.Tensor,
      delta_softplus: bool, use_z: bool) -> torch.Tensor:
    return torch.empty_like(u)
