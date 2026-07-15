from __future__ import annotations
import torch
from ._ops import add_op_namespace_prefix
from .wkv import _rwkv_wkv


@torch.library.custom_op(add_op_namespace_prefix("wkv"), mutates_args=())
def wkv(w: torch.Tensor, u: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return _rwkv_wkv(w, u, k, v)


@wkv.register_fake
def _(w: torch.Tensor, u: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(v)
