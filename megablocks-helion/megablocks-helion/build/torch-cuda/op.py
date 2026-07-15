from __future__ import annotations
import torch
from ._ops import add_op_namespace_prefix
from .ggemm import _grouped_gemm


@torch.library.custom_op(add_op_namespace_prefix("grouped_gemm"), mutates_args=())
def grouped_gemm(a: torch.Tensor, w: torch.Tensor, group_offsets: torch.Tensor) -> torch.Tensor:
    return _grouped_gemm(a, w, group_offsets)


@grouped_gemm.register_fake
def _(a: torch.Tensor, w: torch.Tensor, group_offsets: torch.Tensor) -> torch.Tensor:
    return a.new_empty((a.shape[0], w.shape[2]))
