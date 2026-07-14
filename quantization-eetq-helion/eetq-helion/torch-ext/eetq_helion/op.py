from __future__ import annotations
import torch
from ._ops import add_op_namespace_prefix
from .gemm import _w8_a16_gemm


@torch.library.custom_op(add_op_namespace_prefix("w8_a16_gemm"), mutates_args=())
def w8_a16_gemm(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return _w8_a16_gemm(x, weight, scale)


@w8_a16_gemm.register_fake
def _(x, weight, scale):
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.float16)
