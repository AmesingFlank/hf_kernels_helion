import torch
from ._ops import ops
from .op import w8_a16_gemm as _g


def w8_a16_gemm(x, weight, scale):
    """int8 weight-only GEMM: x @ (weight*scale)^T. x:[M,K] fp16, weight:[N,K] int8, scale:[N]."""
    return ops.w8_a16_gemm(x, weight, scale)


__all__ = ["w8_a16_gemm"]
