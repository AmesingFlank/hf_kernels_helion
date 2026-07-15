import torch

from ._ops import ops
from .op import _gelu_and_mul_op, _gelu_tanh_and_mul_op, _silu_and_mul_op


def silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """out = silu(x[..., :d]) * x[..., d:], written into ``out`` (last dim d)."""
    ops.silu_and_mul(out, x)
    return out


def gelu_and_mul(out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """out = gelu(x[..., :d]) * x[..., d:] (exact erf gelu)."""
    ops.gelu_and_mul(out, x)
    return out


def gelu_tanh_and_mul(out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """out = gelu_tanh(x[..., :d]) * x[..., d:] (tanh approx)."""
    ops.gelu_tanh_and_mul(out, x)
    return out


__all__ = ["silu_and_mul", "gelu_and_mul", "gelu_tanh_and_mul"]
