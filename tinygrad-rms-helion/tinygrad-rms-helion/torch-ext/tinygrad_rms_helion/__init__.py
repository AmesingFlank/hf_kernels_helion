from typing import Optional
import torch
from ._ops import ops
from .op import tinygrad_rms_norm as _t


def tinygrad_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS norm: x * rsqrt(mean(x^2)+eps) * weight, over the last dim."""
    return ops.tinygrad_rms_norm(x, weight, eps)


tinygrad_rms_norm_simple = tinygrad_rms_norm
__all__ = ["tinygrad_rms_norm", "tinygrad_rms_norm_simple"]
