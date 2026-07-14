from typing import Optional
import torch
from ._ops import ops
from .op import rms_norm_fwd as _r


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """RMS norm over last dim (2-D [rows, D] input)."""
    return ops.rms_norm_fwd(x, weight, eps)


def dropout_add_ln_fwd(input, gamma, beta=None, rowscale=None, colscale=None,
                       x0_subset=None, z_subset=None, dropout_p=0.0, epsilon=1e-5,
                       rowscale_const=1.0, z_numrows=0, gen=None,
                       residual_in_fp32=False, is_rms_norm=True, **kwargs):
    """Signature-compatible with kernels-community/layer-norm dropout_add_ln_fwd.
    Only the common path (dropout_p=0, RMSNorm, no residual/subset) is supported.
    Returns a tuple whose first element is the normalized output."""
    assert is_rms_norm, "only RMSNorm path implemented"
    out = ops.rms_norm_fwd(input, gamma, float(epsilon))
    return (out, None, None, None)


__all__ = ["rms_norm", "dropout_add_ln_fwd"]
