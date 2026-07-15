from typing import Optional
import torch
from ._ops import ops
from .op import selective_scan_fn as _ss


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                      delta_softplus=False, return_last_state=False):
    """Mamba selective scan. u,delta:[B,D,L], A:[D,N], B,C:[B,N,L], D:[D], z:[B,D,L].
    (Only the common B/C = [B,N,L] layout is supported; groups/complex not handled.)"""
    if delta_bias is not None:
        delta = delta + delta_bias[None, :, None].to(delta.dtype)
    if D is None:
        D = torch.zeros(u.shape[1], device=u.device, dtype=torch.float32)
    use_z = z is not None
    if z is None:
        z = u
    return ops.selective_scan_fn(u, delta, A, B, C, D.float(), z, delta_softplus, use_z)


__all__ = ["selective_scan_fn"]
