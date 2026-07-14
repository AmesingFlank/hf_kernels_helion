import torch
from ._ops import ops
from .op import wkv as _wkv


def wkv(w: torch.Tensor, u: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """RWKV-v4 WKV. w,u:[C]; k,v:[B,T,C] -> [B,T,C]."""
    return ops.wkv(w, u, k, v)


rwkv_linear_attention = wkv
__all__ = ["wkv", "rwkv_linear_attention"]
