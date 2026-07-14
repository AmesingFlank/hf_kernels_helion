from typing import Optional
import torch
from ._ops import ops
from .op import causal_conv1d_fn as _ccf


def causal_conv1d_fn(x, weight, bias=None, seq_idx=None, initial_states=None,
                     return_final_states=False, final_states_out=None, activation=None):
    """Depthwise causal 1D conv. x:(B,D,L), weight:(D,width), bias:(D,).
    activation in {None,'silu','swish'}. (seq_idx / states variants not supported.)"""
    if bias is None:
        bias = torch.zeros(x.shape[1], device=x.device, dtype=x.dtype)
    silu = activation in ("silu", "swish")
    return ops.causal_conv1d_fn(x, weight, bias, silu)


__all__ = ["causal_conv1d_fn"]
