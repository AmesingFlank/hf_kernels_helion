import torch
from ._ops import ops
from .op import sgmv as _sgmv


def add_lora_sgmv(y, x, wa, wb, s_start, s_end, lora_rank):
    """Segmented LoRA: y[s_i:e_i] += x[s_i:e_i] @ wa[i].T @ wb[i].
    x:[T,IN], wa:[nprob,rank,IN], wb:[nprob,rank,OUT]. In-place on y."""
    ops.sgmv(y, x, wa, wb, s_start, s_end, int(lora_rank))
    return y


__all__ = ["add_lora_sgmv"]
