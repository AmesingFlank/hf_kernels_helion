from __future__ import annotations

import torch

from ._ops import add_op_namespace_prefix
from .nf4 import NF4_CODE, _nf4_gemm

# quant_type: 0 = FP4, 1 = NF4 (bitsandbytes convention). We implement NF4.
_NF4_TENSOR = None


def _get_code(device, dtype=torch.float32):
    global _NF4_TENSOR
    if _NF4_TENSOR is None or _NF4_TENSOR.device != device:
        _NF4_TENSOR = torch.tensor(NF4_CODE, device=device, dtype=dtype)
    return _NF4_TENSOR


@torch.library.custom_op(add_op_namespace_prefix("gemm_4bit_forward"), mutates_args=())
def gemm_4bit_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    absmax: torch.Tensor,
    blocksize: int,
    quant_type: int,
) -> torch.Tensor:
    """4-bit (NF4) weight-only GEMM: out = input @ dequant(weight, absmax).

    ``weight`` is [N, K//2] packed uint8; ``absmax`` is [N, K//blocksize].
    Returns [M, N] in the input dtype.
    """
    orig_dtype = input.dtype
    inp = input.to(torch.bfloat16)
    code = _get_code(input.device)
    out = _nf4_gemm(inp, weight, absmax.to(torch.float32), code, blocksize)
    return out.to(orig_dtype)


@gemm_4bit_forward.register_fake
def _(input, weight, absmax, blocksize, quant_type):
    M = input.shape[0]
    N = weight.shape[0]
    return input.new_empty((M, N))
