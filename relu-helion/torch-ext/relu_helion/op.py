from __future__ import annotations

import torch

import helion
import helion.language as hl

from ._ops import add_op_namespace_prefix


@helion.kernel(static_shapes=False)
def _relu_helion(out: torch.Tensor, x: torch.Tensor) -> None:
    """Element-wise ReLU, writing the result into ``out`` in place."""
    for tile in hl.tile(x.size()):
        out[tile] = torch.relu(x[tile])


@torch.library.custom_op(add_op_namespace_prefix("relu"), mutates_args={"out"})
def _relu(out: torch.Tensor, x: torch.Tensor) -> None:
    _relu_helion(out, x)


@_relu.register_fake
def _(out: torch.Tensor, x: torch.Tensor) -> None:
    pass
