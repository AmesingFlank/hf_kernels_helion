from __future__ import annotations

import torch

from ._ops import add_op_namespace_prefix
from .act import _gelu_and_mul, _gelu_tanh_and_mul, _silu_and_mul


def _flat2d(out: torch.Tensor, x: torch.Tensor):
    """Flatten leading dims so the Helion kernels see 2-D [rows, d] / [rows, 2d]."""
    d = out.shape[-1]
    return out.reshape(-1, d), x.reshape(-1, x.shape[-1])


@torch.library.custom_op(add_op_namespace_prefix("silu_and_mul"), mutates_args={"out"})
def _silu_and_mul_op(out: torch.Tensor, x: torch.Tensor) -> None:
    o2, x2 = _flat2d(out, x)
    _silu_and_mul(o2, x2)


@_silu_and_mul_op.register_fake
def _(out: torch.Tensor, x: torch.Tensor) -> None:
    pass


@torch.library.custom_op(add_op_namespace_prefix("gelu_and_mul"), mutates_args={"out"})
def _gelu_and_mul_op(out: torch.Tensor, x: torch.Tensor) -> None:
    o2, x2 = _flat2d(out, x)
    _gelu_and_mul(o2, x2)


@_gelu_and_mul_op.register_fake
def _(out: torch.Tensor, x: torch.Tensor) -> None:
    pass


@torch.library.custom_op(
    add_op_namespace_prefix("gelu_tanh_and_mul"), mutates_args={"out"}
)
def _gelu_tanh_and_mul_op(out: torch.Tensor, x: torch.Tensor) -> None:
    o2, x2 = _flat2d(out, x)
    _gelu_tanh_and_mul(o2, x2)


@_gelu_tanh_and_mul_op.register_fake
def _(out: torch.Tensor, x: torch.Tensor) -> None:
    pass
