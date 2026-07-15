import torch
import torch.nn as nn

from ._ops import ops


class Attention(nn.Module):
    """Scaled dot-product attention on ``(B, H, S, D)`` tensors."""

    def __init__(self, causal: bool = False) -> None:
        super().__init__()
        self.causal = causal

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        out = torch.empty_like(q)
        ops.attention(out, q, k, v, self.causal)
        return out
