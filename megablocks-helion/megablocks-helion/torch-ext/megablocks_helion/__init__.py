import torch
from ._ops import ops
from .op import grouped_gemm as _gg


def grouped_gemm(a: torch.Tensor, w: torch.Tensor, group_offsets: torch.Tensor) -> torch.Tensor:
    """Grouped MoE expert matmul: out[i] = a[i] @ w[expert(i)].
    a:[total_tokens, K] packed by expert; w:[num_experts, K, N]; group_offsets:[E+1]."""
    return ops.grouped_gemm(a, w, group_offsets)


def gmm(a, b, c, batch_sizes, trans_a=False, trans_b=False):
    """EXACT match to megablocks' grouped_gemm `gmm(a, b, c, batch_sizes, ...)`.

    Standard forward (trans_a=trans_b=False): a:[total_M, K] packed by group,
    b:[G, K, N], batch_sizes:[G] per-group row counts, result written into
    c:[total_M, N]. `batch_sizes`→offsets is pure-PyTorch glue; the grouped
    matmul (the CUDA/Triton hot path in the ref) is the Helion kernel.
    """
    if trans_a or trans_b:
        raise NotImplementedError("only the standard forward grouped GEMM is implemented")
    offs = torch.zeros(batch_sizes.shape[0] + 1, device=a.device, dtype=torch.int32)
    offs[1:] = torch.cumsum(batch_sizes.to(torch.int32), 0)
    out = ops.grouped_gemm(a, b, offs)
    c.copy_(out.to(c.dtype))
    return c


__all__ = ["grouped_gemm", "gmm"]
