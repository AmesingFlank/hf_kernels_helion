"""Helion block-sparse matmul — the core of MRA (multi-resolution attention).

mm_to_sparse computes A@B^T only at the block positions listed in `indices`
(32x32 blocks), the sparse-attention primitive MRA uses. A:[B, Anb*32, D],
Bm:[B, Bnb*32, D], indices:[nnz] flat block ids (i*Bnb+j) -> out:[B, nnz, 32, 32].
Data-dependent block slice via base*blk + arange (Helion gather pattern).
"""
from __future__ import annotations
import torch
import helion
import helion.language as hl


@helion.kernel(static_shapes=False)
def _mm_to_sparse(A: torch.Tensor, Bm: torch.Tensor, indices: torch.Tensor,
                  B_num_block: int, blk: hl.constexpr) -> torch.Tensor:
    Bsz, _, D = A.shape
    nnz = indices.shape[0]
    out = torch.empty([Bsz, nnz, blk, blk], dtype=torch.float32, device=A.device)
    for tbatch in hl.grid(Bsz):
        for tn in hl.grid(nnz):
            idx = indices[tn]
            bi = idx // B_num_block
            bj = idx % B_num_block
            rows = bi * blk + torch.arange(0, blk, device=A.device)
            cols = bj * blk + torch.arange(0, blk, device=A.device)
            acc = hl.zeros([blk, blk], dtype=torch.float32)
            for tk in hl.tile(D):
                a = A[tbatch, rows, tk].to(torch.float32)
                b = Bm[tbatch, cols, tk].to(torch.float32)
                acc = hl.dot(a, b.T, acc=acc)
            out[tbatch, tn, :, :] = acc
    return out
