"""Helion SageAttention2 — INT8-quantized flash attention.

Faithful port of the algorithm in ``thu-ml/SageAttention``'s
``qk_int8_pv_fp16`` path — the ``_qattn_sm80`` CUDA kernel that ``sageattn()``
dispatches to on Ampere/Blackwell. SageAttention does every step below in
CUDA; here we write the numerics in Helion and reuse PyTorch only for the
smooth-K mean (exactly as the reference computes ``km`` in Python).

SageAttention2 algorithm
-------------------------
1. **smooth-K**: subtract the per-``(batch, head)`` sequence-mean of K,
   ``k <- k - mean_seq(k)``. This is mathematically a no-op inside softmax
   (a constant shift of every logit in a row) but it dramatically shrinks the
   INT8 quantization error of K, which is the whole point of SageAttention.
2. **INT8 quantization** of Q and K. We use *per-token* granularity (one scale
   per query/key row, ``scale = rowmax(|x|) / 127``); the reference's sm80
   kernel uses per-block/per-warp scales, which are coarser. Per-token is a
   supported SageAttention granularity and is if anything slightly more
   accurate.
3. **INT8 QK^T** via IMMA (int8 inputs, int32 accumulate), then dequantize by
   ``q_scale * k_scale`` and fold in the softmax scale.
4. **online (flash) softmax** in base-2 (``exp2``), streaming over K/V tiles
   with the usual running max ``m_i`` / denominator ``l_i`` rescaling.
5. **P @ V** with V in fp16 and FP32 accumulation (``pv_accum_dtype="fp32"``).

Inputs to the kernel are reshaped to ``(BH, S, D)`` with ``D == 128`` (the
SageAttention2 head dim). ``km`` is ``(BH, 1, D)`` fp32.
"""

from __future__ import annotations

import math

import torch

import helion
import helion.experimental
import helion.language as hl


def _sdpa_baseline(q, k, v, *, causal):
    # Full-precision baseline for the autotuner to check candidate configs
    # against. SageAttention is an INT8 approximation of SDPA, so we give the
    # autotuner generous tolerances (the quant error floor is ~1% relative).
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)


@helion.experimental.aot_kernel(
    static_shapes=True,
    autotune_baseline_fn=lambda q, k, v, km, s: _sdpa_baseline(
        q.reshape(-1, q.size(-2), q.size(-1)),
        k.reshape(-1, k.size(-2), k.size(-1)),
        v.reshape(-1, v.size(-2), v.size(-1)),
        causal=False,
    ),
    autotune_baseline_atol=5e-2,
    autotune_baseline_rtol=5e-2,
)
def sage_attn_fwd(
    q: torch.Tensor,   # (BH, M, D) fp16/bf16
    k: torch.Tensor,   # (BH, N, D) fp16/bf16
    v: torch.Tensor,   # (BH, N, D) fp16/bf16
    km: torch.Tensor,  # (BH, 1, D) fp32  (per-head mean of K over the seq dim)
    sm_scale: float,
) -> torch.Tensor:
    BH, M, D = q.shape
    N = k.size(1)
    D = hl.specialize(D)
    out = torch.empty_like(q)
    qk_scale = sm_scale * 1.44269504  # fold 1/ln(2) so we can use exp2 in softmax
    for bh in hl.grid(BH):
        km_row = km[bh, 0, :].to(torch.float32)[None, :]  # (1, D)
        for tile_m in hl.tile(M):
            q_f = q[bh, tile_m, :].to(torch.float32)
            # per-token INT8 quant of Q: one scale per query row (reduce over D).
            q_scale = (torch.amax(torch.abs(q_f), dim=-1) + 1e-20) / 127.0  # (tile_m,)
            q_i8 = torch.round(q_f / q_scale[:, None]).to(torch.int8)

            m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
            l_i = hl.zeros([tile_m], dtype=torch.float32)
            acc = hl.zeros([tile_m, D], dtype=torch.float32)
            for tile_n in hl.tile(N):
                k_f = k[bh, tile_n, :].to(torch.float32) - km_row  # smooth-K
                # per-token INT8 quant of K: one scale per key row.
                k_scale = (torch.amax(torch.abs(k_f), dim=-1) + 1e-20) / 127.0  # (tile_n,)
                k_i8 = torch.round(k_f / k_scale[:, None]).to(torch.int8)

                # INT8 QK^T -> int32, dequant by the per-row Q/K scales.
                qk_i32 = hl.dot(q_i8, k_i8.T, out_dtype=torch.int32)
                qk = qk_i32.to(torch.float32) * (q_scale[:, None] * k_scale[None, :]) * qk_scale

                m_ij = torch.maximum(m_i, torch.amax(qk, -1))
                qk = qk - m_ij[:, None]
                p = torch.exp2(qk)
                l_ij = torch.sum(p, -1)
                alpha = torch.exp2(m_i - m_ij)
                l_i = l_i * alpha + l_ij
                acc = acc * alpha[:, None]
                v_f = v[bh, tile_n, :]
                acc = hl.dot(p.to(v_f.dtype), v_f, acc=acc)  # P(fp16) @ V(fp16), fp32 accum
                m_i = m_ij
            acc = acc / l_i[:, None]
            out[bh, tile_m, :] = acc.to(out.dtype)
    return out


@helion.experimental.aot_kernel(
    static_shapes=True,
    autotune_baseline_fn=lambda q, k, v, km, s: _sdpa_baseline(
        q.reshape(-1, q.size(-2), q.size(-1)),
        k.reshape(-1, k.size(-2), k.size(-1)),
        v.reshape(-1, v.size(-2), v.size(-1)),
        causal=True,
    ),
    autotune_baseline_atol=5e-2,
    autotune_baseline_rtol=5e-2,
)
def sage_attn_fwd_causal(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    km: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    BH, M, D = q.shape
    N = k.size(1)
    D = hl.specialize(D)
    out = torch.empty_like(q)
    qk_scale = sm_scale * 1.44269504
    for bh in hl.grid(BH):
        km_row = km[bh, 0, :].to(torch.float32)[None, :]
        for tile_m in hl.tile(M):
            q_f = q[bh, tile_m, :].to(torch.float32)
            q_scale = (torch.amax(torch.abs(q_f), dim=-1) + 1e-20) / 127.0
            q_i8 = torch.round(q_f / q_scale[:, None]).to(torch.int8)

            m_i = hl.full([tile_m], float("-inf"), dtype=torch.float32)
            l_i = hl.zeros([tile_m], dtype=torch.float32)
            acc = hl.zeros([tile_m, D], dtype=torch.float32)
            for tile_n in hl.tile(N):
                k_f = k[bh, tile_n, :].to(torch.float32) - km_row
                k_scale = (torch.amax(torch.abs(k_f), dim=-1) + 1e-20) / 127.0
                k_i8 = torch.round(k_f / k_scale[:, None]).to(torch.int8)

                qk_i32 = hl.dot(q_i8, k_i8.T, out_dtype=torch.int32)
                qk = qk_i32.to(torch.float32) * (q_scale[:, None] * k_scale[None, :]) * qk_scale
                # causal mask: query row index >= key col index
                qk = torch.where(
                    tile_m.index[:, None] >= tile_n.index[None, :], qk, float("-inf")
                )
                m_ij = torch.maximum(m_i, torch.amax(qk, -1))
                qk = qk - m_ij[:, None]
                p = torch.exp2(qk)
                l_ij = torch.sum(p, -1)
                alpha = torch.exp2(m_i - m_ij)
                l_i = l_i * alpha + l_ij
                acc = acc * alpha[:, None]
                v_f = v[bh, tile_n, :]
                acc = hl.dot(p.to(v_f.dtype), v_f, acc=acc)
                m_i = m_ij
            acc = acc / l_i[:, None]
            out[bh, tile_m, :] = acc.to(out.dtype)
    return out


def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """SageAttention2 forward — matches ``sageattention.sageattn`` semantics.

    Args:
        q, k, v: fp16/bf16 tensors, head_dim == 128.
            - ``tensor_layout="HND"``: ``(batch, heads, seq, 128)``
            - ``tensor_layout="NHD"``: ``(batch, seq, heads, 128)``
        is_causal: apply a causal mask (only valid when ``seq_q == seq_k``).
        sm_scale: softmax scale; defaults to ``1/sqrt(head_dim)``.

    Returns:
        Output tensor with the same shape/layout as ``q``.
    """
    if tensor_layout == "NHD":
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
    elif tensor_layout != "HND":
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    B, H, S, D = q.shape
    if D != 128:
        raise ValueError(f"SageAttention2 requires head_dim == 128, got {D}")
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    qf = q.reshape(B * H, S, D).contiguous()
    kf = k.reshape(B * H, k.size(2), D).contiguous()
    vf = v.reshape(B * H, v.size(2), D).contiguous()
    km = kf.mean(dim=1, keepdim=True).to(torch.float32)  # smooth-K mean (PyTorch, reused)

    kern = sage_attn_fwd_causal if is_causal else sage_attn_fwd
    o = kern(qf, kf, vf, km, sm_scale)
    o = o.reshape(B, H, S, D)

    if tensor_layout == "NHD":
        o = o.transpose(1, 2).contiguous()
    return o
