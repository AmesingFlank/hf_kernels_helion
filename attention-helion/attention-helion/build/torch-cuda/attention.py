"""Helion scaled dot-product attention kernels.

Ported (near-verbatim) from the upstream Helion example
``~/helion/examples/attention.py``. Only the forward kernels needed for a
flash-attention-style API are kept: the non-causal and causal variants that
return just the attention output.

Input layout for these kernels is ``[..., seq, head_dim]`` (i.e. the usual
SDPA ``(B, H, S, D)`` convention). The public ``flash_attn_func`` wrapper in
``__init__.py`` adapts the flash-attn ``(B, S, H, D)`` layout to this.
"""

from __future__ import annotations

import math

import torch

import helion
import helion.experimental
import helion.language as hl

# These kernels autotune per input shape on first use and cache the result
# (under $TORCHINDUCTOR_CACHE_DIR/helion). Autotuning works correctly on this
# GPU; an earlier "autotuner broken" impression was a red herring — see
# ISSUES.md: the failure only occurs when a kernel is *called at module top
# level without an `if __name__ == "__main__"` guard*, because Helion's
# autotuner benchmarks configs in spawned subprocesses that re-import the entry
# module. All callers here (custom_op impl, tests, benchmarks) invoke the kernel
# inside functions, so autotuning runs cleanly.
#
# `autotune_baseline_fn` gives the autotuner a correct reference to check
# candidate configs against, computed WITHOUT relying on the kernel's own
# default config. This is essential for the causal kernel: its default config
# fails to compile (a Triton "reshape()" codegen error in the masking path), so
# without a custom baseline the autotuner aborts with "Default config failed
# while computing baseline" before it can search. With the SDPA baseline the
# search proceeds and finds working configs. Tolerances match the upstream
# Helion attention example (fp16/bf16 accumulation drift).


def _sdpa_baseline(q, k, v, *, causal):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)


@helion.experimental.aot_kernel(
    static_shapes=True,
    autotune_baseline_fn=lambda q, k, v: _sdpa_baseline(q, k, v, causal=False),
    autotune_baseline_atol=5e-2,
    autotune_baseline_rtol=2e-2,
)
def attention_output(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> torch.Tensor:
    """Computes scaled dot-product attention, returning only the output."""
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504  # 1/log(2)
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        q = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            q_scaled = q * qk_scale
            k = k_view[tile_b, tile_n, :]
            qk = torch.bmm(q_scaled, k.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            v = v_view[tile_b, tile_n, :]
            p = p.to(v.dtype)
            acc = torch.baddbmm(acc, p, v)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())


@helion.experimental.aot_kernel(
    static_shapes=True,
    autotune_baseline_fn=lambda q, k, v: _sdpa_baseline(q, k, v, causal=True),
    autotune_baseline_atol=5e-2,
    autotune_baseline_rtol=2e-2,
)
def causal_attention_output(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> torch.Tensor:
    """Computes causal scaled dot-product attention, returning only the output."""
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504  # 1/log(2)
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        q = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            q_scaled = q * qk_scale
            k = k_view[tile_b, tile_n, :]
            qk = torch.bmm(q_scaled, k.transpose(1, 2), torch.float32)
            qk = torch.where(
                tile_m.index[None, :, None] >= tile_n.index[None, None, :],
                qk,
                float("-inf"),
            )
            m_ij_keepdim = torch.maximum(
                m_i[:, :, None], torch.amax(qk, -1, keepdim=True)
            )
            qk = qk - m_ij_keepdim
            m_ij = m_ij_keepdim.squeeze(-1)
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            v = v_view[tile_b, tile_n, :]
            p = p.to(v.dtype)
            acc = torch.baddbmm(acc, p, v)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        out[tile_b, tile_m, :] = acc.to(out.dtype)
    return out.view(q_in.size())
