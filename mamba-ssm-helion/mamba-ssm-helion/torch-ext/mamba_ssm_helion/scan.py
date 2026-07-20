"""Helion Mamba selective scan (S6), matching mamba_ssm selective_scan_fn.

Sequential state-space recurrence over the length dim L:
    h[t] = exp(delta[t]*A) * h[t-1] + delta[t]*B[t]*u[t]      (per state n)
    y[t] = sum_n C[t,n]*h[t,n] + D*u[t]  ; optional  y *= silu(z)

Shapes: u,delta:[B,D,L]  A:[D,N]  B,C:[B,N,L]  D_:[D]  z:[B,D,L].
Each program handles a (batch, dim) tile and scans L in-kernel, carrying the
[.,.,N] hidden state.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import helion
import helion.experimental
import helion.language as hl


@helion.experimental.aot_kernel(static_shapes=False)
def _selective_scan(
    u, delta, A, Bm, Cm, D_, z,
    delta_softplus: hl.constexpr, use_z: hl.constexpr,
):
    Bsz, Dim, L = u.shape
    N = hl.specialize(A.shape[1])
    out = torch.empty_like(u)
    for tb, td in hl.tile([Bsz, Dim]):
        h = hl.zeros([tb, td, N], dtype=torch.float32)
        a = A[td, :].to(torch.float32)
        dcoef = D_[td].to(torch.float32)
        for t in range(L):
            dt = delta[tb, td, t].to(torch.float32)
            if delta_softplus:
                dt = F.softplus(dt)
            ut = u[tb, td, t].to(torch.float32)
            bt = Bm[tb, :, t].to(torch.float32)
            ct = Cm[tb, :, t].to(torch.float32)
            dA = torch.exp(dt[:, :, None] * a[None, :, :])
            dBu = dt[:, :, None] * bt[:, None, :] * ut[:, :, None]
            h = dA * h + dBu
            y = torch.sum(h * ct[:, None, :], dim=2) + dcoef[None, :] * ut
            if use_z:
                zt = z[tb, td, t].to(torch.float32)
                y = y * (zt * torch.sigmoid(zt))
            out[tb, td, t] = y.to(out.dtype)
    return out
