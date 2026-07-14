from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, "/home/dev/hf_kernels_helion")
import torch
import torch.nn.functional as F
import kernels
from bench_common import report

REPO = Path("/home/dev/hf_kernels_helion/mamba-ssm-helion/mamba-ssm-helion/result")


def ref(u, delta, A, Bm, Cm, D_, z, softplus, use_z):
    Bsz, Dim, L = u.shape
    dt = F.softplus(delta.float()) if softplus else delta.float()
    h = torch.zeros(Bsz, Dim, A.shape[1], device=u.device, dtype=torch.float32)
    ys = []
    for t in range(L):
        dA = torch.exp(dt[:, :, t, None] * A[None])
        dBu = dt[:, :, t, None] * Bm.float()[:, None, :, t] * u.float()[:, :, t, None]
        h = dA * h + dBu
        y = (h * Cm.float()[:, None, :, t]).sum(-1) + D_[None, :] * u.float()[:, :, t]
        ys.append(y)
    out = torch.stack(ys, -1)
    if use_z:
        zf = z.float()
        out = out * (zf * torch.sigmoid(zf))
    return out


def main():
    k = kernels.get_local_kernel(REPO, "cuda")
    print("=== mamba-ssm-helion selective_scan_fn vs torch sequential ref ===")
    shapes = {"small": (2, 256, 128, 16), "medium": (4, 1024, 512, 16), "large": (8, 2048, 1024, 16)}
    for name, (Bsz, Dim, L, N) in shapes.items():
        u = torch.randn(Bsz, Dim, L, device="cuda", dtype=torch.float16)
        delta = torch.rand(Bsz, Dim, L, device="cuda", dtype=torch.float16)
        A = -torch.rand(Dim, N, device="cuda", dtype=torch.float32)
        Bm = torch.randn(Bsz, N, L, device="cuda", dtype=torch.float16)
        Cm = torch.randn(Bsz, N, L, device="cuda", dtype=torch.float16)
        D_ = torch.randn(Dim, device="cuda", dtype=torch.float32)
        z = torch.randn(Bsz, Dim, L, device="cuda", dtype=torch.float16)
        report(
            f"selective_scan[{name}]", (Bsz, Dim, L, N),
            lambda: k.selective_scan_fn(u, delta, A, Bm, Cm, D=D_, z=z, delta_softplus=True),
            lambda: ref(u, delta, A, Bm, Cm, D_, z, True, True),
            atol=1e-1, rtol=1e-1, base_iters=3, base_warmup=1,
        )


if __name__ == "__main__":
    main()
