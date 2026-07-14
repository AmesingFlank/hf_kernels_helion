"""Benchmark the Helion activation kernel vs PyTorch. Guarded for autotuning."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, "/home/dev/hf_kernels_helion")
import torch
import torch.nn.functional as F
import kernels
from bench_common import report

REPO = Path("/home/dev/hf_kernels_helion/activation-helion/activation-helion/result")


def main():
    k = kernels.get_local_kernel(REPO, "cuda")
    print("=== activation-helion (silu_and_mul, gelu_and_mul) vs torch ===")
    shapes = {"small": (8, 1024, 2048), "medium": (8, 2048, 4096), "large": (8, 4096, 8192)}
    results = []
    for name, (B, S, D2) in shapes.items():
        d = D2 // 2
        x = torch.randn(B, S, D2, device="cuda", dtype=torch.float16)
        out = torch.empty(B, S, d, device="cuda", dtype=torch.float16)
        rw = (x.numel() + out.numel()) * 2  # bytes r+w fp16
        # silu
        results.append(report(
            f"silu_and_mul[{name}]", (B, S, D2),
            lambda: (k.silu_and_mul(out, x), out)[1],
            lambda: F.silu(x[..., :d]) * x[..., d:],
            rw_bytes=rw,
        ))
        # gelu
        results.append(report(
            f"gelu_and_mul[{name}]", (B, S, D2),
            lambda: (k.gelu_and_mul(out, x), out)[1],
            lambda: F.gelu(x[..., :d]) * x[..., d:],
            rw_bytes=rw,
        ))
    return results


if __name__ == "__main__":
    main()
