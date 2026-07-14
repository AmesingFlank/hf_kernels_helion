from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, "/home/dev/hf_kernels_helion")
import torch
import kernels
from bench_common import report

REPO = Path("/home/dev/hf_kernels_helion/tinygrad-rms-helion/tinygrad-rms-helion/result")

def ref(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w).to(x.dtype)

def main():
    k = kernels.get_local_kernel(REPO, "cuda")
    print("=== tinygrad-rms-helion vs torch RMSNorm ===")
    shapes = {"small": (2048, 4096), "medium": (8192, 4096), "large": (16384, 8192)}
    for name,(M,N) in shapes.items():
        x = torch.randn(M,N,device="cuda",dtype=torch.float16)
        w = torch.randn(N,device="cuda",dtype=torch.float16)
        rw = x.numel()*2*2
        report(f"rms_norm[{name}]", (M,N),
               lambda: k.tinygrad_rms_norm(x, w, 1e-6),
               lambda: ref(x, w, 1e-6), rw_bytes=rw)

if __name__ == "__main__":
    main()
