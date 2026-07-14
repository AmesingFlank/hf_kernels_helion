from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, "/home/dev/hf_kernels_helion")
import torch, torch.nn.functional as F
import kernels
from bench_common import report

REPO = Path("/home/dev/hf_kernels_helion/causal-conv1d-helion/causal-conv1d-helion/result")

def ref(x, weight, bias, silu):
    B,D,L = x.shape; width = weight.shape[1]
    out = F.conv1d(F.pad(x,(width-1,0)), weight.unsqueeze(1), bias=bias, groups=D)
    return F.silu(out) if silu else out

def main():
    k = kernels.get_local_kernel(REPO, "cuda")
    print("=== causal-conv1d-helion vs torch F.conv1d(grouped) ===")
    width = 4
    shapes = {"small": (8, 768, 512), "medium": (16, 2048, 2048), "large": (32, 4096, 4096)}
    for name,(B,D,L) in shapes.items():
        x = torch.randn(B,D,L,device="cuda",dtype=torch.float16)
        weight = torch.randn(D,width,device="cuda",dtype=torch.float16)
        bias = torch.randn(D,device="cuda",dtype=torch.float16)
        rw = (x.numel()*2 + x.numel())*2
        report(f"causal_conv1d_silu[{name}]", (B,D,L),
               lambda: k.causal_conv1d_fn(x, weight, bias, activation="silu"),
               lambda: ref(x, weight, bias, True), rw_bytes=rw)

if __name__ == "__main__":
    main()
