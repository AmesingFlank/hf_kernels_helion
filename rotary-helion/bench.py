from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, "/home/dev/hf_kernels_helion")
import torch
import kernels
from bench_common import report

REPO = Path("/home/dev/hf_kernels_helion/rotary-helion/rotary-helion/result")

def ref(x1,x2,cos,sin):
    return x1*cos - x2*sin, x1*sin + x2*cos

def main():
    k = kernels.get_local_kernel(REPO, "cuda")
    print("=== rotary-helion apply_rotary vs torch ===")
    shapes = {"small": (2,128,8,32), "medium": (8,512,32,64), "large": (16,2048,32,64)}
    for name,(B,S,H,R) in shapes.items():
        x1 = torch.randn(B,S,H,R,device="cuda",dtype=torch.float32)
        x2 = torch.randn(B,S,H,R,device="cuda",dtype=torch.float32)
        cos = torch.randn(S,1,R,device="cuda"); sin = torch.randn(S,1,R,device="cuda")
        rw = x1.numel()*4*4
        report(f"apply_rotary[{name}]", (B,S,H,R),
               lambda: k.apply_rotary(x1,x2,cos,sin),
               lambda: ref(x1,x2,cos,sin), rw_bytes=rw)

if __name__ == "__main__":
    main()
