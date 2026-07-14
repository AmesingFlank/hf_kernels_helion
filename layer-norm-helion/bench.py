from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, "/home/dev/hf_kernels_helion")
import torch
import kernels
from bench_common import report

REPO = Path("/home/dev/hf_kernels_helion/layer-norm-helion/layer-norm-helion/result")

def ref(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w).to(x.dtype)

def main():
    k = kernels.get_local_kernel(REPO, "cuda")
    print("=== layer-norm-helion (RMSNorm) vs torch ===")
    # dropout_add_ln_fwd RMSNorm path; shapes from kernels layer_norm bench util
    shapes = {"small": (256, 768), "medium": (2048, 2048), "large": (16384, 8192)}
    for name,(M,N) in shapes.items():
        x = torch.randn(M,N,device="cuda",dtype=torch.float16)
        w = torch.ones(N,device="cuda",dtype=torch.float16)
        rw = x.numel()*2*2
        report(f"rms_norm[{name}]", (M,N),
               lambda: k.dropout_add_ln_fwd(input=x, gamma=w, epsilon=1e-5, is_rms_norm=True)[0],
               lambda: ref(x, w, 1e-5), rw_bytes=rw)

if __name__ == "__main__":
    main()
