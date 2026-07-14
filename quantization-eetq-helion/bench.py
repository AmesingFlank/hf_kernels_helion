from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, "/home/dev/hf_kernels_helion")
import torch
import kernels
from bench_common import report

REPO = Path("/home/dev/hf_kernels_helion/quantization-eetq-helion/eetq-helion/result")

def ref(x,W,scale):
    return (x.float() @ (W.float()*scale.float()[:,None]).T).to(torch.float16)

def main():
    k=kernels.get_local_kernel(REPO,"cuda")
    print("=== eetq-helion int8 w8_a16_gemm vs torch dequant matmul ===")
    cfgs={"small":(16,4096,4096),"medium":(64,4096,4096),"large":(128,8192,8192)}
    for name,(M,N,K) in cfgs.items():
        x=torch.randn(M,K,device="cuda",dtype=torch.float16)
        W=torch.randint(-127,127,(N,K),device="cuda",dtype=torch.int8)
        scale=(torch.rand(N,device="cuda",dtype=torch.float16)*0.02+0.01)
        flops=2*M*N*K
        report(f"w8a16_gemm[{name}]",(M,N,K),
               lambda: k.w8_a16_gemm(x,W,scale),
               lambda: ref(x,W,scale), flops=flops, atol=5e-1, rtol=1e-1)

if __name__=="__main__":
    main()
