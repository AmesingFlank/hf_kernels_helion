from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, "/home/dev/hf_kernels_helion")
import torch
import kernels
from bench_common import report

REPO = Path("/home/dev/hf_kernels_helion/deep-gemm-helion/deep-gemm-helion/result")

def ref(A, B, sA, sB, bk):
    M,K=A.shape; N,_=B.shape
    saf=sA.repeat_interleave(bk,dim=1)[:,:K]; sbf=sB.repeat_interleave(bk,dim=1)[:,:K]
    return ((A.float()*saf) @ (B.float()*sbf).T).to(torch.bfloat16)

def main():
    k=kernels.get_local_kernel(REPO,"cuda")
    print("=== deep-gemm-helion blockwise fp8 GEMM vs torch bf16 (dequant) matmul ===")
    bk=128
    cfgs={"small":(512,512,2048),"medium":(2048,2048,4096),"large":(4096,4096,8192)}
    for name,(M,N,K) in cfgs.items():
        A=(torch.randn(M,K,device="cuda")*0.3).to(torch.float8_e4m3fn)
        B=(torch.randn(N,K,device="cuda")*0.3).to(torch.float8_e4m3fn)
        sA=torch.rand(M,K//bk,device="cuda")*0.5+0.5
        sB=torch.rand(N,K//bk,device="cuda")*0.5+0.5
        flops=2*M*N*K
        report(f"fp8_gemm[{name}]",(M,N,K),
               lambda: k.gemm_fp8_fp8_bf16_nt(A,B,sA,sB,bk),
               lambda: ref(A,B,sA,sB,bk), flops=flops, atol=5e-1, rtol=1e-1)

if __name__=="__main__":
    main()
