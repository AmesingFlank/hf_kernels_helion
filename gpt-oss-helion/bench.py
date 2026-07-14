from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, "/home/dev/hf_kernels_helion")
import torch, torch.nn.functional as F
import kernels
from bench_common import report

REPO = Path("/home/dev/hf_kernels_helion/gpt-oss-helion/gpt-oss-helion/result")

def ref(A,Wg,Wu,offs):
    E=Wg.shape[0]; out=torch.empty(A.shape[0],Wg.shape[2],device=A.device,dtype=A.dtype)
    for e in range(E):
        s,en=int(offs[e]),int(offs[e+1])
        if en>s:
            g=A[s:en].float()@Wg[e].float(); u=A[s:en].float()@Wu[e].float()
            out[s:en]=(F.silu(g)*u).to(A.dtype)
    return out

def main():
    k=kernels.get_local_kernel(REPO,"cuda")
    print("=== gpt-oss-helion MoE SwiGLU vs torch per-expert ===")
    cfgs={"small":(8,256,1024,1024),"medium":(16,512,2048,2048),"large":(32,512,4096,4096)}
    for name,(E,Mavg,H,I) in cfgs.items():
        sizes=torch.randint(Mavg//2,Mavg*3//2,(E,)).tolist()
        offs=torch.tensor([0]+list(torch.cumsum(torch.tensor(sizes),0)),device="cuda",dtype=torch.int32)
        A=torch.randn(sum(sizes),H,device="cuda",dtype=torch.float16)
        Wg=torch.randn(E,H,I,device="cuda",dtype=torch.float16)*0.1
        Wu=torch.randn(E,H,I,device="cuda",dtype=torch.float16)*0.1
        flops=2*2*sum(sizes)*H*I
        report(f"moe_swiglu[{name}]",(E,sum(sizes),H,I),
               lambda: k.moe_swiglu(A,Wg,Wu,offs),
               lambda: ref(A,Wg,Wu,offs), flops=flops, atol=1e-1, rtol=1e-1)

if __name__=="__main__":
    main()
