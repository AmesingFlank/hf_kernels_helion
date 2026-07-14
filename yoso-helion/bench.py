from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, "/home/dev/hf_kernels_helion")
import torch
import kernels
from bench_common import report

REPO = Path("/home/dev/hf_kernels_helion/yoso-helion/yoso-helion/result")

def ref(hc, v, nb):
    B,N,D = v.shape
    out=torch.zeros(B,nb,D,device=v.device,dtype=torch.float32)
    for b in range(B):
        out[b].index_add_(0, hc[b].long(), v[b].float())
    return out

def main():
    k=kernels.get_local_kernel(REPO,"cuda")
    print("=== yoso-helion lsh_cumulation vs torch index_add ===")
    cfgs={"small":(4,1024,64,32),"medium":(8,4096,128,64),"large":(16,8192,128,128)}
    for name,(B,N,D,nb) in cfgs.items():
        hc=torch.randint(0,nb,(B,N),device="cuda",dtype=torch.int32)
        v=torch.randn(B,N,D,device="cuda",dtype=torch.float16)
        rw=v.numel()*2*2
        report(f"lsh_cumulation[{name}]",(B,N,D,nb),
               lambda: k.lsh_cumulation(hc,v,nb),
               lambda: ref(hc,v,nb), rw_bytes=rw, atol=1e-1, rtol=1e-1)

if __name__=="__main__":
    main()
