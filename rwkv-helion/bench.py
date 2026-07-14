from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, "/home/dev/hf_kernels_helion")
import torch
import kernels
from bench_common import report

REPO = Path("/home/dev/hf_kernels_helion/rwkv-helion/rwkv-helion/result")

def ref(w,u,k,v):
    B,T,C = k.shape
    wf,uf,kf,vf = w.float(),u.float(),k.float(),v.float()
    ew = torch.exp(-torch.exp(wf))
    a = torch.zeros(B,C,device=k.device); b = torch.zeros(B,C,device=k.device)
    outs=[]
    for t in range(T):
        eku = torch.exp(uf[None]+kf[:,t])
        outs.append((a+eku*vf[:,t])/(b+eku))
        ek = torch.exp(kf[:,t]); a = ew[None]*a+ek*vf[:,t]; b = ew[None]*b+ek
    return torch.stack(outs,1)

def main():
    k_ = kernels.get_local_kernel(REPO, "cuda")
    print("=== rwkv-helion wkv vs torch sequential ===")
    shapes = {"small": (4,256,512), "medium": (8,1024,1024), "large": (16,1024,2048)}
    for name,(B,T,C) in shapes.items():
        w = torch.randn(C,device="cuda"); u = torch.randn(C,device="cuda")
        k = torch.randn(B,T,C,device="cuda",dtype=torch.float16)*0.5
        v = torch.randn(B,T,C,device="cuda",dtype=torch.float16)
        report(f"wkv[{name}]", (B,T,C),
               lambda: k_.wkv(w,u,k,v),
               lambda: ref(w,u,k,v), atol=5e-2, rtol=5e-2, base_iters=3, base_warmup=1)

if __name__ == "__main__":
    main()
