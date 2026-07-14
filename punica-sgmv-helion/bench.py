from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, "/home/dev/hf_kernels_helion")
import torch
import kernels
from bench_common import report

REPO = Path("/home/dev/hf_kernels_helion/punica-sgmv-helion/punica-sgmv-helion/result")

def ref(y,x,WA,WB,ss,se):
    y=y.clone()
    for i in range(WA.shape[0]):
        s,e=int(ss[i]),int(se[i])
        if e>s: y[s:e]+=(x[s:e].float()@WA[i].float().T@WB[i].float()).to(y.dtype)
    return y

def main():
    k=kernels.get_local_kernel(REPO,"cuda")
    print("=== punica-sgmv-helion (segmented LoRA) vs torch loop ===")
    cfgs={"small":(8,16,1024,1024,64),"medium":(16,16,2048,2048,128),"large":(32,32,2048,2048,128)}
    for name,(nprob,rank,IN,OUT,seg) in cfgs.items():
        starts=[0]
        for _ in range(nprob-1): starts.append(starts[-1]+seg)
        total=seg*nprob
        ss=torch.tensor(starts,device="cuda",dtype=torch.int32)
        se=torch.tensor([s+seg for s in starts],device="cuda",dtype=torch.int32)
        x=torch.randn(total,IN,device="cuda",dtype=torch.float16)
        WA=torch.randn(nprob,rank,IN,device="cuda",dtype=torch.float16)*0.1
        WB=torch.randn(nprob,rank,OUT,device="cuda",dtype=torch.float16)*0.1
        y=torch.randn(total,OUT,device="cuda",dtype=torch.float16)
        flops=2*total*rank*(IN+OUT)
        report(f"sgmv[{name}]",(nprob,total,rank),
               lambda: (k.add_lora_sgmv(y.clone(),x,WA,WB,ss,se,rank)),
               lambda: ref(y,x,WA,WB,ss,se), flops=flops, atol=1e-1, rtol=1e-1)

if __name__=="__main__":
    main()
