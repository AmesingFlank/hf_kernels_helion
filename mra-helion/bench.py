from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, "/home/dev/hf_kernels_helion")
import torch
import kernels
from bench_common import report

REPO = Path("/home/dev/hf_kernels_helion/mra-helion/mra-helion/result")

def ref(A, Bm, indices, Bnb, blk):
    Bsz=A.shape[0]; nnz=indices.shape[0]
    out=torch.empty(Bsz,nnz,blk,blk,device=A.device,dtype=torch.float32)
    for b in range(Bsz):
        for n in range(nnz):
            idx=int(indices[n]); bi=idx//Bnb; bj=idx%Bnb
            out[b,n]=A[b,bi*blk:(bi+1)*blk].float() @ Bm[b,bj*blk:(bj+1)*blk].float().T
    return out

def main():
    k=kernels.get_local_kernel(REPO,"cuda")
    print("=== mra-helion mm_to_sparse (block-sparse A@B^T) vs torch ===")
    blk=32
    cfgs={"small":(4,16,16,64,32),"medium":(8,32,32,128,128),"large":(16,64,64,128,256)}
    for name,(Bsz,Anb,Bnb,D,nnz) in cfgs.items():
        A=torch.randn(Bsz,Anb*blk,D,device="cuda",dtype=torch.float16)
        Bm=torch.randn(Bsz,Bnb*blk,D,device="cuda",dtype=torch.float16)
        indices=torch.randint(0,Anb*Bnb,(nnz,),device="cuda",dtype=torch.int32)
        flops=2*Bsz*nnz*blk*blk*D
        report(f"mm_to_sparse[{name}]",(Bsz,nnz,blk),
               lambda: k.mm_to_sparse(A,Bm,indices,Bnb,blk),
               lambda: ref(A,Bm,indices,Bnb,blk), flops=flops, atol=5e-1, rtol=5e-1, base_iters=5, base_warmup=1)

if __name__=="__main__":
    main()
