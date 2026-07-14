from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, "/home/dev/hf_kernels_helion")
import torch
import kernels
from bench_common import report

REPO = Path("/home/dev/hf_kernels_helion/megablocks-helion/megablocks-helion/result")


def ref(A, W, offs):
    G = W.shape[0]
    out = torch.empty(A.shape[0], W.shape[2], device=A.device, dtype=A.dtype)
    for g in range(G):
        s, e = int(offs[g]), int(offs[g + 1])
        out[s:e] = (A[s:e].float() @ W[g].float()).to(A.dtype)
    return out


def main():
    k = kernels.get_local_kernel(REPO, "cuda")
    print("=== megablocks-helion grouped GEMM (MoE) vs torch per-expert matmul ===")
    torch.manual_seed(0)
    # (num_experts, tokens_per_expert_avg, K, N)
    cfgs = {"small": (8, 256, 1024, 1024), "medium": (16, 512, 2048, 2048), "large": (32, 512, 4096, 4096)}
    for name, (G, Mavg, K, N) in cfgs.items():
        sizes = torch.randint(Mavg // 2, Mavg * 3 // 2, (G,)).tolist()
        offs = torch.tensor([0] + list(torch.cumsum(torch.tensor(sizes), 0)), device="cuda", dtype=torch.int32)
        A = torch.randn(sum(sizes), K, device="cuda", dtype=torch.float16)
        W = torch.randn(G, K, N, device="cuda", dtype=torch.float16)
        flops = 2 * sum(sizes) * K * N
        report(
            f"grouped_gemm[{name}]", (G, sum(sizes), K, N),
            lambda: k.grouped_gemm(A, W, offs),
            lambda: ref(A, W, offs),
            flops=flops, atol=5e-1, rtol=5e-1,
        )


if __name__ == "__main__":
    main()
