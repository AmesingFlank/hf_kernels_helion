"""Benchmark Helion NF4 GEMM vs a PyTorch dequant+matmul reference.

bitsandbytes is not installed, so the baseline is an explicit NF4
dequantization followed by a matmul (what the fused kernel replaces). Guarded
for autotuning.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, "/home/dev/hf_kernels_helion")
import torch
import kernels
from bench_common import report

REPO = Path("/home/dev/hf_kernels_helion/quantization-bitsandbytes-helion/bnb4bit-helion/result")
NF4 = [-1.0,-0.6961928,-0.52507305,-0.39491749,-0.28444138,-0.18477343,-0.09105004,0.0,
       0.0795803,0.1609302,0.2461123,0.33791524,0.44070983,0.562617,0.72295684,1.0]


def ref_dequant_matmul(inp, w_packed, absmax, code, blocksize):
    N, Kp = w_packed.shape
    K = Kp * 2
    b = w_packed.to(torch.int32)
    lo = b & 0xF
    hi = (b >> 4) & 0xF
    w = torch.empty(N, K, device=inp.device, dtype=torch.float32)
    w[:, 0::2] = code[lo]
    w[:, 1::2] = code[hi]
    kidx = torch.arange(K, device=inp.device)
    w = w * absmax[:, kidx // blocksize]
    return (inp.to(torch.bfloat16).float() @ w.t()).to(inp.dtype)


def main():
    k = kernels.get_local_kernel(REPO, "cuda")
    code = torch.tensor(NF4, device="cuda", dtype=torch.float32)
    print("=== bnb4bit-helion NF4 gemm_4bit_forward vs torch dequant+matmul ===")
    bs = 64
    # (M, N, K): weight-only quant GEMM (LLM linear layers)
    shapes = {"small": (16, 4096, 4096), "medium": (64, 4096, 4096), "large": (128, 8192, 8192)}
    for name, (M, N, K) in shapes.items():
        inp = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w_packed = torch.randint(0, 256, (N, K // 2), device="cuda", dtype=torch.uint8)
        absmax = torch.rand(N, K // bs, device="cuda", dtype=torch.float32) + 0.5
        flops = 2 * M * N * K
        report(
            f"gemm_4bit[{name}]", (M, N, K),
            lambda: k.gemm_4bit_forward(inp, w_packed, absmax, bs, 1),
            lambda: ref_dequant_matmul(inp, w_packed, absmax, code, bs),
            flops=flops, atol=2e-1, rtol=2e-1,
        )


if __name__ == "__main__":
    main()
