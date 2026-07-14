"""Attention benchmark: Helion attention vs PyTorch SDPA (and flash-attn3 attempt).

Workload sizes mirror kernels.benchmarks.attention.FlashAttentionBenchmark
(small/medium/large), layout (B, S, H, D), fp16, non-causal.

flash-attn3 is attempted but the Hub build has no sm_100 (Blackwell) SASS, so it
errors on the B200 — recorded rather than timed.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import kernels

DEVICE = "cuda"
WARMUP = 20
ITERS = 100
SEED = 42

# (B, S, H, D) — matches kernels' FlashAttentionBenchmark
WORKLOADS = {
    "small": (2, 128, 8, 64),
    "medium": (4, 512, 16, 64),
    "large": (8, 1024, 32, 128),
}


def _sync():
    torch.cuda.synchronize()


def time_gpu(fn):
    for _ in range(WARMUP):
        fn()
    _sync()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    _sync()
    return start.elapsed_time(end) / ITERS


def time_wall(fn):
    for _ in range(WARMUP):
        fn()
    _sync()
    ts = []
    for _ in range(ITERS):
        s = time.perf_counter()
        fn()
        _sync()
        ts.append((time.perf_counter() - s) * 1000)
    return statistics.fmean(ts)


def sdpa_bshd(q, k, v, causal=False):
    qt, kt, vt = (x.transpose(1, 2) for x in (q, k, v))
    return (
        F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal)
        .transpose(1, 2)
        .contiguous()
    )


def attn_flops(B, S, H, D):
    # 2 matmuls (QK^T and PV), each 2*B*H*S*S*D flops
    return 2 * 2 * B * H * S * S * D


def main():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    helion_k = kernels.get_local_kernel(
        Path("/home/dev/hf_kernels_helion/attention-helion/attention-helion/result"),
        "cuda",
    )

    # flash-attn3 is NOT run here: its Hub build has no sm_100 SASS, and the
    # resulting CUDA launch failure poisons the whole CUDA context (breaking
    # every subsequent kernel in the process). It is probed separately in a
    # throwaway process (see check_fa3.py). Kept out of this run on purpose.
    fa3 = None
    fa3_status = "not runnable on B200 (no sm_100 SASS) — see benchmark_results.md"

    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
    print(f"warmup={WARMUP} iters={ITERS} | fp16 | non-causal (B,S,H,D)")
    print(f"flash-attn3: {fa3_status}\n")

    hdr = (
        f"{'workload':>9} {'B,S,H,D':>15} {'kernel':>14} "
        f"{'gpu_ms':>9} {'wall_ms':>9} {'TFLOP/s':>9} {'vs SDPA':>8} {'ok':>5}"
    )
    print(hdr)
    print("-" * len(hdr))

    for name, (B, S, H, D) in WORKLOADS.items():
        q = torch.randn(B, S, H, D, device=DEVICE, dtype=torch.float16)
        k = torch.randn(B, S, H, D, device=DEVICE, dtype=torch.float16)
        v = torch.randn(B, S, H, D, device=DEVICE, dtype=torch.float16)
        ref = sdpa_bshd(q, k, v, causal=False)
        flops = attn_flops(B, S, H, D)

        cands = {
            "torch-sdpa": lambda: sdpa_bshd(q, k, v, causal=False),
            "helion": lambda: helion_k.flash_attn_func(q, k, v, causal=False),
        }
        if fa3 is not None:
            cands["flash-attn3"] = lambda: fa3.flash_attn_func(q, k, v, causal=False)

        base = None
        for kname, fn in cands.items():
            out = fn()
            _sync()
            o = out[0] if isinstance(out, tuple) else out
            ok = torch.allclose(o.float(), ref.float(), atol=5e-2, rtol=2e-2)
            g = time_gpu(fn)
            w = time_wall(fn)
            if kname == "torch-sdpa":
                base = g
            spd = base / g if base else 1.0
            tflops = flops / (g * 1e-3) / 1e12
            print(
                f"{name:>9} {f'{B},{S},{H},{D}':>15} {kname:>14} "
                f"{g:>9.4f} {w:>9.4f} {tflops:>9.1f} {spd:>7.2f}x {str(ok):>5}"
            )
        print()


if __name__ == "__main__":
    main()
