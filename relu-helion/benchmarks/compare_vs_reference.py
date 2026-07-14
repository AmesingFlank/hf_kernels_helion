"""Thorough benchmark: Helion relu vs precompiled C++ reference vs torch.relu.

Two timing methods:
  * wall  = time.perf_counter per-iter around a sync  (includes CPU dispatch;
            this is what kernels' official runner reports)
  * gpu   = cuda.Event elapsed_time over the whole iter loop / iters
            (GPU-side time, the standard for kernel micro-benchmarking)

Sizes span from launch-overhead-bound (1M) to bandwidth-bound (64M) so the
picture is not distorted by a single regime. ReLU is memory-bound, so at
large sizes all three should approach HBM bandwidth.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import kernels

DEVICE = torch.device("cuda")
WARMUP = 30
ITERS = 300
SEED = 42

# (rows, cols) -> element count. fp32 => bytes = elems * 4 * 2 (read+write)
SHAPES = [
    (1024, 1024),    # 1M   elems  (4 MB rw*2)   launch-overhead regime
    (4096, 4096),    # 16M  elems  (128 MB)
    (8192, 8192),    # 64M  elems  (512 MB)      bandwidth regime
]


def _sync() -> None:
    torch.cuda.synchronize()


def time_wall(fn):
    for _ in range(WARMUP):
        fn()
    _sync()
    times_ms = []
    for _ in range(ITERS):
        start = time.perf_counter()
        fn()
        _sync()
        times_ms.append((time.perf_counter() - start) * 1000.0)
    return {
        "mean_ms": statistics.fmean(times_ms),
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "std_ms": statistics.pstdev(times_ms),
    }


def time_gpu(fn):
    # One big timed region with events bracketing the whole loop, so per-iter
    # CPU launch cost overlaps with GPU execution (steady-state throughput).
    for _ in range(WARMUP):
        fn()
    _sync()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(ITERS):
        fn()
    end_ev.record()
    _sync()
    total_ms = start_ev.elapsed_time(end_ev)
    return total_ms / ITERS


def main() -> None:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    ref = kernels.get_kernel("kernels-community/relu", version=1)
    mine = kernels.get_local_kernel(
        Path("/home/dev/relu-helion/relu-helion/result"), "cuda"
    )

    print(f"torch {torch.__version__} | device {torch.cuda.get_device_name(0)}")
    print(f"warmup={WARMUP} iters={ITERS}  (fp32, ReLU is memory-bound)\n")

    header = (
        f"{'shape':>11} {'elems':>7} {'kernel':>15} "
        f"{'wall_mean':>10} {'gpu_mean':>9} {'gpu_min?':>8} "
        f"{'GB/s(gpu)':>10} {'vs torch':>9} {'ok':>4}"
    )
    print(header)
    print("-" * len(header))

    for (h, w) in SHAPES:
        elems = h * w
        rw_bytes = elems * 4 * 2  # read x + write out, fp32
        x = torch.randn(h, w, device=DEVICE, dtype=torch.float32)
        out = torch.empty_like(x)
        ref_val = F.relu(x)

        candidates = {
            "torch.relu": lambda: F.relu(x),
            "reference-cuda": lambda: ref.relu(x, out),
            "helion": lambda: mine.relu(x, out),
        }

        base_gpu = None
        for name, fn in candidates.items():
            res = fn()
            _sync()
            check = res if res is not None else out
            ok = torch.allclose(check, ref_val, atol=1e-3)

            wall = time_wall(fn)
            gpu_ms = time_gpu(fn)
            if name == "torch.relu":
                base_gpu = gpu_ms
            speedup = base_gpu / gpu_ms if base_gpu else 1.0
            gbps = rw_bytes / (gpu_ms * 1e-3) / 1e9

            elems_str = f"{elems // 1_000_000}M" if elems >= 1_000_000 else str(elems)
            print(
                f"{f'{h}x{w}':>11} {elems_str:>7} {name:>15} "
                f"{wall['mean_ms']:>10.4f} {gpu_ms:>9.4f} {'':>8} "
                f"{gbps:>10.1f} {speedup:>8.2f}x {str(ok):>4}"
            )
        print()

    # Report B200 peak HBM for context (HBM3e ~ 8 TB/s on B200)
    print("Note: B200 peak HBM ~8 TB/s; ReLU is bandwidth-bound so higher GB/s = better.")


if __name__ == "__main__":
    main()
