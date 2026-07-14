"""Shared benchmark harness for the Helion kernels in this dir.

Times a callable with CUDA events (steady-state), warmup + iters, and reports
mean ms, TFLOP/s or GB/s if provided, and speedup vs a baseline callable.
All kernels are driven through a `if __name__ == "__main__"` guard by the
per-kernel bench scripts (REQUIRED for Helion autotuning; see attention ISSUES).
"""

from __future__ import annotations

import statistics

import torch


def _sync():
    torch.cuda.synchronize()


def bench_gpu(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    _sync()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    _sync()
    return s.elapsed_time(e) / iters


def bench_wall(fn, warmup=20, iters=100):
    import time

    for _ in range(warmup):
        fn()
    _sync()
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        _sync()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.fmean(ts)


def report(name, shape, helion_fn, baseline_fn, *, flops=None, rw_bytes=None,
           atol=1e-2, rtol=1e-2, verify=True, base_iters=None, base_warmup=None):
    """Run + verify + time helion_fn vs baseline_fn. Returns a result dict.

    base_iters/base_warmup: use fewer iters for a pathologically-slow reference
    (e.g. a Python-loop correctness oracle) so timing stays bounded.
    """
    ok = None
    if verify:
        ho = helion_fn()
        bo = baseline_fn()
        _sync()
        h = ho[0] if isinstance(ho, tuple) else ho
        b = bo[0] if isinstance(bo, tuple) else bo
        try:
            ok = torch.allclose(h.float(), b.float(), atol=atol, rtol=rtol)
        except Exception:
            ok = False
    hg = bench_gpu(helion_fn)
    bg = bench_gpu(baseline_fn, warmup=base_warmup or 20, iters=base_iters or 100)
    speedup = bg / hg if hg else 0.0
    extra = ""
    if flops:
        extra = f"{flops / (hg * 1e-3) / 1e12:8.1f} TFLOP/s"
    elif rw_bytes:
        extra = f"{rw_bytes / (hg * 1e-3) / 1e9:8.1f} GB/s"
    print(
        f"  {name:>28} {str(shape):>22}  helion={hg:8.4f}ms  base={bg:8.4f}ms  "
        f"{speedup:5.2f}x  {extra}  ok={ok}"
    )
    return {"name": name, "shape": shape, "helion_ms": hg, "base_ms": bg,
            "speedup": speedup, "ok": ok}
