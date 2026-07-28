"""Attention benchmark: Helion ``attention()`` vs PyTorch SDPA, apples-to-apples.

Both sides take the **same** ``(B, H, S, D)`` layout (the SDPA convention), so
there is **no transpose and no ``.contiguous()`` copy on either side** -- we
compare ``attention(q, k, v)`` directly against
``F.scaled_dot_product_attention(q, k, v)``. (The ``flash_attn_func`` wrapper is
deliberately *not* measured here: it exists only to mimic flash-attn's
``(B, S, H, D)`` calling convention, and its output-layout ``.contiguous()`` is a
flash-attn contract, not an SDPA-comparison cost. Callers who want the fast path
use ``attention()`` with ``(B, H, S, D)``, which is what this script exercises.)

Two tables are printed:

* **TUNED** -- the exact shapes the AOT heuristic was pre-tuned on (see
  ``scripts/aot_tune.py:sh_attention``, mirroring the external analysis'
  https://gist.github.com/sayakpaul/a1f858c354f010bbf5aeabd94602b557 sweep).
  This is a *fidelity* check: how well the shipped heuristic serves the shapes
  it trained on.
* **UNTUNED (held-out)** -- shapes the heuristic never saw: new head_dims
  (96), head counts (20, 32), interpolated/extrapolated sequence lengths, and
  batch sizes off the training grid. This is a *generalization* check -- the
  more meaningful number for callers who hit arbitrary shapes.

Timing mirrors the external analysis' methodology:

* ``device_ms``: CUDA-graph replay, so CPU dispatch is excluded -- necessary
  because these kernels run in tens of microseconds, where Python/dispatch
  overhead otherwise dominates. This is the headline metric.
* ``e2e_ms``: plain event-timed Python loop -- what a caller actually sees.

Loads the **local** ``build/`` kernel (the re-tuned, copy-free build in this
repo) via ``kernels.get_local_kernel`` -- run it after ``aot_tune.py`` to see the
effect of a fresh heuristic. bf16, non-causal.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

import kernels

DEVICE = "cuda"
DTYPE = torch.bfloat16
WARMUP = 25
REPS = 30
INNER = 10
SEED = 42

# Local re-tuned build (result/ is a symlink to build/); repo-relative so a
# fresh clone + aot_tune.py just works. The AOT heuristic beside the kernel
# source auto-applies on load -- no HELION_AOT_MODE needed.
BUILD = Path(__file__).resolve().parent.parent / "build"

# (label, B, H, S, D). The shapes the AOT heuristic is PRE-TUNED on. Keep this
# in sync with scripts/aot_tune.py:sh_attention() -- it is the source of truth
# (this benchmark ships standalone on the Hub and can't import it). These are
# the analysis' 19 distinct shapes: upstream small/medium/large, batch + seq
# sweeps at each tuned head_dim, cross combinations (incl. head_dims 32/256),
# and DiT-ish long single-sample sequences.
TUNED_SHAPES = [
    ("upstream_small", 2, 8, 128, 64),
    ("upstream_medium", 4, 16, 512, 64),
    ("upstream_large", 8, 32, 1024, 128),
    ("b1_d64_s1024", 1, 16, 1024, 64),
    ("b2_d64_s1024", 2, 16, 1024, 64),
    ("b4_d64_s1024", 4, 16, 1024, 64),
    ("b8_d128_s1024", 8, 16, 1024, 128),
    ("b16_d128_s1024", 16, 16, 1024, 128),
    ("b32_d128_s1024", 32, 16, 1024, 128),
    ("b4_d64_s2048", 4, 16, 2048, 64),
    ("b4_d64_s4096", 4, 16, 4096, 64),
    ("b8_d128_s2048", 8, 16, 2048, 128),
    ("b8_d128_s4096", 8, 16, 4096, 128),
    ("b2_d128_s1024", 2, 16, 1024, 128),
    ("b8_d64_s1024", 8, 16, 1024, 64),
    ("b4_d32_s1024", 4, 16, 1024, 32),
    ("b8_d256_s1024", 8, 16, 1024, 256),
    ("dit_b1_d128_s4608", 1, 24, 4608, 128),
    ("dit_b2_d64_s4608", 2, 24, 4608, 64),
]

# (label, B, H, S, D). Shapes the heuristic was NOT trained on -- a
# generalization test. Deliberately off the training grid: a new head_dim (96),
# new head counts (20, 32), sequence lengths interpolated (1536, 3072) and
# extrapolated (8192) beyond the trained {512..4608}, and untrained batch sizes.
# All are verified to compile and match SDPA on the shipped build.
UNTUNED_SHAPES = [
    ("gpt2_b2_h12_s768_d64", 2, 12, 768, 64),
    ("b16_h16_s512_d64", 16, 16, 512, 64),
    ("b6_h16_s1024_d128", 6, 16, 1024, 128),
    ("s1536_interp_d128", 8, 16, 1536, 128),
    ("s3072_interp_d128", 4, 16, 3072, 128),
    ("h32_b1_s2048_d128", 1, 32, 2048, 128),
    ("s8192_extrap_d128", 2, 16, 8192, 128),
    ("h20_b8_s1024_d128", 8, 20, 1024, 128),
    ("d96_new_headdim", 4, 16, 1024, 96),
    ("b12_s256_short_d64", 12, 16, 256, 64),
]


def _sync():
    torch.cuda.synchronize()


def time_device(fn):
    """Per-call GPU time in ms via CUDA-graph replay (no CPU dispatch in loop)."""
    for _ in range(WARMUP):
        fn()
    _sync()

    # Capture must happen on a warmed non-default stream.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    _sync()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(INNER):
            fn()
    _sync()
    for _ in range(5):
        graph.replay()
    _sync()

    lat = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(REPS):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        lat.append(start.elapsed_time(end) / INNER)
    del graph
    return statistics.median(lat)


def time_e2e(fn):
    """Per-call end-to-end latency in ms (includes CPU dispatch)."""
    for _ in range(WARMUP):
        fn()
    _sync()
    lat = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(REPS):
        start.record()
        for _ in range(INNER):
            fn()
        end.record()
        end.synchronize()
        lat.append(start.elapsed_time(end) / INNER)
    return statistics.median(lat)


def attn_flops(B, H, S, D):
    # 2 matmuls (QK^T and PV), each 2*B*H*S*S*D flops.
    return 2 * 2 * B * H * S * S * D


def _hdr():
    h = (
        f"{'shape':>20} {'B,H,S,D':>16} "
        f"{'hln_dev':>9} {'sdpa_dev':>9} {'dev x':>7} "
        f"{'hln_e2e':>9} {'sdpa_e2e':>9} {'e2e x':>7} "
        f"{'hln TF/s':>9} {'sdpa TF/s':>9} {'ok':>5}"
    )
    return h


def run_table(title, shapes, helion_k):
    """Time every shape in one set, print a table, return (dev, e2e) speedups."""
    hdr = _hdr()
    print(f"### {title}  ({len(shapes)} shapes)")
    print(hdr)
    print("-" * len(hdr))

    dev_speedups, e2e_speedups, wins, all_ok = [], [], 0, True
    for label, B, H, S, D in shapes:
        q = torch.randn((B, H, S, D), device=DEVICE, dtype=DTYPE)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        flops = attn_flops(B, H, S, D)

        helion_fn = lambda: helion_k.attention(q, k, v, causal=False)
        sdpa_fn = lambda: F.scaled_dot_product_attention(q, k, v)

        # Correctness (vs SDPA, upstream tolerances) before timing.
        out = helion_fn()
        ref = sdpa_fn()
        _sync()
        ok = torch.allclose(out.float(), ref.float(), atol=5e-2, rtol=2e-2)
        all_ok = all_ok and ok

        h_dev = time_device(helion_fn)
        s_dev = time_device(sdpa_fn)
        h_e2e = time_e2e(helion_fn)
        s_e2e = time_e2e(sdpa_fn)

        dev_x = s_dev / h_dev
        e2e_x = s_e2e / h_e2e
        dev_speedups.append(dev_x)
        e2e_speedups.append(e2e_x)
        if dev_x > 1.0:
            wins += 1

        print(
            f"{label:>20} {f'{B},{H},{S},{D}':>16} "
            f"{h_dev*1e3:>8.1f}u {s_dev*1e3:>8.1f}u {dev_x:>6.2f}x "
            f"{h_e2e*1e3:>8.1f}u {s_e2e*1e3:>8.1f}u {e2e_x:>6.2f}x "
            f"{flops/(h_dev*1e-3)/1e12:>9.1f} {flops/(s_dev*1e-3)/1e12:>9.1f} "
            f"{str(ok):>5}"
        )

        # Release before the next shape so the largest (s=8192) don't stack.
        del q, k, v, out, ref
        torch.cuda.empty_cache()

    print("-" * len(hdr))
    print(f"{title}: device geomean {statistics.geometric_mean(dev_speedups):.2f}x, "
          f"{wins}/{len(shapes)} faster | "
          f"e2e geomean {statistics.geometric_mean(e2e_speedups):.2f}x | "
          f"correctness: {'ALL PASS' if all_ok else 'SOME FAIL'}\n")
    return dev_speedups, e2e_speedups


def main():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Sanity: the held-out set must genuinely be disjoint from the tuned set,
    # otherwise the "untuned" table isn't measuring generalization.
    tuned = {s[1:] for s in TUNED_SHAPES}
    overlap = [s for s in UNTUNED_SHAPES if s[1:] in tuned]
    assert not overlap, f"held-out shapes overlap the tuned set: {overlap}"

    helion_k = kernels.get_local_kernel(BUILD, "cuda")

    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} "
          f"| cap {torch.cuda.get_device_capability(0)}")
    print(f"warmup={WARMUP} reps={REPS} inner={INNER} | bf16 | non-causal | "
          f"(B,H,S,D), no transpose either side")
    print("x>1 = Helion faster; times in microseconds (u)\n")

    run_table("TUNED (pre-tuned shapes -- fidelity)", TUNED_SHAPES, helion_k)
    run_table("UNTUNED (held-out shapes -- generalization)", UNTUNED_SHAPES, helion_k)


if __name__ == "__main__":
    main()
