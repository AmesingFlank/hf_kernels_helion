# Helion kernels — AOT pre-tuned vs from-scratch autotuned (Triton backend)

Both columns are the **same Helion kernels on the same B200**, benchmarked
against the same `kernels-community` references. The difference is *how the
config was obtained*:

- **autotuned** (`results/triton/`): from-scratch LLM-guided autotuning on this
  machine — the `autotune` column is the wall-clock search time paid on first
  use if the kernel had no pre-tuned config.
- **pre-tuned** (`results/triton_aot/`): the kernel ships a committed
  `_helion_aot_*_cuda_sm100.py` heuristic (`@aot_kernel`, `HELION_AOT_MODE=
  evaluate`). No search happens; the `autotune` column is just the one-config
  compile a downloader pays.

`Δ speed` = pre-tuned speedup / autotuned speedup (≈1.00 means the pre-tuned
config matches the individually-tuned one; <1 means slightly slower). `autotune`
columns show the headline win: seconds of search collapse to a sub-second
compile. All rows are numerically verified in both modes (✓).

Autotuning-time totals below exclude shapes that only exist in one mode.

## activation

| input size | autotuned Helion (ms) | pre-tuned Helion (ms) | autotuned speedup | pre-tuned speedup | Δ speed | autotune→pre-tune (s) | verified |
|---|---|---|---|---|---|---|---|
| small 8x1024x2048 | 0.0257 | 0.0317 | 1.68× | 1.35× | 0.80× | 142 → 0.5 | ✓ |
| medium 8x2048x4096 | 0.0310 | 0.0335 | 4.61× | 4.25× | 0.92× | 114 → 0.5 | ✓ |
| large 8x4096x8192 | 0.1224 | 0.1228 | 3.67× | 3.65× | 0.99× | 103 → 0.6 | ✓ |

## causal-conv1d

| input size | autotuned Helion (ms) | pre-tuned Helion (ms) | autotuned speedup | pre-tuned speedup | Δ speed | autotune→pre-tune (s) | verified |
|---|---|---|---|---|---|---|---|
| small 8x768x512 | 0.0352 | 0.0348 | 0.39× | 0.43× | 1.11× | 101 → 0.7 | ✓ |
| medium 16x2048x2048 | 0.5620 | 0.5333 | 0.14× | 0.15× | 1.07× | 80 → 0.7 | ✓ |
| large 32x4096x4096 | 4.1575 | 4.2182 | 0.14× | 0.14× | 0.98× | 98 → 0.7 | ✓ |

## rotary

| input size | autotuned Helion (ms) | pre-tuned Helion (ms) | autotuned speedup | pre-tuned speedup | Δ speed | autotune→pre-tune (s) | verified |
|---|---|---|---|---|---|---|---|
| small 2x128x8x32 | 0.0530 | 0.0530 | 0.32× | 0.38× | 1.18× | 155 → 0.8 | ✓ |
| medium 8x512x32x64 | 0.0562 | 0.0566 | 1.32× | 1.31× | 0.99× | 98 → 0.8 | ✓ |
| large 16x2048x32x64 | 0.1635 | 0.1677 | 3.33× | 3.25× | 0.97× | 83 → 0.7 | ✓ |

## paged-attention

| input size | autotuned Helion (ms) | pre-tuned Helion (ms) | autotuned speedup | pre-tuned speedup | Δ speed | autotune→pre-tune (s) | verified |
|---|---|---|---|---|---|---|---|
| small 16x8x64 | 0.0617 | 0.0633 | 0.12× | 0.13× | 1.10× | 116 → 1.2 | ✓ |
| medium 32x16x64 | 0.0618 | 0.0625 | 0.12× | 0.15× | 1.25× | 194 → 1.2 | ✓ |

## mamba-ssm

| input size | autotuned Helion (ms) | pre-tuned Helion (ms) | autotuned speedup | pre-tuned speedup | Δ speed | autotune→pre-tune (s) | verified |
|---|---|---|---|---|---|---|---|
| small 2x256x128 | 0.0545 | 0.0460 | 0.48× | 0.60× | 1.26× | 101 → 1.0 | ✓ |
| medium 4x1024x512 | 0.3345 | 0.1919 | 0.15× | 0.26× | 1.75× | 92 → 1.0 | ✓ |
| large 8x2048x1024 | 1.4605 | 1.0422 | 0.23× | 0.33× | 1.42× | 88 → 1.0 | ✓ |

## megablocks

_Group sizes differ slightly between the two harnesses (random totals when autotuned, fixed totals when pre-tuned), so rows are matched by shape class (small/medium/large); the K/N GEMM dims are identical._

| input size | autotuned Helion (ms) | pre-tuned Helion (ms) | autotuned speedup | pre-tuned speedup | Δ speed | autotune→pre-tune (s) | verified |
|---|---|---|---|---|---|---|---|
| small G8x2195x1024x1024 | 0.2435 | 0.2210 | 0.27× | 0.28× | 1.06× | 71 → 0.7 | ✓ |
| medium G16x8289x2048x2048 | 0.8242 | 0.8318 | 0.14× | 0.16× | 1.13× | 86 → 0.7 | ✓ |
| large G32x17221x4096x4096 | 2.7935 | 2.8680 | 0.21× | 0.20× | 0.96× | 152 → 0.7 | ✓ |

## deformable-detr

| input size | autotuned Helion (ms) | pre-tuned Helion (ms) | autotuned speedup | pre-tuned speedup | Δ speed | autotune→pre-tune (s) | verified |
|---|---|---|---|---|---|---|---|
| small 2x900x8x32 | 0.1433 | 0.1499 | 0.12× | 0.11× | 0.89× | 101 → 2.3 | ✓ |
| medium 4x2000x8x32 | 0.5678 | 0.6083 | 0.12× | 0.11× | 0.93× | 110 → 2.4 | ✓ |
| large 8x4000x8x64 | 2.2937 | 2.7205 | 0.24× | 0.21× | 0.88× | 113 → 2.4 | ✓ |

## tinygrad-rms

| input size | autotuned Helion (ms) | pre-tuned Helion (ms) | autotuned speedup | pre-tuned speedup | Δ speed | autotune→pre-tune (s) | verified |
|---|---|---|---|---|---|---|---|
| small 4096x1024 | 0.0298 | 0.0350 | 0.60× | 0.51× | 0.84× | 119 → 0.5 | ✓ |
| medium 16384x1024 | 0.0302 | 0.0337 | 1.83× | 1.64× | 0.90× | 111 → 0.5 | ✓ |
| large 65536x1024 | 0.0809 | 0.0805 | 2.46× | 2.46× | 1.00× | 78 → 0.5 | ✓ |

## rwkv

| input size | autotuned Helion (ms) | pre-tuned Helion (ms) | autotuned speedup | pre-tuned speedup | Δ speed | autotune→pre-tune (s) | verified |
|---|---|---|---|---|---|---|---|
| small 4x256x512 | 0.0650 | 0.0669 | 0.83× | 0.81× | 0.98× | 88 → 0.8 | ✓ |
| medium 8x1024x1024 | 0.2519 | 0.2489 | 2.09× | 2.17× | 1.04× | 75 → 0.7 | ✓ |
| large 16x1024x2048 | 0.4994 | 0.2491 | 1.23× | 2.47× | 2.01× | 77 → 0.7 | ✓ |

## layer-norm

| input size | autotuned Helion (ms) | pre-tuned Helion (ms) | autotuned speedup | pre-tuned speedup | Δ speed | autotune→pre-tune (s) | verified |
|---|---|---|---|---|---|---|---|
| small 256x768 | 0.0272 | 0.0298 | 0.40× | 0.43× | 1.07× | 137 → 0.5 | ✓ |
| medium 2048x2048 | 0.0262 | 0.0312 | 0.42× | 0.35× | 0.84× | 134 → 0.5 | ✓ |
| large 16384x8192 | 0.0804 | 0.0966 | 2.69× | 2.31× | 0.86× | 82 → 0.5 | ✓ |

## finegrained-fp8

| input size | autotuned Helion (ms) | pre-tuned Helion (ms) | autotuned speedup | pre-tuned speedup | Δ speed | autotune→pre-tune (s) | verified |
|---|---|---|---|---|---|---|---|
| small 512x512x2048 | 0.0511 | 0.0650 | 1.40× | 1.11× | 0.79× | 82 → 0.6 | ✓ |
| medium 2048x2048x4096 | 0.1758 | 0.1726 | 0.81× | 0.83× | 1.02× | 99 → 0.6 | ✓ |
| large 4096x4096x8192 | 1.2803 | 1.4669 | 0.75× | 0.65× | 0.87× | 89 → 0.6 | ✓ |

## attention

| input size | autotuned Helion (ms) | pre-tuned Helion (ms) | autotuned speedup | pre-tuned speedup | Δ speed | autotune→pre-tune (s) | verified |
|---|---|---|---|---|---|---|---|
| small 2x512x8x64 | 0.0819 | 0.0791 | 0.43× | 0.46× | 1.07× | 120 → 1.0 | ✓ |
| medium 4x1024x16x64 | 0.1082 | 0.1076 | 0.32× | 0.33× | 1.02× | 81 → 1.0 | ✓ |
| large 8x2048x16x128 | 0.8581 | 1.0887 | 0.31× | 0.25× | 0.82× | 88 → 1.0 | ✓ |

## sage-attention

| input size | autotuned Helion (ms) | pre-tuned Helion (ms) | autotuned speedup | pre-tuned speedup | Δ speed | autotune→pre-tune (s) | verified |
|---|---|---|---|---|---|---|---|
| small 2x8x512x128 | 0.0930 | 0.0950 | 1.60× | 1.50× | 0.94× | 131 → 3.7 | ✓ |
| medium 4x16x1024x128 | 0.3018 | 0.3029 | 0.58× | 0.58× | 1.01× | 141 → 2.1 | ✓ |
| large 8x16x2048x128 | 1.0292 | 1.1190 | 0.84× | 0.77× | 0.92× | 135 → 2.1 | ✓ |

## Summary

- **38 kernel×shape** pairs compared (present in both modes).
- **Total autotune time: 4065s → 38.5s** (106× faster time-to-first-run) — the pre-tuned kernels skip the search entirely.
- **Performance retained: geomean Δ speed = 1.022×** (pre-tuned vs individually-autotuned); i.e. the shipped configs are within ~2% of per-shape-optimal on average.
- Min Δ speed = 0.79×, max = 2.01× across all shapes.

