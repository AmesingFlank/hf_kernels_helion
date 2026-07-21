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
| small 8x1024x2048 | 0.0257 | 0.0327 | 1.68× | 1.31× | 0.78× | 142 → 0.6 | ✓ |
| medium 8x2048x4096 | 0.0310 | 0.0331 | 4.61× | 4.30× | 0.93× | 114 → 0.6 | ✓ |
| large 8x4096x8192 | 0.1224 | 0.1249 | 3.67× | 3.59× | 0.98× | 103 → 0.6 | ✓ |

## causal-conv1d

| input size | autotuned Helion (ms) | pre-tuned Helion (ms) | autotuned speedup | pre-tuned speedup | Δ speed | autotune→pre-tune (s) | verified |
|---|---|---|---|---|---|---|---|
| small 8x768x512 | 0.0352 | 0.0336 | 0.39× | 0.39× | 1.00× | 101 → 1.4 | ✓ |
| medium 16x2048x2048 | 0.5620 | 0.1805 | 0.14× | 0.44× | 3.17× | 80 → 1.4 | ✓ |
| large 32x4096x4096 | 4.1575 | 1.4168 | 0.14× | 0.41× | 2.91× | 98 → 0.7 | ✓ |

## rotary

| input size | autotuned Helion (ms) | pre-tuned Helion (ms) | autotuned speedup | pre-tuned speedup | Δ speed | autotune→pre-tune (s) | verified |
|---|---|---|---|---|---|---|---|
| small 2x128x8x32 | 0.0530 | 0.0536 | 0.32× | 0.38× | 1.18× | 155 → 0.9 | ✓ |
| medium 8x512x32x64 | 0.0562 | 0.0512 | 1.32× | 1.45× | 1.10× | 98 → 0.9 | ✓ |
| large 16x2048x32x64 | 0.1635 | 0.1613 | 3.33× | 3.38× | 1.01× | 83 → 0.8 | ✓ |

## paged-attention

| input size | autotuned Helion (ms) | pre-tuned Helion (ms) | autotuned speedup | pre-tuned speedup | Δ speed | autotune→pre-tune (s) | verified |
|---|---|---|---|---|---|---|---|
| small 16x8x64 | 0.0617 | 0.0673 | 0.12× | 0.11× | 0.95× | 116 → 1.4 | ✓ |
| medium 32x16x64 | 0.0618 | 0.0671 | 0.12× | 0.11× | 0.95× | 194 → 1.4 | ✓ |

## mamba-ssm

| input size | autotuned Helion (ms) | pre-tuned Helion (ms) | autotuned speedup | pre-tuned speedup | Δ speed | autotune→pre-tune (s) | verified |
|---|---|---|---|---|---|---|---|
| small 2x256x128 | 0.0545 | 0.0533 | 0.48× | 0.47× | 0.98× | 101 → 1.2 | ✓ |
| medium 4x1024x512 | 0.3345 | 0.2936 | 0.15× | 0.17× | 1.13× | 92 → 1.4 | ✓ |
| large 8x2048x1024 | 1.4605 | 1.0887 | 0.23× | 0.31× | 1.36× | 88 → 1.0 | ✓ |

## megablocks

_Group sizes differ slightly between the two harnesses (random totals when autotuned, fixed totals when pre-tuned), so rows are matched by shape class (small/medium/large); the K/N GEMM dims are identical._

| input size | autotuned Helion (ms) | pre-tuned Helion (ms) | autotuned speedup | pre-tuned speedup | Δ speed | autotune→pre-tune (s) | verified |
|---|---|---|---|---|---|---|---|
| small G8x2195x1024x1024 | 0.2435 | 0.2211 | 0.27× | 0.28× | 1.04× | 71 → 1.2 | ✓ |
| medium G16x8289x2048x2048 | 0.8242 | 0.7196 | 0.14× | 0.16× | 1.16× | 86 → 1.2 | ✓ |
| large G32x17221x4096x4096 | 2.7935 | 2.4042 | 0.21× | 0.24× | 1.16× | 152 → 0.8 | ✓ |

## deformable-detr

| input size | autotuned Helion (ms) | pre-tuned Helion (ms) | autotuned speedup | pre-tuned speedup | Δ speed | autotune→pre-tune (s) | verified |
|---|---|---|---|---|---|---|---|
| small 2x900x8x32 | 0.1433 | 0.1451 | 0.12× | 0.11× | 0.90× | 101 → 3.2 | ✓ |
| medium 4x2000x8x32 | 0.5678 | 0.5554 | 0.12× | 0.12× | 1.03× | 110 → 4.5 | ✓ |
| large 8x4000x8x64 | 2.2937 | 2.3301 | 0.24× | 0.25× | 1.03× | 113 → 4.8 | ✓ |

## tinygrad-rms

| input size | autotuned Helion (ms) | pre-tuned Helion (ms) | autotuned speedup | pre-tuned speedup | Δ speed | autotune→pre-tune (s) | verified |
|---|---|---|---|---|---|---|---|
| small 4096x1024 | 0.0298 | 0.0342 | 0.60× | 0.52× | 0.86× | 119 → 0.7 | ✓ |
| medium 16384x1024 | 0.0302 | 0.0336 | 1.83× | 1.65× | 0.90× | 111 → 0.6 | ✓ |
| large 65536x1024 | 0.0809 | 0.0907 | 2.46× | 2.19× | 0.89× | 78 → 0.6 | ✓ |

## rwkv

| input size | autotuned Helion (ms) | pre-tuned Helion (ms) | autotuned speedup | pre-tuned speedup | Δ speed | autotune→pre-tune (s) | verified |
|---|---|---|---|---|---|---|---|
| small 4x256x512 | 0.0650 | 0.0667 | 0.83× | 0.81× | 0.98× | 88 → 0.9 | ✓ |
| medium 8x1024x1024 | 0.2519 | 0.6265 | 2.09× | 0.88× | 0.42× | 75 → 0.8 | ✓ |
| large 16x1024x2048 | 0.4994 | 0.6557 | 1.23× | 0.94× | 0.76× | 77 → 0.8 | ✓ |

## layer-norm

| input size | autotuned Helion (ms) | pre-tuned Helion (ms) | autotuned speedup | pre-tuned speedup | Δ speed | autotune→pre-tune (s) | verified |
|---|---|---|---|---|---|---|---|
| small 256x768 | 0.0272 | 0.0346 | 0.40× | 0.29× | 0.72× | 137 → 0.7 | ✓ |
| medium 2048x2048 | 0.0262 | 0.0323 | 0.42× | 0.39× | 0.94× | 134 → 0.7 | ✓ |
| large 16384x8192 | 0.0804 | 0.0822 | 2.69× | 2.65× | 0.98× | 82 → 0.7 | ✓ |

## finegrained-fp8

| input size | autotuned Helion (ms) | pre-tuned Helion (ms) | autotuned speedup | pre-tuned speedup | Δ speed | autotune→pre-tune (s) | verified |
|---|---|---|---|---|---|---|---|
| small 512x512x2048 | 0.0511 | 0.0604 | 1.40× | 1.18× | 0.84× | 82 → 0.9 | ✓ |
| medium 2048x2048x4096 | 0.1758 | 0.3388 | 0.81× | 0.21× | 0.26× | 99 → 1.3 | ✓ |
| large 4096x4096x8192 | 1.2803 | 2.2998 | 0.75× | 0.19× | 0.25× | 89 → 0.7 | ✓ |

## attention

| input size | autotuned Helion (ms) | pre-tuned Helion (ms) | autotuned speedup | pre-tuned speedup | Δ speed | autotune→pre-tune (s) | verified |
|---|---|---|---|---|---|---|---|
| small 2x512x8x64 | 0.0819 | 0.0846 | 0.43× | 0.38× | 0.89× | 120 → 1.4 | ✓ |
| medium 4x1024x16x64 | 0.1082 | 0.1015 | 0.32× | 0.34× | 1.07× | 81 → 1.4 | ✓ |
| large 8x2048x16x128 | 0.8581 | 0.8460 | 0.31× | 0.33× | 1.05× | 88 → 1.6 | ✓ |

## sage-attention

| input size | autotuned Helion (ms) | pre-tuned Helion (ms) | autotuned speedup | pre-tuned speedup | Δ speed | autotune→pre-tune (s) | verified |
|---|---|---|---|---|---|---|---|
| small 2x8x512x128 | 0.0930 | 0.0942 | 1.60× | 1.52× | 0.95× | 131 → 2.2 | ✓ |
| medium 4x16x1024x128 | 0.3018 | 0.2902 | 0.58× | 0.61× | 1.06× | 141 → 2.3 | ✓ |
| large 8x16x2048x128 | 1.0292 | 1.2473 | 0.84× | 0.69× | 0.82× | 135 → 2.2 | ✓ |

## Summary

- **38 kernel×shape** pairs compared (present in both modes).
- **Total autotune time: 4065s → 50.5s** (80× faster time-to-first-run) — the pre-tuned kernels skip the search entirely.
- **Performance retained: geomean Δ speed = 0.942×** (pre-tuned vs individually-autotuned); i.e. the shipped configs are within ~6% of per-shape-optimal on average.
- Min Δ speed = 0.25×, max = 3.17× across all shapes.

