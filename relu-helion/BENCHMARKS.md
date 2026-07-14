# Benchmark: Helion ReLU vs. `kernels-community/relu`

Head-to-head of this Helion kernel against the reference precompiled **C++ CUDA**
`kernels-community/relu` kernel, with PyTorch's native `torch.relu` as the
ground-truth baseline (both kernels are verified against it).

## Setup

| | |
|---|---|
| GPU | NVIDIA **B200** (sm_100, HBM3e) |
| torch | **2.13.0+cu130** (dedicated `~/venv_torch_213` uv venv — see below) |
| dtype | fp32 |
| Helion | autotuned once per shape (`full` effort), configs cached and reused |
| reference | `kernels-community/relu` v1, precompiled `torch213-cxx11-cu130` variant |

**Why a separate torch 2.13 venv?** The host runs torch 2.14.0a0, for which the
precompiled reference has no ABI-compatible build (it ships variants only up to
2.13). The Helion kernel is `torch-noarch` (JIT) and runs on either, but to
benchmark *both* kernels head-to-head they must share a torch the reference can
load. Created with:

```bash
uv venv ~/venv_torch_213 --python 3.12
VIRTUAL_ENV=~/venv_torch_213 uv pip install \
  "torch==2.13.*" --index-url https://download.pytorch.org/whl/cu130
VIRTUAL_ENV=~/venv_torch_213 uv pip install numpy hatchling hatch-vcs editables \
  "git+https://github.com/huggingface/kernels.git@02d7cb3#subdirectory=kernels"
VIRTUAL_ENV=~/venv_torch_213 uv pip install --no-build-isolation -e ~/helion
```

## Method

Two timing methods per kernel (see `benchmarks/compare_vs_reference.py`), 30
warmup + 300 iterations:

- **`wall_mean`** — `time.perf_counter` around each call + `cuda.synchronize()`.
  Includes Python/CPU dispatch overhead. This is what the official
  `kernels benchmark` runner reports.
- **`gpu_mean`** — `cuda.Event` timing over the whole iteration loop / iters, so
  per-iter CPU launch overlaps with GPU execution. This reflects steady-state
  GPU throughput — the right metric for a memory-bound op.

`GB/s(gpu)` = bytes moved (read `x` + write `out` = `elems·4·2`) ÷ `gpu_mean`.
ReLU is **memory-bound**, so higher GB/s is better; B200 peak HBM is ~8 TB/s.

## Results

Numbers below are stable to ~1% across three runs.

```
      shape   elems          kernel  wall_mean  gpu_mean  GB/s(gpu)  vs torch   ok
-------------------------------------------------------------------------------------
  1024x1024      1M      torch.relu     0.0139    0.0057     1482.8     1.00x  True
  1024x1024      1M  reference-cuda     0.0161    0.0055     1537.2     1.04x  True
  1024x1024      1M          helion     0.0438    0.0323      259.7     0.18x  True

  4096x4096     16M      torch.relu     0.0308    0.0195     6873.8     1.00x  True
  4096x4096     16M  reference-cuda     0.0487    0.0381     3521.5     0.51x  True
  4096x4096     16M          helion     0.0665    0.0323     4156.2     0.60x  True

  8192x8192     67M      torch.relu     0.0916    0.0801     6702.1     1.00x  True
  8192x8192     67M  reference-cuda     0.1340    0.1231     4361.8     0.65x  True
  8192x8192     67M          helion     0.1192    0.0809     6635.6     0.99x  True
```

(`vs torch` = GPU-time speedup relative to `torch.relu`; `ok` = `allclose` vs
`F.relu`, atol 1e-3.)

## Analysis

The story depends entirely on problem size, because ReLU does almost no
arithmetic — performance is a race between launch overhead and HBM bandwidth.

- **Small (1M elems): Helion loses badly (0.18× torch).** At this size the whole
  op is a few microseconds of GPU work, so *dispatch* dominates. Helion's path
  (a `torch.library.custom_op` wrapping a JIT-compiled Triton kernel) has
  materially higher per-call overhead than a thin C++ `torch.ops` shim or a
  native elementwise kernel. The GPU-time gap (0.0323 vs ~0.0055 ms) is a
  fixed-cost launch penalty, not a compute deficiency.

- **Medium (16M): Helion (4156 GB/s) already beats the reference C++ kernel
  (3521 GB/s)**, though both trail torch (6874). Launch overhead is amortizing;
  the autotuned Helion kernel moves memory faster than the hand-written
  reference.

- **Large (67M, bandwidth-bound): Helion ≈ torch and clearly beats the
  reference.** Helion hits **6636 GB/s — 99% of native `torch.relu` (6702)** and
  **~1.52× the reference C++ kernel (4362 GB/s)**. At this size the autotuned
  Helion kernel is essentially bandwidth-optimal on the B200.

### Takeaways

1. **The autotuned Helion kernel is competitive where it matters.** For large,
   bandwidth-bound tensors it matches PyTorch's native kernel and beats the
   precompiled reference by ~1.5×.
2. **Helion's weakness here is per-launch overhead**, which only hurts at tiny
   sizes where a single elementwise ReLU is too small to fill the GPU. This is
   the JIT/custom-op dispatch cost, not the generated kernel.
3. **Portability is a real, separate win.** Being `torch-noarch`, the Helion
   kernel ran on torch 2.14 out of the box; the precompiled reference could not
   run there at all. JIT kernels dodge the ABI-lock-per-torch-version treadmill.
4. **Caveat — autotuning cost.** These Helion numbers assume a one-time autotune
   (~5 min/shape here), cached thereafter. Without autotuning
   (`HELION_AUTOTUNE_EFFORT=none`), the default config is markedly slower; the
   autotune is what closes the gap to `torch.relu` at large sizes.

## Reproduce

```bash
# In the torch 2.13 venv, with the kernel already built (`kernel-builder build`):
~/venv_torch_213/bin/python benchmarks/compare_vs_reference.py
# First run per shape autotunes Helion (~5 min); subsequent runs reuse the cache.
```
