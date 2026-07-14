# Building a Helion ReLU kernel with `kernel-builder`

This documents building a [Helion](https://github.com/pytorch/helion) ReLU
kernel with Hugging Face `kernel-builder`, matching the tests and benchmark of
the reference [`kernels-community/relu`](https://huggingface.co/kernels-community/relu).

**Outcome: it works.** The Helion kernel builds, loads via `get_kernel`, passes
the full reference test suite, and benchmarks competitively (see
[`BENCHMARKS.md`](./BENCHMARKS.md)). No hacks were needed — but there are a few
non-obvious points and one genuine environment limitation, documented below.

## How Helion support works in `kernels` / `kernel-builder`

Helion support was added in [huggingface/kernels#706](https://github.com/huggingface/kernels/pull/706).
The PR is small and purely additive: it whitelists `helion` as an allowed value
of the experimental `python-depends` build option (in `python_depends.json` and
`kernel-requirements.md`) and unbreaks the `helion` nixpkgs package for
non-CUDA builds. **It does not add a new framework section or an example
kernel.**

Consequently, a Helion kernel is expressed with the machinery that already
exists:

- **Framework section `[torch-noarch]`** — Helion kernels are JIT-compiled at
  runtime (Helion emits Triton), so there is nothing to compile ahead of time.
  This is the same section the `relu-triton` example uses.
- **`python-depends = ["helion"]`** in `[general]` — declares the runtime
  dependency. It is written into `metadata.json` and checked by
  `kernels.get_kernel` / `get_local_kernel` at load time via
  `validate_dependencies` (`kernels/deps.py`): if `helion` is not importable,
  loading raises a clear `ImportError`.

The `relu-triton` example kernel is the ideal structural template — swap its
Triton `op.py` for a Helion implementation and the rest (bindings, layers,
tests) is identical.

## Kernel structure

```
relu-helion/
├── build.toml                       # [torch-noarch] + python-depends=["helion"]
├── torch-ext/relu_helion/
│   ├── op.py                        # @helion.kernel relu + torch.library.custom_op
│   ├── __init__.py                  # relu(x, out=None) public API
│   └── layers.py                    # nn.Module ReLU layer
├── tests/
│   ├── conftest.py                  # device fixture
│   └── test_relu_helion.py          # mirrors kernels-community/relu tests
└── benchmarks/
    ├── benchmark.py                 # kernels.benchmark.Benchmark (matches reference)
    └── compare_vs_reference.py      # head-to-head vs reference C++ kernel
```

The op is registered as a `torch.library.custom_op` that mutates `out`, exactly
like `relu-triton`, so the public API matches the reference kernel
(`relu(x, out=None)` and a `layers.ReLU` module):

```python
@helion.kernel(static_shapes=False)
def _relu_helion(out: torch.Tensor, x: torch.Tensor) -> None:
    for tile in hl.tile(x.size()):
        out[tile] = torch.relu(x[tile])

@torch.library.custom_op(add_op_namespace_prefix("relu"), mutates_args={"out"})
def _relu(out: torch.Tensor, x: torch.Tensor) -> None:
    _relu_helion(out, x)
```

## Build & test commands

```bash
# From the kernel directory (must be a git repo with everything committed —
# nix flakes only see tracked files):
kernel-builder build -L                 # builds the torch-cuda (noarch) variant

# Correctness (against the built bundle; HELION_AUTOTUNE_EFFORT=none uses a
# default config and skips the ~5 min autotune):
HELION_AUTOTUNE_EFFORT=none \
  PYTHONPATH=result/torch-cuda \
  python -m pytest tests/ -v
# => 5 passed  (test_relu[fp32/fp16/bf16], test_relu_views, test_relu_layer)
```

## Non-obvious points encountered

1. **Must be a committed git repo.** `kernel-builder build` failed with *"Kernel
   is not in a git repository, this will create a non-reproducible build"* until
   the scaffold was `git init` + committed. Nix flakes only copy git-tracked
   files into the store, so *uncommitted* changes are invisible to the build.

2. **`kernel-builder init` only scaffolds AOT CUDA kernels.** There is no
   `--framework noarch` flag. You scaffold, then convert: set `[torch-noarch]`,
   add `python-depends`, delete the generated `*.cu` / `torch_binding.*`, and
   write `op.py`. The generated `flake.nix` works unchanged.

3. **Autotuning cost dominates first use.** A full Helion autotune of this
   trivial kernel took **~275–310 s per shape** on a B200. For correctness runs
   use `HELION_AUTOTUNE_EFFORT=none`; for benchmarks, autotune once (results are
   cached under `$TORCHINDUCTOR_CACHE_DIR/helion`, reused on later runs).

4. **Nix builds `helion` from PyPI/source, not your local checkout.** The build
   pulls `helion` (and its closure: scikit-learn, scipy, joblib, …) from
   nixpkgs and compiles/tests them — the first build is long. This is the
   nixpkgs `helion` that PR #706 unbroke, independent of any `~/helion` editable
   install. The nix sandbox has no GPU, so the build only verifies the kernel
   *loads* (`get_kernel`), not that it executes — runtime correctness must be
   checked on the host (it passes; see tests above).

## Genuine limitation: precompiled reference can't run on torch 2.14

Not a Helion/`kernel-builder` bug, but it shaped the benchmark setup and is
worth recording:

- This machine runs **torch 2.14.0a0** (a dev build). Torch extensions are
  ABI-locked to a torch minor version.
- The reference `kernels-community/relu` is a **precompiled C++ CUDA** kernel
  whose Hub build variants only go up to **torch 2.13**. `kernel-builder`
  0.17.0-dev0 itself also only emits variants up to torch 2.13. So the reference
  **cannot be loaded or rebuilt for torch 2.14** — `get_kernel` fails with
  *"Cannot find a build variant for this system"*.
- The **Helion kernel is unaffected**: `[torch-noarch]` means it JIT-compiles at
  runtime against whatever torch is present, so it runs natively on torch 2.14.

To get a fair, kernel-vs-kernel comparison, the benchmark was run in a dedicated
**torch 2.13** virtual environment (created with `uv`, see `BENCHMARKS.md`)
where *both* the precompiled reference and the (portable) Helion kernel load.
This is a portability win for JIT/noarch kernels, and the reason the benchmark
lives in its own venv.
