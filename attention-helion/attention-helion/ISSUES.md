# Attention-Helion — notes on this environment

Environment: **NVIDIA B200** (sm_100 / Blackwell), torch **2.14.0a0+git**, Triton
**3.6.0**, Helion (editable `~/helion`, dev build). CUDA driver 13.2.

The kernel **builds, autotunes, and runs correctly** — both non-causal and
causal. There is one real external limitation (flash-attn3 has no Blackwell
binary) and one *operational gotcha* that initially looked like a Helion bug but
is actually a documented `multiprocessing` requirement.

## 1. Autotuner + the `if __name__ == "__main__"` guard (RESOLVED — not a bug)

**Symptom:** running an autotuning attention kernel aborted with

```
helion.exc.NoConfigFound: No working config found from autotuning
# often preceded by: "failed to send job to worker", "benchmark timeout after 30.0s"
```

**Root cause:** Helion benchmarks candidate configs in a **spawned** subprocess
(so a hung kernel can be killed without losing autotune progress). Python's
`spawn` start method **re-imports the entry module** in every worker. If a kernel
is *invoked at module top level* without an `if __name__ == "__main__"` guard,
each worker re-runs that top-level call on import, which starts *its own*
autotune, which spawns more workers… a recursive spawn storm. The workers can't
service benchmark jobs, every config "fails", and the search ends in
`NoConfigFound`.

This is **documented Helion behavior**, not a bug in Helion, the `kernels`
package, or `kernel-builder` — see `~/helion/docs/api/autotuner.md`:

> Guard your entry script with `if __name__ == "__main__":`. … the worker
> re-runs it on import and autotuning aborts with `NoConfigFound` (often with
> `failed to send job to worker`). Either add the guard, or set
> `HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=0` to benchmark in-process.

(57 of 58 upstream Helion examples use the guard.)

**Proof it's the guard, not the GPU:**

| scenario | autotune procs | result |
|---|---|---|
| kernel called at module top level, **no guard** | 5–11 (fork storm) | `NoConfigFound` |
| same call under `if __name__ == "__main__"` | 1 | ✅ completes, config found |
| inside a pytest test / benchmark method | 1 | ✅ completes |

**What this means here:** the kernel's own callers — the `custom_op` impl, the
tests, and the benchmarks — all invoke the kernel *inside functions/methods*, so
they autotune cleanly. Only ad-hoc probe scripts that called the kernel at top
level tripped the storm. The shipped `example.py` uses the guard (and says why).

**How this kernel handles it:** the `@helion.kernel` functions are left to
**autotune per shape and cache** (under `$TORCHINDUCTOR_CACHE_DIR/helion`),
rather than pinning a config. First use of each shape pays a one-time autotune
(~3 min); subsequent runs load from cache.

## 2. Causal attention works (with autotuning)

Earlier this looked broken: with `HELION_AUTOTUNE_EFFORT=none`, causal fails to
compile —

```
triton.compiler.errors.CompilationError: reshape() cannot change total number
of elements in tensor   # in the torch.where + amax(..., keepdim=True) path
```

— because Helion's *default* config can't compile the causal reduction. But
**autotuning finds working configs**: a guarded forced autotune completed in
181s (225/558 candidate configs failed to compile, which is normal — the search
discards them) and produced a correct kernel. So causal is fully functional; it
just *must* be autotuned, not run with `autotune_effort="none"`.

## 3. flash-attn3 reference cannot run on B200 (real, external)

`kernels-community/flash-attn3` **loads** on torch 2.14 (`torch-stable-abi`
variants are forward-compatible) but **fails at execution**:

```
CUDA error: no kernel image is available for execution on the device
```

`cuobjdump --list-elf` on its `.so` shows only **sm_80 (102) + sm_90 (302)**
images — **no sm_100**. FA3's kernels target Hopper; the Hub build has no
Blackwell code, so it can't run on the B200 on any torch version. The attention
benchmark therefore uses **PyTorch SDPA** (its flash backend runs on B200 and is
what `kernels`' own attention util verifies against) as the baseline.

> Launching FA3 poisons the CUDA context (later kernels in the same process then
> fail), so FA3 is probed only in a throwaway process (`benchmarks/check_flash_attn3.py`).

## Summary

| Capability | Status on B200 |
|---|---|
| Build (`kernel-builder build`) | ✅ |
| Load via `get_kernel` / `get_local_kernel` | ✅ |
| Non-causal attention (autotuned) | ✅ correct |
| Causal attention (autotuned) | ✅ correct |
| Autotuning | ✅ works — **caller must guard top-level calls with `if __name__=="__main__"`** |
| `autotune_effort="none"` (causal) | ❌ default config can't compile causal; autotune instead |
| flash-attn3 reference | ❌ no sm_100 SASS — can't execute on Blackwell |
