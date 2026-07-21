#!/usr/bin/env python
"""One-click AOT pre-tuning for every Helion kernel in this repo.

Usage (from anywhere, after activating the venv):

    python scripts/aot_tune.py            # tune ALL kernels, 7 shapes each
    python scripts/aot_tune.py rwkv fp8   # tune only the named kernels
    python scripts/aot_tune.py --effort quick   # override effort for this run

Autotuner choice comes from the ENVIRONMENT, not this script — export
``HELION_AUTOTUNER`` (and ``HELION_LLM_PROVIDER``/``HELION_LLM_MODEL`` for the
LLM-guided tuner) as usual; unset means Helion's default (LFBOTreeSearch). E.g.:

    HELION_AUTOTUNER=LLMGuidedSearch HELION_LLM_PROVIDER=bedrock \\
      HELION_LLM_MODEL=... python scripts/aot_tune.py

Effort follows the same rule: ``--effort {quick,full}`` overrides for this run,
else the ambient ``HELION_AUTOTUNE_EFFORT``, else Helion's default. Progress is
printed per kernel (index, done/remaining, elapsed, ETA).

What it does, per kernel:
  1. runs Helion's AOT collect -> measure -> build workflow across 7 input
     shapes (small..large) using the DEFAULT autotuner (LFBOTreeSearch),
  2. writes the per-shape heuristic next to the kernel's runtime source as
     ``_helion_aot_<file>_<device>_<compute>.py`` (e.g. ``..._cuda_sm100.py``),
  3. copies that heuristic back into ``torch-ext/`` (the committed source of
     truth) so a fresh clone ships the pre-tuned configs.

Machine-general: the target filename embeds THIS machine's GPU
(device_kind + compute capability), detected by Helion. Run it on each GPU you
care about and commit the resulting ``_helion_aot_*_<dev>_<compute>.py`` files;
they coexist, and Helion picks the right one (with older-arch fallback) at load.

Zero hardcoded paths: the repo root is derived from this file's location, so
``git clone ... && <activate venv> && python scripts/aot_tune.py`` just works.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

# ------------------------------------------------------------------ paths ----
# repo root = parent of the scripts/ dir this file lives in (no hardcoding).
HH = Path(__file__).resolve().parent.parent
if str(HH) not in sys.path:
    sys.path.insert(0, str(HH))

import torch  # noqa: E402
import kernels  # noqa: E402


def local(repo: str):
    """Load a built noarch kernel by its <repo>/<repo>/result path."""
    return kernels.get_local_kernel(HH / repo / repo / "result", "cuda")


# ---------------------------------------------------------------- kernels ----
# Each kernel provides:
#   repo:     "<name>-helion" directory
#   shapes(): list of 7 shape tuples, small -> large
#   run(m, shape): invoke the kernel once for a shape (on CUDA)
# Shapes are chosen to span the regimes a user is likely to hit; the heuristic
# generalizes across them (and emits a decision tree when they diverge).

def _f16(*s):
    return torch.randn(*s, device="cuda", dtype=torch.float16)


def _bf16(*s):
    return torch.randn(*s, device="cuda", dtype=torch.bfloat16)


def _f32(*s):
    return torch.randn(*s, device="cuda", dtype=torch.float32)


def sh_activation():
    # (B, S, D2) with D2 the gate+value width; 7 sizes.
    return [(8, 512, 1024), (8, 1024, 2048), (8, 1024, 4096), (8, 2048, 4096),
            (8, 2048, 8192), (8, 4096, 8192), (16, 4096, 8192)]


def run_activation(m, s):
    B, S, D2 = s
    d = D2 // 2
    m.silu_and_mul(torch.empty(B, S, d, device="cuda", dtype=torch.float16), _f16(B, S, D2))


def sh_rotary():
    return [(2, 128, 8, 32), (4, 256, 16, 64), (8, 512, 32, 64), (8, 1024, 32, 64),
            (16, 1024, 32, 64), (16, 2048, 32, 64), (32, 2048, 32, 128)]


def run_rotary(m, s):
    B, S, H, R = s
    x1, x2 = _f32(B, S, H, R), _f32(B, S, H, R)
    cos, sin = _f32(S, 1, R), _f32(S, 1, R)
    m.apply_rotary(x1, x2, cos, sin)


def sh_layer_norm():
    return [(256, 768), (1024, 1024), (2048, 2048), (4096, 4096),
            (8192, 4096), (16384, 8192), (32768, 8192)]


def run_layer_norm(m, s):
    M, N = s
    m.dropout_add_ln_fwd(input=_f16(M, N), gamma=torch.ones(N, device="cuda", dtype=torch.float16),
                         epsilon=1e-5, is_rms_norm=True)


def sh_causal_conv1d():
    return [(8, 768, 512), (8, 1024, 1024), (16, 2048, 2048), (16, 2048, 4096),
            (32, 2048, 4096), (32, 4096, 4096), (32, 4096, 8192)]


def run_causal_conv1d(m, s):
    B, D, L = s
    m.causal_conv1d_fn(_f16(B, D, L), _f16(D, 4), _f16(D), activation="silu")


def sh_fp8():
    return [(512, 512, 2048), (1024, 1024, 2048), (2048, 2048, 4096), (4096, 2048, 4096),
            (4096, 4096, 8192), (8192, 4096, 8192), (8192, 8192, 8192)]


def run_fp8(m, s):
    M, N, K = s
    bn, bk = 128, 128
    A = (torch.randn(M, K, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    B = (torch.randn(N, K, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    As = torch.rand(M, K // bk, device="cuda") * 0.5 + 0.5
    Bs = torch.rand(N // bn, K // bk, device="cuda") * 0.5 + 0.5
    m.w8a8_block_fp8_matmul(A, B, As, Bs, [bn, bk], torch.bfloat16)


def sh_mamba():
    return [(2, 256, 128, 16), (2, 512, 256, 16), (4, 1024, 512, 16), (4, 2048, 512, 16),
            (8, 2048, 1024, 16), (8, 4096, 1024, 16), (16, 4096, 1024, 16)]


def run_mamba(m, s):
    B, D, L, N = s
    u, delta = _f16(B, D, L), torch.rand(B, D, L, device="cuda", dtype=torch.float16)
    A = -torch.rand(D, N, device="cuda", dtype=torch.float32)
    Bm, Cm = _f16(B, N, L), _f16(B, N, L)
    Dp = _f32(D)
    z = _f16(B, D, L)
    m.selective_scan_fn(u, delta, A, Bm, Cm, D=Dp, z=z, delta_softplus=True)


def sh_paged():
    return [(16, 8, 8, 64, 16, 8), (16, 8, 8, 64, 16, 16), (32, 16, 8, 64, 16, 16),
            (32, 16, 8, 128, 16, 16), (64, 16, 8, 128, 16, 16), (64, 32, 8, 128, 16, 16),
            (128, 32, 8, 128, 16, 16)]


def run_paged(m, s):
    S, H, KVH, D, bs, mb = s
    nb, xx = S * mb, 8
    q = _f16(S, H, D)
    kc = _f16(nb, KVH, D // xx, bs, xx)
    vc = _f16(nb, KVH, D, bs)
    sl = torch.randint(1, mb * bs, (S,), device="cuda", dtype=torch.int32)
    bt = torch.randint(0, nb, (S, mb), device="cuda", dtype=torch.int32)
    ks = torch.tensor(1.0, device="cuda")
    vs = torch.tensor(1.0, device="cuda")
    m.paged_attention_v1(torch.empty_like(q), q, kc, vc, KVH, 1.0 / math.sqrt(D),
                         bt, sl, bs, mb * bs, None, "auto", ks, vs)


def sh_megablocks():
    return [(8, 2048, 1024, 1024), (8, 4096, 2048, 2048), (16, 8192, 2048, 2048),
            (16, 8192, 4096, 4096), (32, 16384, 4096, 4096), (32, 16384, 8192, 8192),
            (64, 16384, 8192, 8192)]


def run_megablocks(m, s):
    G, Mtot, K, N = s
    sizes = torch.full((G,), Mtot // G, dtype=torch.int64)
    total = int(sizes.sum())
    m.gmm(_bf16(total, K), _bf16(G, K, N), torch.empty(total, N, device="cuda", dtype=torch.bfloat16),
          sizes.to("cuda"))


def sh_tinygrad_rms():
    return [(2048, 1024), (4096, 1024), (8192, 1024), (16384, 1024),
            (32768, 1024), (65536, 1024), (131072, 1024)]


def run_tinygrad_rms(m, s):
    M, N = s
    m.tinygrad_rms_norm(_f32(M, N), torch.ones(N, device="cuda", dtype=torch.float32), 1e-6)


def sh_rwkv():
    return [(4, 256, 512), (8, 512, 1024), (8, 1024, 1024), (16, 1024, 2048),
            (16, 2048, 2048), (32, 2048, 2048), (32, 2048, 4096)]


def run_rwkv(m, s):
    B, T, C = s
    w, u = _f32(C), _f32(C)
    k = _f32(B, T, C) * 0.5
    v = _f32(B, T, C)
    m.wkv(w, u, k, v)


def sh_deformable():
    return [(2, 8, 32, 4, 16, 900), (4, 8, 32, 8, 32, 2000), (8, 8, 32, 8, 32, 2000),
            (8, 8, 64, 8, 48, 4000), (8, 8, 64, 8, 64, 4000), (16, 8, 64, 8, 64, 4000),
            (16, 8, 64, 8, 64, 8000)]


def run_deformable(m, s):
    B, H, Dh, P, G, Nq = s
    Nkv = G * G
    m.ms_deform_attn(_f32(B, Nkv, H, Dh),
                     torch.rand(B, Nq, H, P, 2, device="cuda", dtype=torch.float32),
                     torch.rand(B, Nq, H, P, device="cuda", dtype=torch.float32))


def sh_attention():
    return [(2, 512, 8, 64), (4, 512, 16, 64), (4, 1024, 16, 64), (8, 1024, 16, 128),
            (8, 2048, 16, 128), (16, 2048, 16, 128), (16, 4096, 16, 128)]


def run_attention(m, s):
    B, S, H, D = s
    m.flash_attn_func(_bf16(B, S, H, D), _bf16(B, S, H, D), _bf16(B, S, H, D), causal=False)


def sh_sage():
    return [(2, 512, 8, 128), (4, 512, 16, 128), (4, 1024, 16, 128), (8, 1024, 16, 128),
            (8, 2048, 16, 128), (16, 2048, 16, 128), (16, 4096, 16, 128)]


def run_sage(m, s):
    B, S, H, D = s  # sage takes HND (B, H, S, D)
    q, k, v = _bf16(B, H, S, D), _bf16(B, H, S, D), _bf16(B, H, S, D)
    m.sageattn(q, k, v, tensor_layout="HND", is_causal=False)


# name -> (repo dir, shapes fn, run fn)
KERNELS = {
    "activation": ("activation-helion", sh_activation, run_activation),
    "rotary": ("rotary-helion", sh_rotary, run_rotary),
    "layer_norm": ("layer-norm-helion", sh_layer_norm, run_layer_norm),
    "causal_conv1d": ("causal-conv1d-helion", sh_causal_conv1d, run_causal_conv1d),
    "fp8": ("finegrained-fp8-helion", sh_fp8, run_fp8),
    "mamba": ("mamba-ssm-helion", sh_mamba, run_mamba),
    "paged": ("paged-attention-helion", sh_paged, run_paged),
    "megablocks": ("megablocks-helion", sh_megablocks, run_megablocks),
    "tinygrad_rms": ("tinygrad-rms-helion", sh_tinygrad_rms, run_tinygrad_rms),
    "rwkv": ("rwkv-helion", sh_rwkv, run_rwkv),
    "deformable": ("deformable-detr-helion", sh_deformable, run_deformable),
    "attention": ("attention-helion", sh_attention, run_attention),
    "sage": ("sage-attention-helion", sh_sage, run_sage),
}


# ----------------------------------------------------- driver (child) mode ---
def _drive_one(name: str) -> None:
    """Invoked in the aot_runner subprocess: run the kernel over all its shapes.
    HELION_AOT_MODE (set by the runner) controls collect/measure/evaluate."""
    repo, shapes_fn, run_fn = KERNELS[name]
    m = local(repo)
    for s in shapes_fn():
        run_fn(m, s)
        torch.cuda.synchronize()


# --------------------------------------------------- orchestrator (parent) ---
def _hardware_tag() -> str:
    from helion._hardware import get_hardware_info

    hw = get_hardware_info()
    return f"{hw.device_kind}_{hw.compute_capability}"


def _sync_heuristics() -> None:
    """Copy generated build/torch-cuda/_helion_aot_*.py -> torch-ext/ (source of
    truth), and (re)build the noarch layout so result/ points at fresh files."""
    import shutil

    for heur in HH.glob("*/*/build/torch-cuda/_helion_aot_*.py"):
        te = heur.parent.parent.parent / "torch-ext"
        pkgs = [d for d in te.iterdir() if d.is_dir() and d.name != "__pycache__"]
        if pkgs:
            shutil.copy2(heur, pkgs[0] / heur.name)


def _orchestrate(names: list[str], effort: str | None) -> None:
    tag = _hardware_tag()
    # The autotuner is chosen by the ambient environment, NOT by this script:
    # whatever HELION_AUTOTUNER (+ any HELION_LLM_* for the LLM tuner) you export
    # is passed straight through. Unset -> Helion's default (LFBOTreeSearch).
    env = dict(os.environ)
    autotuner = env.get("HELION_AUTOTUNER") or "default (LFBOTreeSearch)"
    # Effort: honor an explicit --effort flag; else an ambient
    # HELION_AUTOTUNE_EFFORT; else Helion's own default (full).
    if effort is not None:
        env["HELION_AUTOTUNE_EFFORT"] = effort
    eff = env.get("HELION_AUTOTUNE_EFFORT", "full (helion default)")
    # Operational defaults required for AOT-tuning get_local_kernel modules
    # (harmless to the autotuner choice); keep any value the user already set.
    env.setdefault("HELION_AUTOTUNE_BENCHMARK_SUBPROCESS", "0")
    env.setdefault("HELION_AUTOTUNE_IGNORE_ERRORS", "1")
    print(f"=== AOT pre-tuning {len(names)} kernel(s) for {tag} "
          f"(autotuner={autotuner}, effort={eff}, 7 shapes each) ===", flush=True)

    import time

    def _hms(secs: float) -> str:
        secs = int(secs)
        h, rem = divmod(secs, 3600)
        mnt, sec = divmod(rem, 60)
        return f"{h:d}h{mnt:02d}m{sec:02d}s" if h else f"{mnt:d}m{sec:02d}s"

    total = len(names)
    ok, failed = [], []
    t_start = time.monotonic()
    for i, name in enumerate(names, 1):
        done = i - 1
        remaining = total - done
        elapsed = time.monotonic() - t_start
        eta = (elapsed / done * remaining) if done else 0.0
        print(f"\n########## [{i}/{total}] {name}  "
              f"(done={done}, remaining={remaining}, "
              f"elapsed={_hms(elapsed)}"
              + (f", eta~{_hms(eta)}" if done else "") + ") ##########",
              flush=True)
        t_k = time.monotonic()
        # One aot_runner invocation per kernel; the runner spawns THIS file in
        # driver mode for each phase. A crash in one kernel can't kill the sweep.
        cmd = [sys.executable, "-m", "helion.autotuner.aot_runner",
               "--phase", "all", "--goal", "max_slowdown", "--threshold", "1.15",
               "--max-configs", "8", "-k", name,
               "--", sys.executable, str(Path(__file__).resolve()), "--_drive", name]
        rc = subprocess.run(cmd, env=env).returncode
        _sync_heuristics()
        (ok if rc == 0 else failed).append(name)
        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        print(f"---- [{i}/{total}] {name} {status} in {_hms(time.monotonic() - t_k)} "
              f"| ok={len(ok)} failed={len(failed)} remaining={total - i} ----",
              flush=True)

    print(f"\n=== DONE for {tag} in {_hms(time.monotonic() - t_start)}: "
          f"{len(ok)}/{total} ok, {len(failed)} failed ===", flush=True)
    if ok:
        print(f"  ok: {', '.join(ok)}", flush=True)
    if failed:
        print(f"  failed: {', '.join(failed)}", flush=True)
        print("  (re-run those names, or try --effort quick — some configs can "
              "hit backend-specific crashes under full effort)", flush=True)
    print("  heuristics written to <kernel>/torch-ext/<pkg>/_helion_aot_*_"
          f"{tag}.py — commit them.", flush=True)


def main() -> None:
    args = sys.argv[1:]

    # Child/driver mode: `--_drive <name>` (called by aot_runner subprocess).
    if args and args[0] == "--_drive":
        _drive_one(args[1])
        return

    # Effort is optional: only override when --effort is passed; otherwise the
    # ambient HELION_AUTOTUNE_EFFORT (or Helion's default) applies.
    effort = None
    if "--effort" in args:
        i = args.index("--effort")
        effort = args[i + 1]
        del args[i:i + 2]

    names = args or list(KERNELS)
    unknown = [n for n in names if n not in KERNELS]
    if unknown:
        sys.exit(f"unknown kernel(s): {unknown}. valid: {list(KERNELS)}")
    _orchestrate(names, effort)


if __name__ == "__main__":
    main()
