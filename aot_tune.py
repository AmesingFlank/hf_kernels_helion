"""AOT tuning driver for the hf_kernels_helion kernels.

Invoked by `helion.experimental.aot_runner` across its phases (collect/measure/
build/evaluate) via HELION_AOT_MODE. For the given kernel (argv[1]), it loads the
built noarch kernel via get_local_kernel and calls it on the SAME benchmark
shapes used in rebench_llm.py, so the pre-tuned heuristic covers exactly the
shapes we benchmark.

The build phase writes the heuristic next to the kernel's runtime source
(build/torch-cuda/<file>.py) since that's the co_filename Helion sees. A
post-step (sync_aot_heuristics.py) copies it into torch-ext/ as source of truth.

Usage (via runner):
  python -m helion.experimental.aot_runner --phase all -k activation \
    -- python aot_tune.py activation
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, "/home/dev")
sys.path.insert(0, "/home/dev/hf_kernels_helion")
import torch
import kernels

HH = "/home/dev/hf_kernels_helion"


def local(n, s):
    return kernels.get_local_kernel(Path(f"{HH}/{n}/{s}/result"), "cuda")


# Each entry: name -> callable that runs the kernel across the 3 bench shapes,
# mirroring rebench_llm.py EXACTLY so the heuristic covers the benchmark shapes.

def run_activation():
    m = local("activation-helion", "activation-helion")
    for (B, S, D2) in [(8, 1024, 2048), (8, 2048, 4096), (8, 4096, 8192)]:
        d = D2 // 2
        x = torch.randn(B, S, D2, device="cuda", dtype=torch.float16)
        out = torch.empty(B, S, d, device="cuda", dtype=torch.float16)
        m.silu_and_mul(out, x)
        torch.cuda.synchronize()


def run_rotary():
    m = local("rotary-helion", "rotary-helion")
    for (B, S, H, R) in [(2, 128, 8, 32), (8, 512, 32, 64), (16, 2048, 32, 64)]:
        x1 = torch.randn(B, S, H, R, device="cuda", dtype=torch.float32)
        x2 = torch.randn_like(x1)
        cos = torch.randn(S, 1, R, device="cuda")
        sin = torch.randn(S, 1, R, device="cuda")
        m.apply_rotary(x1, x2, cos, sin)
        torch.cuda.synchronize()


def run_layer_norm():
    m = local("layer-norm-helion", "layer-norm-helion")
    for (M, N) in [(256, 768), (2048, 2048), (16384, 8192)]:
        x = torch.randn(M, N, device="cuda", dtype=torch.float16)
        w = torch.ones(N, device="cuda", dtype=torch.float16)
        m.dropout_add_ln_fwd(input=x, gamma=w, epsilon=1e-5, is_rms_norm=True)
        torch.cuda.synchronize()


def run_causal_conv1d():
    m = local("causal-conv1d-helion", "causal-conv1d-helion")
    for (B, D, L) in [(8, 768, 512), (16, 2048, 2048), (32, 4096, 4096)]:
        x = torch.randn(B, D, L, device="cuda", dtype=torch.float16)
        w = torch.randn(D, 4, device="cuda", dtype=torch.float16)
        b = torch.randn(D, device="cuda", dtype=torch.float16)
        m.causal_conv1d_fn(x, w, b, activation="silu")
        torch.cuda.synchronize()


def run_fp8():
    m = local("finegrained-fp8-helion", "finegrained-fp8-helion")
    bn, bk = 128, 128
    for (M, N, K) in [(512, 512, 2048), (2048, 2048, 4096), (4096, 4096, 8192)]:
        A = (torch.randn(M, K, device="cuda") * 0.3).to(torch.float8_e4m3fn)
        B = (torch.randn(N, K, device="cuda") * 0.3).to(torch.float8_e4m3fn)
        As = torch.rand(M, K // bk, device="cuda") * 0.5 + 0.5
        Bs = torch.rand(N // bn, K // bk, device="cuda") * 0.5 + 0.5
        m.w8a8_block_fp8_matmul(A, B, As, Bs, [bn, bk], torch.bfloat16)
        torch.cuda.synchronize()


def run_mamba():
    m = local("mamba-ssm-helion", "mamba-ssm-helion")
    for (B, D, L, N) in [(2, 256, 128, 16), (4, 1024, 512, 16), (8, 2048, 1024, 16)]:
        u = torch.randn(B, D, L, device="cuda", dtype=torch.float16)
        delta = torch.rand(B, D, L, device="cuda", dtype=torch.float16)
        A = -torch.rand(D, N, device="cuda", dtype=torch.float32)
        Bm = torch.randn(B, N, L, device="cuda", dtype=torch.float16)
        Cm = torch.randn(B, N, L, device="cuda", dtype=torch.float16)
        Dp = torch.randn(D, device="cuda", dtype=torch.float32)
        z = torch.randn(B, D, L, device="cuda", dtype=torch.float16)
        m.selective_scan_fn(u, delta, A, Bm, Cm, D=Dp, z=z, delta_softplus=True)
        torch.cuda.synchronize()


def run_paged():
    m = local("paged-attention-helion", "paged-attention-helion")
    for (S, H, KVH, D, bs, mb) in [(16, 8, 8, 64, 16, 16), (32, 16, 8, 64, 16, 16)]:
        nb = S * mb
        xx = 8
        q = torch.randn(S, H, D, device="cuda", dtype=torch.float16)
        kc = torch.randn(nb, KVH, D // xx, bs, xx, device="cuda", dtype=torch.float16)
        vc = torch.randn(nb, KVH, D, bs, device="cuda", dtype=torch.float16)
        sl = torch.randint(1, mb * bs, (S,), device="cuda", dtype=torch.int32)
        bt = torch.randint(0, nb, (S, mb), device="cuda", dtype=torch.int32)
        scale = 1.0 / math.sqrt(D)
        ks = torch.tensor(1.0, device="cuda")
        vs = torch.tensor(1.0, device="cuda")
        oh = torch.empty_like(q)
        m.paged_attention_v1(oh, q, kc, vc, KVH, scale, bt, sl, bs, mb * bs, None, "auto", ks, vs)
        torch.cuda.synchronize()


def run_megablocks():
    m = local("megablocks-helion", "megablocks-helion")
    # Fixed group sizes (no RNG in this env) so shapes are deterministic.
    for (G, Mtot, K, N) in [(8, 2048, 1024, 1024), (16, 8192, 2048, 2048), (32, 16384, 4096, 4096)]:
        per = Mtot // G
        sizes = torch.full((G,), per, dtype=torch.int64)
        total = int(sizes.sum())
        a = torch.randn(total, K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(G, K, N, device="cuda", dtype=torch.bfloat16)
        c = torch.empty(total, N, device="cuda", dtype=torch.bfloat16)
        m.gmm(a, b, c, sizes.to("cuda"))
        torch.cuda.synchronize()


def run_tinygrad_rms():
    m = local("tinygrad-rms-helion", "tinygrad-rms-helion")
    N = 1024
    for M in [4096, 16384, 65536]:
        x = torch.randn(M, N, device="cuda", dtype=torch.float32)
        w = torch.ones(N, device="cuda", dtype=torch.float32)
        m.tinygrad_rms_norm(x, w, 1e-6)
        torch.cuda.synchronize()


def run_rwkv():
    m = local("rwkv-helion", "rwkv-helion")
    for (B, T, C) in [(4, 256, 512), (8, 1024, 1024), (16, 1024, 2048)]:
        w = torch.randn(C, device="cuda", dtype=torch.float32)
        u = torch.randn(C, device="cuda", dtype=torch.float32)
        k = torch.randn(B, T, C, device="cuda", dtype=torch.float32) * 0.5
        v = torch.randn(B, T, C, device="cuda", dtype=torch.float32)
        m.wkv(w, u, k, v)
        torch.cuda.synchronize()


def run_deformable():
    m = local("deformable-detr-helion", "deformable-detr-helion")
    for (B, H, Dh, P, G, Nq) in [(2, 8, 32, 4, 16, 900), (4, 8, 32, 8, 32, 2000), (8, 8, 64, 8, 48, 4000)]:
        Nkv = G * G
        value = torch.randn(B, Nkv, H, Dh, device="cuda", dtype=torch.float32)
        loc = torch.rand(B, Nq, H, P, 2, device="cuda", dtype=torch.float32)
        aw = torch.rand(B, Nq, H, P, device="cuda", dtype=torch.float32)
        m.ms_deform_attn(value, loc, aw)
        torch.cuda.synchronize()


def run_attention():
    m = local("attention-helion", "attention-helion")
    for (B, S, H, D) in [(2, 512, 8, 64), (4, 1024, 16, 64), (8, 2048, 16, 128)]:
        q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        m.flash_attn_func(q, k, v, causal=False)
        torch.cuda.synchronize()


def run_sage():
    m = local("sage-attention-helion", "sage-attention-helion")
    for (B, S, H, D) in [(2, 512, 8, 128), (4, 1024, 16, 128), (8, 2048, 16, 128)]:
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H, S, D, device="cuda", dtype=torch.bfloat16)
        m.sageattn(q, k, v, tensor_layout="HND", is_causal=False)
        torch.cuda.synchronize()


KERNELS = {
    "activation": run_activation,
    "rotary": run_rotary,
    "layer_norm": run_layer_norm,
    "causal_conv1d": run_causal_conv1d,
    "fp8": run_fp8,
    "mamba": run_mamba,
    "paged": run_paged,
    "megablocks": run_megablocks,
    "tinygrad_rms": run_tinygrad_rms,
    "rwkv": run_rwkv,
    "deformable": run_deformable,
    "attention": run_attention,
    "sage": run_sage,
}


if __name__ == "__main__":
    import os
    name = sys.argv[1]
    mode = os.environ.get("HELION_AOT_MODE", "disabled")
    print(f"[aot_tune] kernel={name} mode={mode}", flush=True)
    KERNELS[name]()
    print(f"[aot_tune] {name} done ({mode})", flush=True)
