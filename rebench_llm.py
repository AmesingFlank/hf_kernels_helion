"""Re-benchmark all Helion kernels vs their real kernels-community references,
using LLM-guided autotuning (Bedrock/Claude). Records per-kernel LLM autotune
time. Writes results to /tmp/rebench_<kernel>.json.

Run ONE kernel per process invocation (argv[1]) so a fork/spawn issue in one
can't take down the others, and so autotune time is cleanly attributable.

Usage: python rebench_llm.py <kernel_name>
"""
from __future__ import annotations
import os
# --- LLM-guided autotuning config (must be set before helion import) ---
os.environ.setdefault("HELION_AUTOTUNER", "LLMGuidedSearch")
os.environ.setdefault("HELION_LLM_PROVIDER", "bedrock")
os.environ.setdefault("HELION_LLM_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
os.environ.setdefault("AWS_REGION", "us-east-2")
# In-process benchmarking avoids the spawn-reimport recursion for get_local_kernel
# modules (they don't live under __main__, so the __main__ guard can't protect them).
os.environ.setdefault("HELION_AUTOTUNE_BENCHMARK_SUBPROCESS", "0")
os.environ.setdefault("HELION_SKIP_CACHE", "1")
os.environ.setdefault("HELION_AUTOTUNE_IGNORE_ERRORS", "1")

import sys, json, time, math
from pathlib import Path
sys.path.insert(0, "/home/dev/hf_kernels_helion")
import torch
import kernels

HH = "/home/dev/hf_kernels_helion"
def local(n, s): return kernels.get_local_kernel(Path(f"{HH}/{n}/{s}/result"), "cuda")
HH_REFS = str(Path.home() / "hf_kernels_refs")
def ref(repo):
    """Load the reference kernel from a LOCAL copy under ~/hf_kernels_refs
    (fetched once from the Hub), so benchmarking makes zero Hub calls and can't
    be throttled. `repo` is the kernels-community repo name (e.g. 'layer-norm')."""
    return kernels.get_local_kernel(Path(f"{HH_REFS}/{repo}/variant"), "cuda")


def bench_gpu(fn, warmup=15, iters=60):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def timed_autotune(fn):
    """First call triggers LLM autotune; time it, then return (result, autotune_seconds)."""
    t0 = time.time()
    out = fn()
    torch.cuda.synchronize()
    return out, time.time() - t0


RESULTS = []
_SHAPE_FILTER = os.environ.get("REBENCH_SHAPE")  # e.g. "small"; None = all
_ROW_I = [0]
def record(kernel, size, helion_call, ref_call, atol, rtol):
    # Optionally run only one shape per process so each row's autotune time is
    # a genuine fresh measurement (no in-process config reuse across shapes).
    if _SHAPE_FILTER is not None and not size.startswith(_SHAPE_FILTER):
        return
    # autotune (timed) on first helion call
    oh, at = timed_autotune(helion_call)
    orf = ref_call(); torch.cuda.synchronize()
    h = oh[0] if isinstance(oh, tuple) else oh
    r = orf[0] if isinstance(orf, tuple) else orf
    ok = bool(torch.allclose(h.float(), r.float(), atol=atol, rtol=rtol))
    hms = bench_gpu(helion_call)
    rms = bench_gpu(ref_call)
    row = {"kernel": kernel, "size": size, "helion_ms": round(hms, 4),
           "ref_ms": round(rms, 4), "speedup": round(rms / hms, 3),
           "autotune_s": round(at, 1), "ok": ok}
    RESULTS.append(row)
    print(f"  {size:>20} helion={hms:8.4f}ms ref={rms:8.4f}ms {rms/hms:6.2f}x autotune={at:6.1f}s ok={ok}", flush=True)


# ---------------- per-kernel head-to-heads ----------------

def activation():
    mine = local("activation-helion","activation-helion"); r = ref("activation")
    for nm,(B,S,D2) in {"small":(8,1024,2048),"medium":(8,2048,4096),"large":(8,4096,8192)}.items():
        d=D2//2; x=torch.randn(B,S,D2,device="cuda",dtype=torch.float16)
        oh=torch.empty(B,S,d,device="cuda",dtype=torch.float16); orf=torch.empty_like(oh)
        record("activation.silu_and_mul", f"{nm} {B}x{S}x{D2}",
               lambda: (mine.silu_and_mul(oh,x), oh)[1], lambda: (r.silu_and_mul(orf,x), orf)[1], 1e-2, 1e-2)

def rotary():
    mine = local("rotary-helion","rotary-helion"); r = ref("rotary")
    for nm,(B,S,H,R) in {"small":(2,128,8,32),"medium":(8,512,32,64),"large":(16,2048,32,64)}.items():
        x1=torch.randn(B,S,H,R,device="cuda",dtype=torch.float32); x2=torch.randn_like(x1)
        cos=torch.randn(S,1,R,device="cuda"); sin=torch.randn(S,1,R,device="cuda")
        r1=x1.clone(); r2=x2.clone()
        record("rotary.apply_rotary", f"{nm} {B}x{S}x{H}x{R}",
               lambda: mine.apply_rotary(x1,x2,cos,sin)[0],
               lambda: (r1.copy_(x1), r2.copy_(x2), r.apply_rotary(r1,r2,cos,sin,r1,r2,False), r1)[3], 1e-2, 1e-2)

def layer_norm():
    mine = local("layer-norm-helion","layer-norm-helion"); r = ref("layer-norm")
    for nm,(M,N) in {"small":(256,768),"medium":(2048,2048),"large":(16384,8192)}.items():
        x=torch.randn(M,N,device="cuda",dtype=torch.float16); w=torch.ones(N,device="cuda",dtype=torch.float16)
        def rc():
            rr=r.dropout_add_ln_fwd(x,w,None,None,None,None,None,0.0,1e-5,1.0,0,None,False,True)
            return rr[0] if isinstance(rr,(tuple,list)) else rr
        record("layer_norm.rms_fwd", f"{nm} {M}x{N}",
               lambda: mine.dropout_add_ln_fwd(input=x,gamma=w,epsilon=1e-5,is_rms_norm=True)[0], rc, 2e-2, 2e-2)

def causal_conv1d():
    mine = local("causal-conv1d-helion","causal-conv1d-helion"); r = ref("causal-conv1d")
    for nm,(B,D,L) in {"small":(8,768,512),"medium":(16,2048,2048),"large":(32,4096,4096)}.items():
        x=torch.randn(B,D,L,device="cuda",dtype=torch.float16); w=torch.randn(D,4,device="cuda",dtype=torch.float16); b=torch.randn(D,device="cuda",dtype=torch.float16)
        record("causal_conv1d", f"{nm} {B}x{D}x{L}",
               lambda: mine.causal_conv1d_fn(x,w,b,activation="silu"),
               lambda: r.causal_conv1d_fn(x,w,b,activation="silu"), 2e-2, 2e-2)

def fp8():
    mine = local("finegrained-fp8-helion","finegrained-fp8-helion"); r = ref("finegrained-fp8")
    bn,bk=128,128
    for nm,(M,N,K) in {"small":(512,512,2048),"medium":(2048,2048,4096),"large":(4096,4096,8192)}.items():
        A=(torch.randn(M,K,device="cuda")*0.3).to(torch.float8_e4m3fn); B=(torch.randn(N,K,device="cuda")*0.3).to(torch.float8_e4m3fn)
        As=torch.rand(M,K//bk,device="cuda")*0.5+0.5; Bs=torch.rand(N//bn,K//bk,device="cuda")*0.5+0.5
        record("finegrained_fp8.w8a8_block", f"{nm} {M}x{N}x{K}",
               lambda: mine.w8a8_block_fp8_matmul(A,B,As,Bs,[bn,bk],torch.bfloat16),
               lambda: r.w8a8_block_fp8_matmul(A,B,As,Bs,[bn,bk],torch.bfloat16), 5e-1, 1e-1)

def mamba():
    mine = local("mamba-ssm-helion","mamba-ssm-helion"); r = ref("mamba-ssm")
    for nm,(B,D,L,N) in {"small":(2,256,128,16),"medium":(4,1024,512,16),"large":(8,2048,1024,16)}.items():
        u=torch.randn(B,D,L,device="cuda",dtype=torch.float16); delta=torch.rand(B,D,L,device="cuda",dtype=torch.float16)
        A=-torch.rand(D,N,device="cuda",dtype=torch.float32); Bm=torch.randn(B,N,L,device="cuda",dtype=torch.float16)
        Cm=torch.randn(B,N,L,device="cuda",dtype=torch.float16); Dp=torch.randn(D,device="cuda",dtype=torch.float32); z=torch.randn(B,D,L,device="cuda",dtype=torch.float16)
        def rc():
            o=r.selective_scan_fn(u,delta,A,Bm,Cm,D=Dp,z=z,delta_softplus=True); return o[0] if isinstance(o,tuple) else o
        record("mamba.selective_scan", f"{nm} {B}x{D}x{L}",
               lambda: mine.selective_scan_fn(u,delta,A,Bm,Cm,D=Dp,z=z,delta_softplus=True), rc, 1e-1, 1e-1)

def paged():
    mine = local("paged-attention-helion","paged-attention-helion"); r = ref("paged-attention")
    for nm,(S,H,KVH,D,bs,mb) in {"small":(16,8,8,64,16,16),"medium":(32,16,8,64,16,16)}.items():
        nb=S*mb; x=8
        q=torch.randn(S,H,D,device="cuda",dtype=torch.float16)
        kc=torch.randn(nb,KVH,D//x,bs,x,device="cuda",dtype=torch.float16); vc=torch.randn(nb,KVH,D,bs,device="cuda",dtype=torch.float16)
        sl=torch.randint(1,mb*bs,(S,),device="cuda",dtype=torch.int32); bt=torch.randint(0,nb,(S,mb),device="cuda",dtype=torch.int32)
        scale=1.0/math.sqrt(D); ks=torch.tensor(1.0,device="cuda"); vs=torch.tensor(1.0,device="cuda")
        oh=torch.empty_like(q); orf=torch.empty_like(q)
        record("paged_attention_v1", f"{nm} {S}x{H}x{D}",
               lambda: (mine.paged_attention_v1(oh,q,kc,vc,KVH,scale,bt,sl,bs,mb*bs,None,"auto",ks,vs), oh)[1],
               lambda: (r.paged_attention_v1(orf,q,kc,vc,KVH,scale,bt,sl,bs,mb*bs,None,"auto",ks,vs), orf)[1], 5e-2, 5e-2)

def megablocks():
    mine = local("megablocks-helion","megablocks-helion"); r = ref("megablocks")
    gmm = r.gg_ops.gmm
    for nm,(G,Mavg,K,N) in {"small":(8,256,1024,1024),"medium":(16,512,2048,2048),"large":(32,512,4096,4096)}.items():
        sizes=torch.randint(Mavg//2,Mavg*3//2,(G,)); total=int(sizes.sum())
        a=torch.randn(total,K,device="cuda",dtype=torch.bfloat16); b=torch.randn(G,K,N,device="cuda",dtype=torch.bfloat16)
        bs=sizes.to(torch.int64); c=torch.empty(total,N,device="cuda",dtype=torch.bfloat16)
        record("megablocks.gmm", f"{nm} G{G}x{total}x{K}x{N}",
               lambda: mine.gmm(a,b,c,bs.to("cuda")), lambda: gmm(a,b,bs), 5e-1, 5e-1)


def attention():
    # Helion flash_attn_func vs kernels-community/flash-attn4 (Blackwell CuTeDSL).
    # Both take (B,S,H,D) and flash_attn_func(q,k,v,causal=...). FA4 wants bf16.
    mine = local("attention-helion","attention-helion")
    r = kernels.get_local_kernel(Path(f"{HH_REFS}/flash-attn4/variant"), "cuda")
    for nm,(B,S,H,D) in {"small":(2,512,8,64),"medium":(4,1024,16,64),"large":(8,2048,16,128)}.items():
        q=torch.randn(B,S,H,D,device="cuda",dtype=torch.bfloat16)
        k=torch.randn(B,S,H,D,device="cuda",dtype=torch.bfloat16)
        v=torch.randn(B,S,H,D,device="cuda",dtype=torch.bfloat16)
        def rc():
            o=r.flash_attn_func(q,k,v,causal=False); return o[0] if isinstance(o,tuple) else o
        record("attention.flash_attn_func", f"{nm} {B}x{S}x{H}x{D}",
               lambda: mine.flash_attn_func(q,k,v,causal=False), rc, 5e-2, 5e-2)

def sage():
    # Helion SageAttention2 vs thu-ml/SageAttention built from source for sm_100
    # (B200). Ref = sageattn_qk_int8_pv_fp16_cuda (the sm80 INT8-QK/FP16-PV path
    # that sageattn() dispatches to on Blackwell), pv_accum_dtype="fp32".
    # Both are INT8-quantized attention; head_dim is fixed at 128 (SageAttention2).
    mine = local("sage-attention-helion","sage-attention-helion")
    from sageattention.core import sageattn_qk_int8_pv_fp16_cuda as _sage_ref
    for nm,(B,S,H,D) in {"small":(2,512,8,128),"medium":(4,1024,16,128),"large":(8,2048,16,128)}.items():
        # HND layout (B,H,S,D) for both.
        q=torch.randn(B,H,S,D,device="cuda",dtype=torch.bfloat16)
        k=torch.randn(B,H,S,D,device="cuda",dtype=torch.bfloat16)
        v=torch.randn(B,H,S,D,device="cuda",dtype=torch.bfloat16)
        def rc():
            o=_sage_ref(q,k,v,tensor_layout="HND",is_causal=False,pv_accum_dtype="fp32")
            return o[0] if isinstance(o,tuple) else o
        record("sage_attention.sageattn", f"{nm} {B}x{H}x{S}x{D}",
               lambda: mine.sageattn(q,k,v,tensor_layout="HND",is_causal=False), rc, 5e-2, 5e-2)

def tinygrad_rms():
    # ref: tinygrad_rms_norm(x, eps, out) — NO weight, fp32, hidden locked to 1024.
    mine = local("tinygrad-rms-helion","tinygrad-rms-helion"); r = ref("tinygrad-rms")
    for nm,M in {"small":4096,"medium":16384,"large":65536}.items():
        N=1024
        x=torch.randn(M,N,device="cuda",dtype=torch.float32); w=torch.ones(N,device="cuda",dtype=torch.float32)
        def rc():
            o=r.tinygrad_rms_norm(x,1e-6); return o[0] if isinstance(o,(tuple,list)) else o
        record("tinygrad_rms_norm", f"{nm} {M}x{N}",
               lambda: mine.tinygrad_rms_norm(x,w,1e-6), rc, 2e-2, 2e-2)

def rwkv():
    # ref: forward(w, u, k, v, y) writes into y. w,u:[C]; k,v,y:[B,T,C].
    mine = local("rwkv-helion","rwkv-helion"); r = ref("rwkv")
    for nm,(B,T,C) in {"small":(4,256,512),"medium":(8,1024,1024),"large":(16,1024,2048)}.items():
        # ref rwkv forward requires fp32 tensors.
        w=torch.randn(C,device="cuda",dtype=torch.float32); u=torch.randn(C,device="cuda",dtype=torch.float32)
        k=torch.randn(B,T,C,device="cuda",dtype=torch.float32)*0.5; v=torch.randn(B,T,C,device="cuda",dtype=torch.float32)
        y=torch.empty(B,T,C,device="cuda",dtype=torch.float32)
        def rc():
            r.forward(w,u,k,v,y); return y
        record("rwkv.wkv", f"{nm} {B}x{T}x{C}",
               lambda: mine.wkv(w,u,k,v), rc, 5e-2, 5e-2)

def deformable():
    # ref ms_deform_attn_forward(value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step)
    # single-level (L=1). value:[B,Nkv,H,Dh]; ref sampling_loc:[B,Nq,H,L,P,2]; returns [B,Nq,H*Dh].
    mine = local("deformable-detr-helion","deformable-detr-helion"); r = ref("deformable-detr")
    for nm,(B,H,Dh,P,G,Nq) in {"small":(2,8,32,4,16,900),"medium":(4,8,32,8,32,2000),"large":(8,8,64,8,48,4000)}.items():
        Nkv=G*G
        value=torch.randn(B,Nkv,H,Dh,device="cuda",dtype=torch.float32)
        loc=torch.rand(B,Nq,H,P,2,device="cuda",dtype=torch.float32)
        aw=torch.rand(B,Nq,H,P,device="cuda",dtype=torch.float32)
        spatial=torch.tensor([[G,G]],device="cuda",dtype=torch.int64); lsi=torch.tensor([0],device="cuda",dtype=torch.int64)
        loc5=loc.unsqueeze(3); aw5=aw.unsqueeze(3)  # add L=1 dim for ref
        def rc():
            o=r.ms_deform_attn_forward(value,spatial,lsi,loc5,aw5,64); return o.reshape(B,Nq,H,Dh)
        record("deformable.ms_deform_attn", f"{nm} {B}x{Nq}x{H}x{Dh}",
               lambda: mine.ms_deform_attn(value,loc,aw), rc, 5e-2, 5e-2)

KERNELS = {"activation":activation,"rotary":rotary,"layer_norm":layer_norm,"causal_conv1d":causal_conv1d,
           "fp8":fp8,"mamba":mamba,"paged":paged,"megablocks":megablocks,
           "tinygrad_rms":tinygrad_rms,"rwkv":rwkv,"deformable":deformable,"attention":attention,
           "sage":sage}

if __name__ == "__main__":
    name = sys.argv[1]
    print(f"=== {name} (LLM-guided autotune, bedrock/haiku-4.5) shape={_SHAPE_FILTER or 'all'} ===", flush=True)
    KERNELS[name]()
    outp = Path(f"/tmp/rebench_{name}.json")
    # When running one shape per process, accumulate rows across invocations.
    existing = []
    if _SHAPE_FILTER is not None and outp.exists():
        try:
            existing = [r for r in json.loads(outp.read_text())
                        if not any(r["size"] == n["size"] for n in RESULTS)]
        except Exception:
            existing = []
    outp.write_text(json.dumps(existing + RESULTS, indent=2))
    print(f"WROTE {outp} (+{len(RESULTS)} rows, {len(existing)+len(RESULTS)} total)", flush=True)
