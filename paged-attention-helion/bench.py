from __future__ import annotations
import sys, math
from pathlib import Path
sys.path.insert(0, "/home/dev/hf_kernels_helion")
import torch
import kernels
from bench_common import report

REPO = Path("/home/dev/hf_kernels_helion/paged-attention-helion/paged-attention-helion/result")


def ref_paged(query, key_cache, value_cache, block_tables, seq_lens, scale, block_size, num_kv_heads):
    num_seqs, num_heads, head_size = query.shape
    q_per_kv = num_heads // num_kv_heads
    out = torch.zeros_like(query)
    for s in range(num_seqs):
        L = int(seq_lens[s])
        idx = torch.arange(L, device=query.device)
        blk = block_tables[s, idx // block_size]
        off = idx % block_size
        for h in range(num_heads):
            kv = h // q_per_kv
            K = key_cache[blk, kv, :, off]      # [L, head]
            V = value_cache[blk, kv, :, off]
            sc = (query[s, h].float() @ K.float().t()) * scale
            p = torch.softmax(sc, -1)
            out[s, h] = (p @ V.float()).to(out.dtype)
    return out


def main():
    k = kernels.get_local_kernel(REPO, "cuda")
    print("=== paged-attention-helion v1 (decode) vs torch paged reference ===")
    # (num_seqs, num_heads, num_kv_heads, head_size, block_size, max_blocks)
    cfgs = {
        "small": (16, 8, 8, 64, 16, 16),
        "medium": (64, 32, 8, 128, 16, 32),
        "large": (128, 32, 8, 128, 16, 64),
    }
    for name, (S, H, KVH, D, bs, mb) in cfgs.items():
        nb = S * mb
        q = torch.randn(S, H, D, device="cuda", dtype=torch.float16)
        kc = torch.randn(nb, KVH, D, bs, device="cuda", dtype=torch.float16)
        vc = torch.randn(nb, KVH, D, bs, device="cuda", dtype=torch.float16)
        seq_lens = torch.randint(1, mb * bs, (S,), device="cuda", dtype=torch.int32)
        block_tables = torch.randint(0, nb, (S, mb), device="cuda", dtype=torch.int32)
        scale = 1.0 / math.sqrt(D)
        out = torch.empty_like(q)
        # The torch reference is a per-seq/head Python oracle (very slow) used only
        # for correctness; time it with few iters so it doesn't dominate wall clock.
        report(
            f"paged_attn_v1[{name}]", (S, H, D),
            lambda: k.paged_attention_v1(out, q, kc, vc, KVH, scale, block_tables, seq_lens, bs, mb * bs),
            lambda: ref_paged(q, kc, vc, block_tables, seq_lens, scale, bs, KVH),
            atol=2e-2, rtol=2e-2, base_iters=3, base_warmup=1,
        )


if __name__ == "__main__":
    main()
