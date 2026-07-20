"""Quick check that a reconstructed noarch kernel loads and runs end-to-end
(op registration + Helion JIT), using the LLM autotuner env for speed."""
import os
os.environ.setdefault("HELION_AUTOTUNER", "LLMGuidedSearch")
os.environ.setdefault("HELION_LLM_PROVIDER", "bedrock")
os.environ.setdefault("HELION_LLM_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
os.environ.setdefault("AWS_REGION", "us-east-2")
os.environ.setdefault("HELION_AUTOTUNE_BENCHMARK_SUBPROCESS", "0")
os.environ.setdefault("HELION_AUTOTUNE_IGNORE_ERRORS", "1")
import sys
sys.path.insert(0, "/home/dev")
from pathlib import Path
import torch, kernels

m = kernels.get_local_kernel(
    Path("/home/dev/hf_kernels_helion/activation-helion/activation-helion/result"), "cuda")
print("loaded, has silu_and_mul:", hasattr(m, "silu_and_mul"), flush=True)
B, S, D2 = 8, 1024, 2048
d = D2 // 2
x = torch.randn(B, S, D2, device="cuda", dtype=torch.float16)
out = torch.empty(B, S, d, device="cuda", dtype=torch.float16)
m.silu_and_mul(out, x)
torch.cuda.synchronize()
ref = torch.nn.functional.silu(x[..., :d]) * x[..., d:]
print("RUNS; matches:", torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2), flush=True)
