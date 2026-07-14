import torch

from kernels.benchmark import Benchmark


def _extract_output(result):
    if isinstance(result, tuple):
        return result[0]
    return result


def _reference_attention(query, key, value, causal=False):
    """Reference SDPA on (B, S, H, D) inputs (matches kernels' util)."""
    query, key, value = (x.transpose(1, 2).contiguous() for x in (query, key, value))
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=causal
        )
    return out.transpose(1, 2).contiguous()


class AttentionHelionBenchmark(Benchmark):
    """Mirrors kernels.benchmarks.attention.FlashAttentionBenchmark.

    Uses the (B, S, H, D) flash-attn layout and calls ``flash_attn_func`` so
    the same workloads apply to this Helion kernel and to flash-attn3.
    """

    seed: int = 42

    def setup_small(self):
        B, S, H, D = 2, 128, 8, 64
        self.q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        self.k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        self.v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        self.out = torch.empty(B, S, H, D, device="cuda", dtype=torch.float16)

    def benchmark_small(self):
        self.out = _extract_output(
            self.kernel.flash_attn_func(self.q, self.k, self.v, causal=False)
        )

    def verify_small(self) -> torch.Tensor:
        return _reference_attention(self.q, self.k, self.v, causal=False)

    def setup_medium(self):
        B, S, H, D = 4, 512, 16, 64
        self.q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        self.k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        self.v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        self.out = torch.empty(B, S, H, D, device="cuda", dtype=torch.float16)

    def benchmark_medium(self):
        self.out = _extract_output(
            self.kernel.flash_attn_func(self.q, self.k, self.v, causal=False)
        )

    def verify_medium(self) -> torch.Tensor:
        return _reference_attention(self.q, self.k, self.v, causal=False)

    def setup_large(self):
        B, S, H, D = 8, 1024, 32, 128
        self.q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        self.k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        self.v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float16)
        self.out = torch.empty(B, S, H, D, device="cuda", dtype=torch.float16)

    def benchmark_large(self):
        self.out = _extract_output(
            self.kernel.flash_attn_func(self.q, self.k, self.v, causal=False)
        )

    def verify_large(self) -> torch.Tensor:
        return _reference_attention(self.q, self.k, self.v, causal=False)
