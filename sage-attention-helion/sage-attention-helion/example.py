# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "kernels",
#     "numpy",
#     "torch",
# ]
# ///

import platform
from pathlib import Path

import kernels
import torch
import torch.nn.functional as F


def main() -> None:
    # Load the locally built kernel. The second arg is the *backend* ("cuda").
    repo = Path(__file__).parent
    build_dir = repo / "result" if (repo / "result").exists() else repo / "build"
    kernel = kernels.get_local_kernel(build_dir, "cuda")

    if torch.version.cuda is not None and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # SageAttention2 requires head_dim == 128. Inputs are (B, H, S, D) for the
    # default "HND" layout.
    B, H, S, D = 2, 8, 512, 128
    q = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)

    # Run the INT8-quantized SageAttention2 kernel.
    result = kernel.sageattn(q, k, v, tensor_layout="HND", is_causal=False)
    print(f"Output shape: {tuple(result.shape)}")

    # SageAttention is an INT8 approximation of full-precision attention, so we
    # verify against SDPA with a tolerance matching the INT8 quant error floor.
    expected = F.scaled_dot_product_attention(q, k, v)
    rel = (result.float() - expected.float()).norm() / expected.float().norm()
    print(f"Relative error vs SDPA: {rel.item():.4f}")
    assert rel < 0.05, "Kernel output too far from SDPA!"
    print("Success!")


# The `if __name__ == "__main__"` guard is REQUIRED (not stylistic): Helion
# autotunes in a spawned subprocess that re-imports this module; an unguarded
# top-level kernel call would recursively spawn autotuners.
if __name__ == "__main__":
    main()
