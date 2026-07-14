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
    # Load the locally built kernel. The second arg is the *backend* ("cuda"),
    # not the kernel name. `result` is the build-output symlink from
    # `kernel-builder build`; fall back to `build` if present.
    repo = Path(__file__).parent
    build_dir = repo / "result" if (repo / "result").exists() else repo / "build"
    kernel = kernels.get_local_kernel(build_dir, "cuda")

    # Select device
    if platform.system() == "Darwin":
        device = torch.device("mps")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.version.cuda is not None and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Create (B, H, S, D) inputs for scaled dot-product attention
    B, H, S, D = 2, 8, 512, 64
    q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

    # Run kernel (scaled dot-product attention)
    result = kernel.attention(q, k, v)
    print(f"Output shape: {tuple(result.shape)}")

    # Verify result against PyTorch SDPA
    expected = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(result, expected, atol=5e-2, rtol=2e-2), (
        "Kernel output doesn't match SDPA!"
    )
    print("Success!")


# NOTE: the `if __name__ == "__main__"` guard is REQUIRED, not stylistic.
# Helion autotunes in a *spawned* subprocess that re-imports this module; an
# unguarded top-level kernel call would be re-executed on every worker import,
# recursively spawning autotuners until it aborts with NoConfigFound. See
# ISSUES.md ("Autotuner spawn / __main__ guard").
if __name__ == "__main__":
    main()
