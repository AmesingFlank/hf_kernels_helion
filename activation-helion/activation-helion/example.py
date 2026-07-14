from pathlib import Path
import torch
import torch.nn.functional as F
import kernels


def main():
    repo = Path(__file__).parent
    build_dir = repo / "result" if (repo / "result").exists() else repo / "build"
    k = kernels.get_local_kernel(build_dir, "cuda")
    x = torch.randn(4, 512, 2048, device="cuda", dtype=torch.float16)
    out = torch.empty(4, 512, 1024, device="cuda", dtype=torch.float16)
    k.silu_and_mul(out, x)
    d = 1024
    ref = F.silu(x[..., :d]) * x[..., d:]
    assert torch.allclose(out, ref, atol=1e-2, rtol=1e-2)
    print("Success!")


if __name__ == "__main__":
    main()
