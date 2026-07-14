import pytest
import torch
import torch.nn.functional as F

import activation_helion


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.kernels_ci
def test_silu_and_mul(device):
    x = torch.randn(8, 256, 2048, device=device, dtype=torch.float16)
    out = torch.empty(8, 256, 1024, device=device, dtype=torch.float16)
    activation_helion.silu_and_mul(out, x)
    d = 1024
    ref = F.silu(x[..., :d]) * x[..., d:]
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


@pytest.mark.kernels_ci
def test_gelu_and_mul(device):
    x = torch.randn(8, 256, 2048, device=device, dtype=torch.float16)
    out = torch.empty(8, 256, 1024, device=device, dtype=torch.float16)
    activation_helion.gelu_and_mul(out, x)
    d = 1024
    ref = F.gelu(x[..., :d]) * x[..., d:]
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
