import platform

import torch

import finegrained_fp8_helion


def test_finegrained_fp8_helion():
    if platform.system() == "Darwin":
        device = torch.device("mps")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.version.cuda is not None and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    x = torch.randn(1024, 1024, dtype=torch.float32, device=device)
    expected = x + 1.0
    result = finegrained_fp8_helion.finegrained_fp8_helion(x)
    torch.testing.assert_close(result, expected)