# /// script
# requires-python = ">=3.13"
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

# Load the locally built kernel
kernel = kernels.get_local_kernel(Path("build"), "relu_helion")

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

# Create input tensor
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device=device)
print(f"Input:  {x}")

# Run kernel (element-wise ReLU)
result = kernel.relu(x)
print(f"Output: {result}")

# Verify result
expected = torch.relu(x)
assert torch.allclose(result, expected), "Kernel output doesn't match expected!"
print("Success!")