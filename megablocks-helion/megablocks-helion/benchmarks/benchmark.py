import torch

from kernels.benchmark import Benchmark


class MegablocksHelionBenchmark(Benchmark):
    def setup(self):
        self.size = 1024
        self.input = torch.randn(self.size, self.size, device=self.device)
        self.out = torch.empty_like(self.input)

    def benchmark_base(self):
        self.kernel.megablocks_helion(self.out, self.input)

    def verify_base(self) -> torch.Tensor:
        # Reference implementation
        return self.input.clone()