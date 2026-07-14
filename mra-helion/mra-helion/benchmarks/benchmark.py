import torch

from kernels.benchmark import Benchmark


class MraHelionBenchmark(Benchmark):
    def setup(self):
        self.size = 1024
        self.input = torch.randn(self.size, self.size, device=self.device)
        self.out = torch.empty_like(self.input)

    def benchmark_base(self):
        self.kernel.mra_helion(self.out, self.input)

    def verify_base(self) -> torch.Tensor:
        # Reference implementation
        return self.input.clone()