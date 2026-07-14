from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, "/home/dev/hf_kernels_helion")
import torch
import kernels
from bench_common import report

REPO = Path("/home/dev/hf_kernels_helion/deformable-detr-helion/deformable-detr-helion/result")


def ref(value, loc, aw, Hg, Wg):
    B, Nkv, H, Dh = value.shape
    Nq = loc.shape[1]; P = loc.shape[3]
    vg = value.permute(0, 2, 3, 1).reshape(B * H, Dh, Hg, Wg)
    grid = loc.permute(0, 2, 1, 3, 4).reshape(B * H, Nq, P, 2) * 2 - 1
    samp = torch.nn.functional.grid_sample(vg.float(), grid.float(), mode="bilinear",
                                           padding_mode="zeros", align_corners=False)
    samp = samp.reshape(B, H, Dh, Nq, P)
    a = aw.permute(0, 2, 1, 3)
    return (samp * a[:, :, None].float()).sum(-1).permute(0, 3, 1, 2)


def main():
    k = kernels.get_local_kernel(REPO, "cuda")
    print("=== deformable-detr-helion ms_deform_attn vs torch grid_sample ===")
    cfgs = {"small": (2, 8, 32, 4, 16, 900), "medium": (4, 8, 32, 8, 32, 2000), "large": (8, 8, 64, 8, 48, 4000)}
    for name, (B, H, Dh, P, G, Nq) in cfgs.items():
        Nkv = G * G
        value = torch.randn(B, Nkv, H, Dh, device="cuda", dtype=torch.float16)
        loc = torch.rand(B, Nq, H, P, 2, device="cuda", dtype=torch.float16)
        aw = torch.rand(B, Nq, H, P, device="cuda", dtype=torch.float16)
        report(
            f"ms_deform_attn[{name}]", (B, Nq, H, Dh),
            lambda: k.ms_deform_attn(value, loc, aw),
            lambda: ref(value, loc, aw, G, G),
            atol=5e-2, rtol=5e-2,
        )


if __name__ == "__main__":
    main()
