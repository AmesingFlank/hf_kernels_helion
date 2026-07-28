#!/usr/bin/env python
"""Upload a Helion kernel to the HF Hub as a `kernel` repo, matching the
published kernels-community layout (build/ + README + benchmarks — NOT the
kernel-builder source dirs like torch-ext/, flake.*, build.toml).

    python scripts/upload_kernel.py <kernel-dir> <repo-id> [--dry-run]

e.g. python scripts/upload_kernel.py attention-helion/attention-helion HelionDSL/attention

Ships build/torch-cuda/ (the loadable, pre-tuned artifact) at the repo root,
plus README.md, benchmarks/, example.py, tests/. Excludes source/build-tooling
files that real Hub kernels don't carry, and deletes any such stale files a
prior upload may have left in the repo. Also creates a `v1` branch so
`get_kernel(..., version=1)` resolves.

NOTE: this huggingface_hub version's write validators reject repo_type="kernel"
even though the server supports it, so we extend constants.REPO_TYPES in-process.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import huggingface_hub.constants as _C

if "kernel" not in _C.REPO_TYPES:
    _C.REPO_TYPES = list(_C.REPO_TYPES) + ["kernel"]

from huggingface_hub import HfApi, upload_folder  # noqa: E402

# What a published HF kernel repo should contain (mirrors flash-attn3/activation).
# Everything else in the kernel dir is source/build-tooling and is NOT uploaded.
ALLOW = [
    "build/**",          # the loadable, pre-tuned artifact (build/torch-cuda/...)
    "benchmarks/**",
    "tests/**",
    "README.md",
    "example.py",
    ".gitattributes",
]
# Stale files a prior (over-broad) upload may have left in the repo — remove them
# so the published repo matches the clean layout.
DELETE = [
    "torch-ext/**",
    "flake.lock", "flake.nix", "build.toml",
    "CARD.md", ".gitignore",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel_dir", help="path to <name>-helion/<name>-helion")
    ap.add_argument("repo_id", help="e.g. HelionDSL/attention")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what WOULD be uploaded/deleted; do not touch the Hub")
    args = ap.parse_args()

    kdir = Path(args.kernel_dir).resolve()
    if not (kdir / "build" / "torch-cuda").is_dir():
        sys.exit(f"ERROR: {kdir}/build/torch-cuda not found — run scripts/rebuild_noarch.py first")

    # Compute the concrete file set for visibility (glob the ALLOW patterns).
    import fnmatch
    all_files = [p for p in kdir.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    rels = [str(p.relative_to(kdir)) for p in all_files]
    def matches(rel, pats):
        return any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(rel, pat.replace("/**", "/*"))
                   or rel.startswith(pat.replace("/**", "/")) for pat in pats)
    to_upload = sorted(r for r in rels if matches(r, ALLOW))

    print(f"=== {args.repo_id}  (from {kdir}) ===")
    print(f"WOULD UPLOAD ({len(to_upload)} files):")
    for r in to_upload:
        print(f"  + {r}")
    print(f"WOULD DELETE stale (if present): {', '.join(DELETE)}")

    if args.dry_run:
        print("\n[dry-run] no Hub changes made.")
        return

    api = HfApi()
    url = upload_folder(
        folder_path=str(kdir),
        repo_id=args.repo_id,
        repo_type="kernel",
        commit_message="Publish kernel (build/ + card + benchmarks); drop source/build-tooling files",
        allow_patterns=ALLOW,
        delete_patterns=DELETE,
    )
    print("UPLOAD OK:", url)
    api.create_branch(args.repo_id, branch="v1", repo_type="kernel",
                      revision="main", exist_ok=True)
    print("v1 branch ensured (for version=1)")


if __name__ == "__main__":
    main()
