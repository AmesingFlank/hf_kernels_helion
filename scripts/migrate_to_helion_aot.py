"""Migrate HF kernel sources from the deprecated
`@helion.experimental.aot_kernel` to the promoted `@helion.aot_kernel`
(helion PR #3108). Idempotent. Run from repo root:

    python scripts/migrate_to_helion_aot.py
"""
import glob
import re

HH = "/home/dev/hf_kernels_helion"

files = glob.glob(f"{HH}/*/*/torch-ext/*/*.py")
changed = 0
for p in files:
    s = open(p).read()
    if "helion.experimental" not in s:
        continue
    orig = s
    has_plain = re.search(r"^import helion$", s, re.M) is not None
    if has_plain:
        # drop the now-redundant experimental import line
        s = re.sub(r"^import helion\.experimental\n", "", s, flags=re.M)
    else:
        # convert `import helion.experimental` -> `import helion`
        s = re.sub(r"^import helion\.experimental$", "import helion", s, flags=re.M)
    # decorator + any stray refs
    s = s.replace("@helion.experimental.aot_kernel", "@helion.aot_kernel")
    s = s.replace("helion.experimental.aot_kernel", "helion.aot_kernel")
    s = s.replace("helion.experimental.aot_runner", "helion.autotuner.aot_runner")
    if s != orig:
        open(p, "w").write(s)
        changed += 1
        print(f"  migrated {p.split('/torch-ext/')[1]}")
print(f"migrated {changed} files to helion.aot_kernel")
