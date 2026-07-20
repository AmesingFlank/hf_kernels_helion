"""Convert a kernel file's @helion.kernel decorators to
@helion.experimental.aot_kernel, and ensure `import helion.experimental` is
present. Idempotent. Handles both `@helion.kernel(` and `@helion.kernel()` and
multi-line decorator forms (only the decorator token is replaced)."""
import sys, re

path = sys.argv[1]
s = open(path).read()

# 1) ensure the experimental import exists (right after `import helion`)
if "import helion.experimental" not in s:
    s = re.sub(r"(^import helion\n)", r"\1import helion.experimental\n", s, count=1, flags=re.M)

# 2) replace the decorator token (works for both () and (... multi-line)
n = s.count("@helion.kernel")
s = s.replace("@helion.kernel(", "@helion.experimental.aot_kernel(")

open(path, "w").write(s)
print(f"{path}: converted {n} decorators; import ok={'import helion.experimental' in s}")
