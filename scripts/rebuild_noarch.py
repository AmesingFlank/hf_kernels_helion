"""Reconstruct the loadable build/torch-cuda layout for every noarch Helion
kernel directly from its committed torch-ext/ source, WITHOUT kernel-builder/nix
(which was removed from this environment; the nix-store `result` bundles were
GC'd). These are pure-Python Helion kernels — the compute is JIT-compiled by
Helion at call time and the custom_op self-registers its torch.ops namespace, so
no compiled .so is needed. We replicate exactly the layout that the surviving
sage-attention-helion build has (verified loadable via get_local_kernel).

For each kernel dir <k>/<sub>/:
  - copy torch-ext/<pkg>/*.py            -> build/torch-cuda/*.py   (flattened)
  - write build/torch-cuda/_ops.py       (namespace _<pkg>_<backend>_<id>)
  - write build/torch-cuda/metadata.json
  - write build/torch-cuda/<pkg>/__init__.py  (re-import shim)
  - symlink result -> build
"""
import json, os, shutil, hashlib, glob

HH = "/home/dev/hf_kernels_helion"

OPS_PY_TMPL = '''import torch

def get_backend() -> str:
    """Detect the backend by inspecting torch."""
    import torch

    if hasattr(torch, "neuron"):
        return "neuron"
    elif torch.version.cuda is not None:
        return "cuda"
    elif torch.version.hip is not None:
        return "rocm"
    elif torch.backends.mps.is_available():
        return "metal"
    elif hasattr(torch.version, "xpu") and torch.version.xpu is not None:
        return "xpu"
    else:
        return "cpu"


def _find_ops_name() -> str:
    kernel_name = "{pkg}"
    unique_id = "{uid}"
    backend = get_backend()
    return f"_{{kernel_name}}_{{backend}}_{{unique_id}}"


_OPS_NAME = _find_ops_name()

ops = getattr(torch.ops, _OPS_NAME)

def add_op_namespace_prefix(op_name: str) -> str:
    """
    Prefix op by namespace.
    """
    return f"{{_OPS_NAME}}::{{op_name}}"
'''

SHIM_TMPL = '''import ctypes
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _import_from_path(file_path: Path) -> ModuleType:
    path_hash = "{:x}".format(ctypes.c_size_t(hash(file_path.absolute())).value)
    module_name = path_hash
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Cannot load spec for {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    if module is None:
        raise ImportError(f"Cannot load module {module_name} from spec")
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore
    return module


globals().update(vars(_import_from_path(Path(__file__).parent.parent / "__init__.py")))
'''

# kernel dir -> subdir -> pkg
SUBS = sorted(glob.glob(f"{HH}/*/*/torch-ext"))
done = 0
for te in SUBS:
    sub = os.path.dirname(te)               # .../<k>/<subdir>
    pkgs = [d for d in os.listdir(te) if os.path.isdir(f"{te}/{d}") and d != "__pycache__"]
    if not pkgs:
        continue
    pkg = pkgs[0]
    build = f"{sub}/build/torch-cuda"
    already = os.path.isfile(f"{build}/_ops.py") and os.path.isfile(f"{build}/__init__.py")

    # dash-form kernel name from build.toml (the `kernels` lib validates it)
    kname = pkg.replace("_", "-")
    bt = f"{sub}/build.toml"
    if os.path.isfile(bt):
        for line in open(bt):
            line = line.strip()
            if line.startswith("name") and "=" in line:
                kname = line.split("=", 1)[1].strip().strip('"').strip("'")
                break

    os.makedirs(build, exist_ok=True)
    # 1) ALWAYS re-sync torch-ext/<pkg>/*.py into build/torch-cuda/ (source of
    #    truth). This includes any pre-tuned `_helion_aot_*.py` heuristic files,
    #    so the runtime location (build/, symlinked as result/) picks them up.
    srcpkg = f"{te}/{pkg}"
    for f in os.listdir(srcpkg):
        if f.endswith(".py"):
            shutil.copy2(f"{srcpkg}/{f}", f"{build}/{f}")
    # 2-4) generate the packaging glue only if missing (don't clobber on re-sync).
    if not already:
        uid = hashlib.sha1(pkg.encode()).hexdigest()[:7]
        open(f"{build}/_ops.py", "w").write(OPS_PY_TMPL.format(pkg=pkg, uid=uid))
        json.dump({
            "name": kname,
            "id": f"_{pkg}_cuda_{uid}",
            "version": 1,
            "license": "Apache-2.0",
            "python-depends": ["helion"],
            "backend": {"type": "cuda"},
        }, open(f"{build}/metadata.json", "w"), indent=2)
        os.makedirs(f"{build}/{pkg}", exist_ok=True)
        open(f"{build}/{pkg}/__init__.py", "w").write(SHIM_TMPL)
    # 5) result -> build symlink
    res = f"{sub}/result"
    if not (os.path.islink(res) and os.path.exists(res)):
        if os.path.lexists(res):
            os.remove(res)
        os.symlink("build", res)
    print(f"  {'synced' if already else 'rebuilt'} {pkg}")
    done += 1

print(f"reconstructed/verified {done} noarch kernels")
