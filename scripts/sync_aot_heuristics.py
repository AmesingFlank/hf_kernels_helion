"""Copy generated `_helion_aot_*.py` heuristic files from each kernel's
build/torch-cuda/ (where the AOT build phase writes them, since that's the
runtime co_filename) back into torch-ext/<pkg>/ as the committed source of
truth. rebuild_noarch.py then re-propagates them to build/ on any rebuild."""
import glob, os, shutil

HH = "/home/dev/hf_kernels_helion"
n = 0
for heur in glob.glob(f"{HH}/*/*/build/torch-cuda/_helion_aot_*.py"):
    build_dir = os.path.dirname(heur)          # .../build/torch-cuda
    sub = os.path.dirname(os.path.dirname(build_dir))  # .../<repo>/<subdir>
    te = f"{sub}/torch-ext"
    pkgs = [d for d in os.listdir(te) if os.path.isdir(f"{te}/{d}") and d != "__pycache__"]
    if not pkgs:
        continue
    dst = f"{te}/{pkgs[0]}/{os.path.basename(heur)}"
    shutil.copy2(heur, dst)
    print(f"  synced {os.path.basename(heur)} -> {pkgs[0]}/")
    n += 1
print(f"synced {n} heuristic files to torch-ext source")
