# Re-pre-tuning the attention kernel on an H100 (or any other GPU)

The pre-tuned configs are **per-GPU**: each is written as
`_helion_aot_attention_cuda_<compute>.py` next to the kernel source, keyed on the
GPU's compute capability (B200 = `sm100`, H100 = `sm90`). Running the tuner on an
H100 produces `_helion_aot_attention_cuda_sm90.py`; it coexists with the sm100
file, and `get_kernel` picks the right one for the running GPU (with fallback to
older compatible archs).

## Prereqs on the H100 box

```bash
git clone <this repo>                 # or pull latest
cd hf_kernels_helion
python -m venv .venv && source .venv/bin/activate   # (or reuse your venv)
# install: a CUDA torch, triton, helion, and the kernels lib
pip install -U torch triton kernels
pip install git+https://github.com/pytorch/helion    # or your helion checkout
```

The tuner auto-detects the GPU — no flags to change per machine.

## Re-tune the attention kernel (full LFBO, the default autotuner)

```bash
# tune only attention, full effort (7 bhsd shapes: head_dim 64/128, batch + seq sweeps)
python scripts/aot_tune.py --effort full attention
```

`aot_tune.py` sets no `HELION_AUTOTUNER`, so it uses Helion's **default LFBO
tree-search** autotuner (this is what "full LFBO" means). `--effort full` is the
most thorough search. Progress prints per shape (done / remaining / ETA).

When it finishes it writes:

```
attention-helion/attention-helion/torch-ext/attention_helion/_helion_aot_attention_cuda_sm90.py
attention-helion/attention-helion/build/torch-cuda/_helion_aot_attention_cuda_sm90.py
```

## Verify it beats / matches SDPA on the H100

```bash
# the external analysis' own benchmark (device-time, CUDA-graph replay):
python /tmp/helionbench/bench_one.py bf16 8 32 1024 128 /tmp/r.json bhsd
# or sweep the full shape set:
python /tmp/helionbench/run_bench.py bhsd
```

(Point `bench_one.py` at the local build via a `KERNELS_LOCAL_OVERRIDE`, or upload
first — see below.)

## Publish the H100 config

```bash
# from the H100 box, add just the new sm90 heuristic to the Hub repo:
python scripts/upload_kernel.py attention-helion/attention-helion HelionDSL/attention
# (uploads build/ incl. both _helion_aot_attention_cuda_sm90.py and _sm100.py)
```

## Notes

- To tune **all** kernels on the H100 (not just attention), drop the kernel name:
  `python scripts/aot_tune.py --effort full`.
- The autotuner is chosen by env, not the script — to instead use the LLM tuner,
  export `HELION_AUTOTUNER=LLMGuidedSearch` (+ Bedrock/LLM vars) before running,
  or use `scripts/pretune_all_bedrock_llm.sh`.
- Full-effort LFBO is slow (~10-15 min/shape on the big attention shapes); the
  7-shape attention sweep is ~1.5-2 h. Use `--effort quick` for a fast pass.
