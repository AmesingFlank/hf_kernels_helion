# Env for AOT pre-tuning with Helion's DEFAULT autotuner (LFBOTreeSearch) at
# FULL effort — NOT the LLM-guided autotuner. We deliberately do NOT set
# HELION_AUTOTUNER (so it defaults to LFBOTreeSearch) and pin full effort.
export HELION_AUTOTUNE_EFFORT=full
# Same operational gotchas as before: in-process benchmarking avoids the
# spawn-reimport recursion for get_local_kernel modules; ignore a non-compiling
# candidate config rather than aborting the whole search.
export HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=0
export HELION_AUTOTUNE_IGNORE_ERRORS=1
# Make sure no stale LLM autotuner selection leaks in from the shell.
unset HELION_AUTOTUNER
unset HELION_LLM_PROVIDER
unset HELION_LLM_MODEL
