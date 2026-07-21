"""
Auto-generated heuristic for kernel: sage_attn_fwd
Backend: decision_tree

Provides:
- key_sage_attn_fwd(*args): Returns config index (cache key)
- autotune_sage_attn_fwd(*args): Returns config dict for the given arguments
"""

import torch


def key_sage_attn_fwd(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    # No features needed
    return 0


def autotune_sage_attn_fwd(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [256, 128], 'range_unroll_factors': [0, 0, 0], 'range_warp_specializes': [None, None, False], 'range_num_stages': [0, 2, 0], 'range_multi_buffers': [None, None, False], 'range_flattens': [None, None, False], 'load_eviction_policies': ['', '', '', 'last'], 'num_warps': 8, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 1},
    ]
    return _C[key_sage_attn_fwd(*args)]
