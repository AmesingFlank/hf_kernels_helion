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
        {'block_sizes': [256, 64], 'range_unroll_factors': [0, 0, 0], 'range_warp_specializes': [None, None, None], 'range_num_stages': [0, 0, 0], 'range_multi_buffers': [None, None, None], 'range_flattens': [None, None, None], 'load_eviction_policies': ['', '', '', ''], 'num_warps': 8, 'num_stages': 4, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_sage_attn_fwd(*args)]
