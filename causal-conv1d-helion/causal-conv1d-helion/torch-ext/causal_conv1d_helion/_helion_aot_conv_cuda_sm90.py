"""
Auto-generated heuristic for kernel: _causal_conv1d
Backend: decision_tree

Provides:
- key__causal_conv1d(*args): Returns config index (cache key)
- autotune__causal_conv1d(*args): Returns config dict for the given arguments
"""

import torch


def key__causal_conv1d(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    # No features needed
    return 0


def autotune__causal_conv1d(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [1, 32, 128], 'loop_orders': [[0, 1, 2]], 'l2_groupings': [16], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'static_ranges': [False], 'load_eviction_policies': ['', '', ''], 'num_warps': 4, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key__causal_conv1d(*args)]
