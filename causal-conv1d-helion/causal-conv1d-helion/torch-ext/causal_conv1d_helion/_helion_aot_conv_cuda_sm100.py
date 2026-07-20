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
        {'block_sizes': [1, 16, 16], 'loop_orders': [[0, 2, 1]], 'l2_groupings': [32], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [None, None], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, None], 'range_flattens': [None, True], 'static_ranges': [False], 'load_eviction_policies': ['first', 'first', 'last'], 'num_warps': 1, 'num_stages': 1, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key__causal_conv1d(*args)]
