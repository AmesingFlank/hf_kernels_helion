"""
Auto-generated heuristic for kernel: _rwkv_wkv
Backend: decision_tree

Provides:
- key__rwkv_wkv(*args): Returns config index (cache key)
- autotune__rwkv_wkv(*args): Returns config dict for the given arguments
"""

import torch


def key__rwkv_wkv(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    # No features needed
    return 0


def autotune__rwkv_wkv(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [4, 64], 'loop_orders': [[0, 1]], 'l2_groupings': [32], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [None, True], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, False], 'range_flattens': [None, False], 'load_eviction_policies': ['', '', '', 'first'], 'num_warps': 4, 'num_stages': 7, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'xyz'},
    ]
    return _C[key__rwkv_wkv(*args)]
