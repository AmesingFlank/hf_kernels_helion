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
        {'block_sizes': [2, 32], 'loop_orders': [[0, 1]], 'l2_groupings': [1], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['', '', '', ''], 'num_warps': 2, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 4, 'maxnreg': 128},
    ]
    return _C[key__rwkv_wkv(*args)]
