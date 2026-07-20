"""
Auto-generated heuristic for kernel: _selective_scan
Backend: decision_tree

Provides:
- key__selective_scan(*args): Returns config index (cache key)
- autotune__selective_scan(*args): Returns config dict for the given arguments
"""

import torch


def key__selective_scan(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    # No features needed
    return 0


def autotune__selective_scan(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [1, 4], 'loop_orders': [[0, 1]], 'l2_groupings': [1], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [None, None], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['', '', '', '', '', '', ''], 'num_warps': 1, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key__selective_scan(*args)]
