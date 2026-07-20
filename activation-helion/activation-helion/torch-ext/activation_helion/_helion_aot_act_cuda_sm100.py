"""
Auto-generated heuristic for kernel: _silu_and_mul
Backend: decision_tree

Provides:
- key__silu_and_mul(*args): Returns config index (cache key)
- autotune__silu_and_mul(*args): Returns config dict for the given arguments
"""

import torch


def key__silu_and_mul(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    # No features needed
    return 0


def autotune__silu_and_mul(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [32, 32], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0], 'range_warp_specializes': [None], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['first', ''], 'num_warps': 2, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key__silu_and_mul(*args)]
