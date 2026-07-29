"""
Auto-generated heuristic for kernel: _apply_rotary
Backend: decision_tree

Provides:
- key__apply_rotary(*args): Returns config index (cache key)
- autotune__apply_rotary(*args): Returns config dict for the given arguments
"""

import torch


def key__apply_rotary(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    # No features needed
    return 0


def autotune__apply_rotary(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [1, 2, 32], 'loop_orders': [[0, 1, 2]], 'l2_groupings': [4], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', '', '', '', '', '', ''], 'num_warps': 4, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key__apply_rotary(*args)]
