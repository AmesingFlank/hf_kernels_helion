"""
Auto-generated heuristic for kernel: _paged_attention_v1
Backend: decision_tree

Provides:
- key__paged_attention_v1(*args): Returns config index (cache key)
- autotune__paged_attention_v1(*args): Returns config dict for the given arguments
"""

import torch


def key__paged_attention_v1(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    # No features needed
    return 0


def autotune__paged_attention_v1(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [1, 1], 'loop_orders': [[0, 1]], 'l2_groupings': [1], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['', '', '', '', ''], 'num_warps': 1, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key__paged_attention_v1(*args)]
