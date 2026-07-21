"""
Auto-generated heuristic for kernel: attention_output
Backend: decision_tree

Provides:
- key_attention_output(*args): Returns config index (cache key)
- autotune_attention_output(*args): Returns config dict for the given arguments
"""

import torch


def key_attention_output(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    # No features needed
    return 0


def autotune_attention_output(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [1, 128, 64], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [None, False], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, None], 'range_flattens': [None, False], 'load_eviction_policies': ['', 'last', ''], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_attention_output(*args)]
