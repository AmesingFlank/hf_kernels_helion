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
    _arg0_dim0 = int(args[0].shape[0]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 0 else 0
    if _arg0_dim0 <= 4.0:
        return 1
    else:
        return 0


def autotune_attention_output(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [1, 128, 128], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [1, 2], 'range_warp_specializes': [], 'range_num_stages': [4, 1], 'range_multi_buffers': [True, False], 'range_flattens': [False, None], 'load_eviction_policies': ['first', 'last', ''], 'num_warps': 8, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 1},
        {'block_sizes': [1, 64, 64], 'loop_orders': [[0, 1]], 'l2_groupings': [1], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['', '', ''], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key_attention_output(*args)]
