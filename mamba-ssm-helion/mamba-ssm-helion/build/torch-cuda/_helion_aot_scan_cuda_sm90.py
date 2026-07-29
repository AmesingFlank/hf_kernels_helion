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
    _arg0_dim1 = int(args[0].shape[1]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 1 else 0
    if _arg0_dim1 <= 1024.0:
        return 1
    else:
        return 0


def autotune__selective_scan(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [1, 8], 'loop_orders': [[0, 1]], 'l2_groupings': [1], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['', '', '', '', '', '', ''], 'num_warps': 1, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1, 2], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['', '', '', '', '', '', ''], 'num_warps': 1, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key__selective_scan(*args)]
