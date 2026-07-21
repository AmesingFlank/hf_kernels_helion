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
    _arg0_dim0 = int(args[0].shape[0]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 0 else 0
    if _arg0_dim0 <= 2.0:
        return 1
    else:
        return 0


def autotune__selective_scan(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [1, 16], 'loop_orders': [[1, 0]], 'l2_groupings': [32], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [None, False], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, True], 'range_flattens': [None, True], 'load_eviction_policies': ['last', 'first', '', '', '', '', ''], 'num_warps': 2, 'num_stages': 1, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1, 2], 'loop_orders': [[1, 0]], 'l2_groupings': [8], 'range_unroll_factors': [3, 4], 'range_warp_specializes': [False, False], 'range_num_stages': [3, 3], 'range_multi_buffers': [True, None], 'range_flattens': [None, None], 'load_eviction_policies': ['last', 'first', '', '', '', '', ''], 'num_warps': 1, 'num_stages': 6, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 2, 'maxnreg': 256},
    ]
    return _C[key__selective_scan(*args)]
