"""
Auto-generated heuristic for kernel: _w8a8_block_fp8_matmul
Backend: decision_tree

Provides:
- key__w8a8_block_fp8_matmul(*args): Returns config index (cache key)
- autotune__w8a8_block_fp8_matmul(*args): Returns config dict for the given arguments
"""

import torch


def key__w8a8_block_fp8_matmul(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    _arg0_dim0 = int(args[0].shape[0]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 0 else 0
    if _arg0_dim0 <= 512.0:
        return 1
    else:
        return 0


def autotune__w8a8_block_fp8_matmul(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [256], 'loop_orders': [[0, 1]], 'l2_groupings': [1], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [None, None], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['', '', '', ''], 'num_warps': 8, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [32], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [3, 2], 'range_warp_specializes': [False, False], 'range_num_stages': [0, 2], 'range_multi_buffers': [True, True], 'range_flattens': [None, True], 'load_eviction_policies': ['', '', 'first', ''], 'num_warps': 16, 'num_stages': 3, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 4, 'maxnreg': 128},
    ]
    return _C[key__w8a8_block_fp8_matmul(*args)]
