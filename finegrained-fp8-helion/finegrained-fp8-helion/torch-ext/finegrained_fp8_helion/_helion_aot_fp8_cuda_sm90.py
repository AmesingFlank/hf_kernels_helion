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
    # No features needed
    return 0


def autotune__w8a8_block_fp8_matmul(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [32], 'loop_orders': [[0, 1]], 'l2_groupings': [8], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 0], 'range_multi_buffers': [None, None], 'range_flattens': [None, None], 'load_eviction_policies': ['', '', '', ''], 'num_warps': 4, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key__w8a8_block_fp8_matmul(*args)]
