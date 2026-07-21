"""
Auto-generated heuristic for kernel: _grouped_gemm
Backend: decision_tree

Provides:
- key__grouped_gemm(*args): Returns config index (cache key)
- autotune__grouped_gemm(*args): Returns config dict for the given arguments
"""

import torch


def key__grouped_gemm(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    # No features needed
    return 0


def autotune__grouped_gemm(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [256, 256, 32], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 0, 2, 0], 'range_warp_specializes': [None, None, False, None], 'range_num_stages': [0, 1, 2, 0], 'range_multi_buffers': [None, None, None, False], 'range_flattens': [None, None, True, False], 'load_eviction_policies': ['last', 'last', '', 'last'], 'num_warps': 8, 'num_stages': 6, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat', 'epilogue_subtile': 2},
    ]
    return _C[key__grouped_gemm(*args)]
