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
        {'block_sizes': [128, 256, 64], 'loop_orders': [[0, 1]], 'range_unroll_factors': [0, 0, 0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 0, 0, 0], 'range_multi_buffers': [None, None, None, None], 'range_flattens': [None, None, None, None], 'load_eviction_policies': ['', '', '', ''], 'num_warps': 8, 'num_stages': 3, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 1},
    ]
    return _C[key__grouped_gemm(*args)]
