"""
Auto-generated heuristic for kernel: _rms_norm
Backend: decision_tree

Provides:
- key__rms_norm(*args): Returns config index (cache key)
- autotune__rms_norm(*args): Returns config dict for the given arguments
"""

import torch


def key__rms_norm(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    _arg0_dim0 = int(args[0].shape[0]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 0 else 0
    if _arg0_dim0 <= 16384.0:
        if _arg0_dim0 <= 2048.0:
            return 0
        else:
            return 1
    else:
        return 0


def autotune__rms_norm(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [1], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['last', 'last', 'last', 'last', 'last'], 'num_warps': 8, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [4], 'reduction_loops': [None], 'range_unroll_factors': [0], 'range_warp_specializes': [], 'range_num_stages': [0], 'range_multi_buffers': [None], 'range_flattens': [None], 'load_eviction_policies': ['', '', '', '', ''], 'num_warps': 8, 'num_stages': 2, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key__rms_norm(*args)]
