"""
Auto-generated heuristic for kernel: _deform_attn
Backend: decision_tree

Provides:
- key__deform_attn(*args): Returns config index (cache key)
- autotune__deform_attn(*args): Returns config dict for the given arguments
"""

import torch


def key__deform_attn(*args) -> int:
    """Select config index for the given arguments (also serves as cache key)."""
    # No features needed
    return 0


def autotune__deform_attn(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [128], 'range_unroll_factors': [0, 4, 2, 0], 'range_warp_specializes': [None, None, False, None], 'range_num_stages': [0, 4, 2, 0], 'range_multi_buffers': [None, False, None, None], 'range_flattens': [None, True, False, None], 'static_ranges': [False, True], 'load_eviction_policies': ['', '', '', '', '', '', ''], 'num_warps': 32, 'num_stages': 2, 'indexing': ['pointer', 'tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key__deform_attn(*args)]
