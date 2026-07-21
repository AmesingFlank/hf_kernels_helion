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
        {'block_sizes': [128], 'range_unroll_factors': [0, 0, 0, 0], 'range_warp_specializes': [None, None, None, None], 'range_num_stages': [0, 1, 0, 0], 'range_multi_buffers': [None, None, None, None], 'range_flattens': [None, None, None, None], 'static_ranges': [False, False], 'load_eviction_policies': ['', '', '', '', '', '', ''], 'num_warps': 32, 'num_stages': 4, 'indexing': ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
    ]
    return _C[key__deform_attn(*args)]
