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
    _arg0_dim1 = int(args[0].shape[1]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 1 else 0
    _arg0_dim2 = int(args[0].shape[2]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 2 else 0
    _arg0_dim3 = int(args[0].shape[3]) if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].ndim > 3 else 0
    _arg0_numel = int(args[0].numel()) if len(args) > 0 and isinstance(args[0], torch.Tensor) else 0
    if _arg0_dim3 <= 64.0:
        if _arg0_numel <= 1048576.0:
            if _arg0_dim0 <= 1.0:
                return 5
            else:
                return 6
        else:
            if _arg0_dim3 <= 32.0:
                return 2
            else:
                if _arg0_dim1 <= 16.0:
                    return 1
                else:
                    return 4
    else:
        if _arg0_dim3 <= 128.0:
            if _arg0_numel <= 4194304.0:
                return 3
            else:
                if _arg0_dim2 <= 2048.0:
                    return 0
                else:
                    if _arg0_dim0 <= 1.0:
                        return 0
                    else:
                        return 3
        else:
            return 2


def autotune_attention_output(*args) -> dict:
    """Select the optimal config for the given arguments."""
    _C = [
        {'block_sizes': [1, 128, 128], 'loop_orders': [[0, 1]], 'l2_groupings': [8], 'range_unroll_factors': [0, 2], 'range_warp_specializes': [], 'range_num_stages': [0, 3], 'range_multi_buffers': [None, None], 'range_flattens': [None, False], 'load_eviction_policies': ['last', '', 'first'], 'num_warps': 8, 'num_stages': 3, 'indexing': ['pointer', 'tensor_descriptor', 'pointer', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 1},
        {'block_sizes': [1, 128, 64], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 3], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, None], 'range_flattens': [None, True], 'load_eviction_policies': ['', 'last', 'first'], 'num_warps': 4, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'tensor_descriptor', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1, 128, 128], 'loop_orders': [[0, 1]], 'l2_groupings': [16], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, False], 'range_flattens': [None, True], 'load_eviction_policies': ['', 'last', 'last'], 'num_warps': 8, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1, 128, 128], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, True], 'range_flattens': [None, False], 'load_eviction_policies': ['last', 'first', 'first'], 'num_warps': 8, 'num_stages': 3, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'persistent_interleaved', 'num_sm_multiplier': 1},
        {'block_sizes': [1, 128, 64], 'loop_orders': [[0, 1]], 'l2_groupings': [1], 'range_unroll_factors': [0, 4], 'range_warp_specializes': [], 'range_num_stages': [0, 1], 'range_multi_buffers': [None, None], 'range_flattens': [None, True], 'load_eviction_policies': ['first', '', 'last'], 'num_warps': 4, 'num_stages': 4, 'indexing': ['tensor_descriptor', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1, 64, 128], 'loop_orders': [[1, 0]], 'l2_groupings': [64], 'range_unroll_factors': [0, 1], 'range_warp_specializes': [], 'range_num_stages': [0, 4], 'range_multi_buffers': [None, None], 'range_flattens': [None, True], 'load_eviction_policies': ['first', '', 'last'], 'num_warps': 4, 'num_stages': 1, 'indexing': ['pointer', 'pointer', 'pointer', 'tensor_descriptor'], 'atomic_indexing': [], 'pid_type': 'flat'},
        {'block_sizes': [1, 16, 128], 'loop_orders': [[1, 0]], 'l2_groupings': [1], 'range_unroll_factors': [0, 0], 'range_warp_specializes': [], 'range_num_stages': [0, 2], 'range_multi_buffers': [None, True], 'range_flattens': [None, True], 'load_eviction_policies': ['last', 'first', ''], 'num_warps': 4, 'num_stages': 2, 'indexing': ['tensor_descriptor', 'pointer', 'tensor_descriptor', 'pointer'], 'atomic_indexing': [], 'pid_type': 'persistent_blocked', 'num_sm_multiplier': 1},
    ]
    return _C[key_attention_output(*args)]
