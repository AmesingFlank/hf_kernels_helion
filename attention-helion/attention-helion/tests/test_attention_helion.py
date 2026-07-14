import pytest
import torch
import torch.nn.functional as F

import attention_helion


def _sdpa_bshd(q, k, v, causal=False):
    """Reference SDPA on (B, S, H, D) inputs, matching flash_attn_func layout."""
    qt, kt, vt = (x.transpose(1, 2) for x in (q, k, v))
    out = F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal)
    return out.transpose(1, 2).contiguous()


@pytest.mark.kernels_ci
@pytest.mark.parametrize("causal", [False, True])
def test_attention_bhsd(device, causal):
    # Native (B, H, S, D) API vs SDPA.
    B, H, S, D = 2, 8, 512, 64
    q, k, v = (
        torch.randn(B, H, S, D, dtype=torch.float16, device=device) for _ in range(3)
    )
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    out = attention_helion.attention(q, k, v, causal=causal)
    torch.testing.assert_close(out, ref, atol=5e-2, rtol=2e-2)


@pytest.mark.kernels_ci
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_func_bshd(device, causal):
    # flash-attn (B, S, H, D) API vs SDPA, matching the kernels benchmark util.
    B, S, H, D = 4, 512, 16, 64
    q, k, v = (
        torch.randn(B, S, H, D, dtype=torch.float16, device=device) for _ in range(3)
    )
    ref = _sdpa_bshd(q, k, v, causal=causal)
    out = attention_helion.flash_attn_func(q, k, v, causal=causal)
    assert out.shape == (B, S, H, D)
    torch.testing.assert_close(out, ref, atol=5e-2, rtol=2e-2)


@pytest.mark.kernels_ci
def test_attention_layer(device):
    B, H, S, D = 2, 8, 256, 64
    q, k, v = (
        torch.randn(B, H, S, D, dtype=torch.float16, device=device) for _ in range(3)
    )
    layer = attention_helion.layers.Attention(causal=False)
    ref = F.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(layer(q, k, v), ref, atol=5e-2, rtol=2e-2)
