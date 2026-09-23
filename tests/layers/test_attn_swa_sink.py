# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
from einops import rearrange

from fla.layers.attn_swa_sink import AttentionSWASink
from fla.models.utils import Cache
from fla.ops.attn.naive import naive_parallel_attn
from fla.utils import device


def make_layer(window_size: int = 16, sink: bool = True, sink_init: float = 0.0, num_heads: int = 4, head_dim: int = 32):
    layer = AttentionSWASink(
        hidden_size=num_heads * head_dim,
        num_heads=num_heads,
        window_size=window_size,
        attention_sink=sink,
        sink_init=sink_init,
        rope_theta=10000.,
        max_position_embeddings=2048,
        layer_idx=0,
    ).to(device=device).to(torch.float32)
    # identity projections so the attention core can be inspected directly
    eye = torch.eye(num_heads * head_dim, device=device)
    with torch.no_grad():
        layer.q_proj.weight.copy_(eye)
        layer.k_proj.weight.copy_(eye)
        layer.v_proj.weight.copy_(eye)
        layer.o_proj.weight.copy_(eye)
    return layer


def attention_core(layer: AttentionSWASink, x: torch.Tensor):
    """Project and rotate inputs the same way the layer does.

    Args:
        layer: the attention layer.
        x (torch.Tensor): inputs of shape [B, T, H*D].

    Returns:
        tuple: q, k, v tensors of shape [B, T, H, D].
    """
    d = layer.head_dim
    q = rearrange(layer.q_proj(x), '... (h d) -> ... h d', d=d)
    k = rearrange(layer.k_proj(x), '... (h d) -> ... h d', d=d)
    v = rearrange(layer.v_proj(x), '... (h d) -> ... h d', d=d)
    q, k = layer.rotary(q, k, seqlen_offset=0, max_seqlen=x.shape[1])
    return q, k, v


@pytest.mark.parametrize('dtype', [torch.float32])
def test_attn_swa_sink_parallel_vs_naive(dtype):
    torch.manual_seed(42)
    B, T, H, D, W = 2, 64, 4, 32, 16
    layer = make_layer(W)
    x = torch.randn(B, T, H * D, device=device, dtype=dtype)

    q, k, v = attention_core(layer, x)
    o_naive, _ = naive_parallel_attn(q, k, v, scale=D ** -0.5, window_size=W, sink_bias=layer.sink_bias.detach())

    out, _, _ = layer(x)
    assert (out - o_naive.reshape(B, T, -1)).abs().max().item() / o_naive.abs().max().item() < 1e-2


@pytest.mark.parametrize('dtype', [torch.float32])
def test_attn_swa_sink_decode_matches_parallel(dtype):
    torch.manual_seed(42)
    B, T, H, D, W = 2, 64, 4, 32, 16
    layer = make_layer(W)
    x = torch.randn(B, T, H * D, device=device, dtype=dtype)
    out_full, _, _ = layer(x)

    # pure step-by-step decode from an empty cache
    outs = []
    cache = Cache()
    for i in range(T):
        o, _, cache = layer(x[:, i:i + 1], past_key_values=cache, use_cache=True)
        outs.append(o)
    assert (out_full - torch.cat(outs, 1)).abs().max().item() / out_full.abs().max().item() < 1e-2

    # prefill then decode the remainder
    cache = Cache()
    o_prefill, _, cache = layer(x[:, :T // 2], past_key_values=cache, use_cache=True)
    outs = [o_prefill]
    for i in range(T // 2, T):
        o, _, cache = layer(x[:, i:i + 1], past_key_values=cache, use_cache=True)
        outs.append(o)
    assert (out_full - torch.cat(outs, 1)).abs().max().item() / out_full.abs().max().item() < 1e-2


def test_attn_swa_sink_semantics():
    torch.manual_seed(42)
    B, T, H, D, W = 1, 64, 4, 32, 16
    layer = make_layer(W)
    x = torch.randn(B, T, H * D, device=device)

    # a huge sink bias absorbs all attention mass, so the output vanishes
    with torch.no_grad():
        layer.sink_bias.fill_(20.)
    out, _, _ = layer(x)
    assert out.abs().max().item() < 1e-2

    # a very negative sink bias recovers plain window attention
    with torch.no_grad():
        layer.sink_bias.fill_(-1e9)
    out, _, _ = layer(x)
    q, k, v = attention_core(layer, x)
    o_plain, _ = naive_parallel_attn(q, k, v, scale=D ** -0.5, window_size=W)
    assert (out - o_plain.reshape(B, T, -1)).abs().max().item() / o_plain.abs().max().item() < 1e-2


def test_attn_swa_sink_window_isolation():
    torch.manual_seed(42)
    H, D, W = 4, 32, 16
    layer = make_layer(W)
    x = torch.randn(1, 128, H * D, device=device)
    out, _, _ = layer(x)
    # perturbing tokens strictly outside the window of position 100 must not change its output
    x_pert = x.clone()
    x_pert[0, :80] += 10.
    out_pert, _, _ = layer(x_pert)
    assert torch.equal(out[0, 100], out_pert[0, 100])


def test_attn_swa_sink_padding():
    torch.manual_seed(42)
    H, D, W = 4, 32, 16
    layer = make_layer(W)
    x = torch.randn(3, 96, H * D, device=device)
    mask = torch.ones(3, 96, dtype=torch.long, device=device)
    for b, l in enumerate([96, 77, 40]):
        mask[b, l:] = 0

    # prefill with padding
    out_masked, _, _ = layer(x, attention_mask=mask)
    for b, l in enumerate([96, 77, 40]):
        out_single, _, _ = layer(x[b: b + 1, :l])
        assert torch.allclose(out_masked[b, :l], out_single[0], atol=1e-3)

    # one decoding step with left padding (the HF decoder-only convention;
    # right padding is unsupported for decode, matching fla.layers.attn.Attention)
    left_pad = [5, 0, 11]
    mask = torch.ones(3, 96, dtype=torch.long, device=device)
    for b, p in enumerate(left_pad):
        mask[b, :p] = 0
    cache = Cache()
    _, _, cache = layer(x, attention_mask=mask, past_key_values=cache, use_cache=True)
    xd = torch.randn(3, 1, H * D, device=device)
    mask_d = torch.cat([mask, torch.ones(3, 1, dtype=torch.long, device=device)], 1)
    o_masked, _, _ = layer(xd, attention_mask=mask_d, past_key_values=cache, use_cache=True)
    for b in range(3):
        valid = mask[b].bool()
        c = Cache()
        layer(x[b: b + 1][:, valid], past_key_values=c, use_cache=True)
        o_single, _, _ = layer(xd[b: b + 1], past_key_values=c, use_cache=True)
        assert torch.allclose(o_masked[b], o_single[0], atol=1e-3)


def test_attn_swa_sink_gradients_and_gqa():
    torch.manual_seed(42)
    layer = AttentionSWASink(
        hidden_size=256, num_heads=8, num_kv_heads=4, window_size=128, layer_idx=0,
    ).to(device=device).to(torch.float32)
    x = torch.randn(2, 100, 256, device=device, requires_grad=True)
    out, _, _ = layer(x)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.sink_bias.grad is not None and layer.sink_bias.grad.abs().sum() > 0
