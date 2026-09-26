# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.ops.flex_attn.softplus.decoding import flex_attn_decoding_one_step
from fla.ops.flex_attn.softplus.naive import naive_flex_attn_decoding, naive_parallel_flex_attn
from fla.ops.flex_attn.softplus.parallel import parallel_flex_attn
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    ('T', 'window_size', 'use_g', 'use_weight_bias', 'use_sink_bias'),
    [
        pytest.param(63, None, False, False, False, id='base'),
        pytest.param(65, None, False, False, True, id='sink'),
        pytest.param(100, 31, True, True, False, id='gated-window-weight'),
        pytest.param(129, None, True, True, True, id='gated-weight-sink'),
    ],
)
def test_parallel(T, window_size, use_g, use_weight_bias, use_sink_bias):
    torch.manual_seed(42)
    dtype = torch.float16
    B, H, HQ, D = 1, 2, 4, 32

    q = torch.randn(B, T, HQ, D, dtype=dtype, device=device)
    k = torch.randn(B, T, H, D, dtype=dtype, device=device)
    v = torch.randn(B, T, H, D, dtype=dtype, device=device)
    g = torch.randn(B, T, HQ, dtype=torch.float, device=device) * 0.02 if use_g else None
    weight_bias = torch.randn(HQ, dtype=torch.float, device=device) if use_weight_bias else None
    sink_bias = torch.randn(HQ, dtype=torch.float, device=device) if use_sink_bias else None
    do = torch.randn(B, T, HQ, D, dtype=dtype, device=device)

    q_ref = q.float().detach().requires_grad_(True)
    k_ref = k.float().detach().requires_grad_(True)
    v_ref = v.float().detach().requires_grad_(True)
    g_ref = g.detach().clone().requires_grad_(True) if g is not None else None
    weight_bias_ref = weight_bias.detach().clone().requires_grad_(True) if weight_bias is not None else None
    sink_bias_ref = sink_bias.detach().clone().requires_grad_(True) if sink_bias is not None else None
    ref = naive_parallel_flex_attn(
        q=q_ref,
        k=k_ref,
        v=v_ref,
        g=g_ref,
        weight_bias=weight_bias_ref,
        sink_bias=sink_bias_ref,
        window_size=window_size,
    ).to(dtype)
    ref.backward(do)

    q_tri = q.detach().requires_grad_(True)
    k_tri = k.detach().requires_grad_(True)
    v_tri = v.detach().requires_grad_(True)
    g_tri = g.detach().clone().requires_grad_(True) if g is not None else None
    weight_bias_tri = weight_bias.detach().clone().requires_grad_(True) if weight_bias is not None else None
    sink_bias_tri = sink_bias.detach().clone().requires_grad_(True) if sink_bias is not None else None
    tri = parallel_flex_attn(
        q=q_tri,
        k=k_tri,
        v=v_tri,
        g=g_tri,
        weight_bias=weight_bias_tri,
        sink_bias=sink_bias_tri,
        window_size=window_size,
    )
    tri.backward(do)

    assert_close(' o', ref, tri, 0.005)
    assert_close('dq', q_ref.grad, q_tri.grad, 0.01)
    assert_close('dk', k_ref.grad, k_tri.grad, 0.01)
    assert_close('dv', v_ref.grad, v_tri.grad, 0.01)
    if g is not None:
        assert_close('dg', g_ref.grad, g_tri.grad, 0.02)
    if weight_bias is not None:
        assert_close('dw', weight_bias_ref.grad, weight_bias_tri.grad, 0.02)
    if sink_bias is not None:
        assert_close('ds', sink_bias_ref.grad, sink_bias_tri.grad, 0.02)


@pytest.mark.smoke
def test_parallel_varlen():
    torch.manual_seed(42)
    dtype = torch.float16
    H, HQ, D = 2, 4, 32
    cu_seqlens = torch.tensor([0, 17, 63, 128], dtype=torch.int32, device=device)
    T = cu_seqlens[-1].item()

    q = torch.randn(1, T, HQ, D, dtype=dtype, device=device)
    k = torch.randn(1, T, H, D, dtype=dtype, device=device)
    v = torch.randn(1, T, H, D, dtype=dtype, device=device)
    g = torch.randn(1, T, HQ, dtype=torch.float, device=device) * 0.02
    weight_bias = torch.randn(HQ, dtype=torch.float, device=device)
    sink_bias = torch.randn(HQ, dtype=torch.float, device=device)
    do = torch.randn(1, T, HQ, D, dtype=dtype, device=device)

    q_ref = q.float().detach().requires_grad_(True)
    k_ref = k.float().detach().requires_grad_(True)
    v_ref = v.float().detach().requires_grad_(True)
    g_ref = g.detach().clone().requires_grad_(True)
    weight_bias_ref = weight_bias.detach().clone().requires_grad_(True)
    sink_bias_ref = sink_bias.detach().clone().requires_grad_(True)
    ref = torch.cat([
        naive_parallel_flex_attn(
            q=q_ref[:, bos:eos],
            k=k_ref[:, bos:eos],
            v=v_ref[:, bos:eos],
            g=g_ref[:, bos:eos],
            weight_bias=weight_bias_ref,
            sink_bias=sink_bias_ref,
            window_size=19,
        )
        for bos, eos in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=False)
    ], dim=1).to(dtype)
    ref.backward(do)

    q_tri = q.detach().requires_grad_(True)
    k_tri = k.detach().requires_grad_(True)
    v_tri = v.detach().requires_grad_(True)
    g_tri = g.detach().clone().requires_grad_(True)
    weight_bias_tri = weight_bias.detach().clone().requires_grad_(True)
    sink_bias_tri = sink_bias.detach().clone().requires_grad_(True)
    tri = parallel_flex_attn(
        q=q_tri,
        k=k_tri,
        v=v_tri,
        g=g_tri,
        weight_bias=weight_bias_tri,
        sink_bias=sink_bias_tri,
        window_size=19,
        cu_seqlens=cu_seqlens,
    )
    tri.backward(do)

    assert_close(' o', ref, tri, 0.005)
    assert_close('dq', q_ref.grad, q_tri.grad, 0.01)
    assert_close('dk', k_ref.grad, k_tri.grad, 0.01)
    assert_close('dv', v_ref.grad, v_tri.grad, 0.01)
    assert_close('dg', g_ref.grad, g_tri.grad, 0.02)
    assert_close('dw', weight_bias_ref.grad, weight_bias_tri.grad, 0.02)
    assert_close('ds', sink_bias_ref.grad, sink_bias_tri.grad, 0.02)


@pytest.mark.parametrize(
    ('cu_seqlens', 'use_g', 'do_gate_scale', 'use_weight_bias', 'use_sink_bias'),
    [
        pytest.param([0, 17, 46], False, False, False, False, id='base'),
        pytest.param([0, 0, 17, 46], True, True, True, True, id='empty-gated-weight-sink'),
    ],
)
def test_decoding(cu_seqlens, use_g, do_gate_scale, use_weight_bias, use_sink_bias):
    torch.manual_seed(42)
    dtype = torch.float16
    H, HQ, D = 2, 4, 32
    cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    N, T = len(cu_seqlens) - 1, cu_seqlens[-1].item()

    q = torch.randn(1, N, HQ, D, dtype=dtype, device=device)
    k = torch.randn(1, T, H, D, dtype=dtype, device=device)
    v = torch.randn(1, T, H, D, dtype=dtype, device=device)
    g = torch.randn(1, T, HQ, dtype=torch.float, device=device) * 0.02 if use_g else None
    weight_bias = torch.randn(HQ, dtype=torch.float, device=device) if use_weight_bias else None
    sink_bias = torch.randn(HQ, dtype=torch.float, device=device) if use_sink_bias else None

    ref = naive_flex_attn_decoding(
        q=q.float(),
        k=k.float(),
        v=v.float(),
        g=g,
        cu_seqlens=cu_seqlens,
        do_gate_scale=do_gate_scale,
        weight_bias=weight_bias,
        sink_bias=sink_bias,
    ).to(dtype)
    tri = flex_attn_decoding_one_step(
        q=q,
        k=k,
        v=v,
        g=g,
        cu_seqlens=cu_seqlens,
        do_gate_scale=do_gate_scale,
        weight_bias=weight_bias,
        sink_bias=sink_bias,
    )

    assert_close('o', ref, tri, 0.005)
