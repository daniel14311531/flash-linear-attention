# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.ops.conceptual_delta_rule import chunk_conceptual_delta_rule, fused_recurrent_conceptual_delta_rule
from fla.ops.conceptual_delta_rule.naive import naive_recurrent_conceptual_delta_rule
from fla.ops.conceptual_delta_rule.precondition import effective_beta_bwd, effective_beta_fwd
from fla.utils import assert_close, device


@pytest.mark.parametrize("eta", [0.1, 1.0, 100.0])
@pytest.mark.parametrize(("H", "HV", "K"), [(1, 1, 32), (2, 4, 64)])
def test_effective_beta(eta: float, H: int, HV: int, K: int):
    torch.manual_seed(42)
    k = (torch.randn(2, 17, H, K, device=device) * 1.5).requires_grad_()
    beta = torch.rand(2, 17, HV, device=device).requires_grad_()
    dc = torch.randn_like(beta)

    a = eta * beta
    norm2 = k.square().sum(-1).repeat_interleave(HV // H, dim=2)
    ref = a / (1 + a * norm2)
    (ref * dc).sum().backward()
    ref_dk, ref_dbeta = k.grad.clone(), beta.grad.clone()

    tri, inv_denom = effective_beta_fwd(k=k.detach(), beta=beta.detach(), eta=eta)
    tri_dk, tri_dbeta = effective_beta_bwd(
        k=k.detach(),
        effective_beta=tri,
        inv_denom=inv_denom,
        deffective_beta=dc,
        eta=eta,
    )

    assert_close("effective_beta", ref, tri, 1e-5)
    assert_close("dk", ref_dk, tri_dk, 1e-5)
    assert_close("dbeta", ref_dbeta, tri_dbeta, 1e-5)


@pytest.mark.parametrize("eta", [0.1, 10.0, 100.0])
def test_chunk(eta: float):
    torch.manual_seed(42)
    B, T, H, K, V = 2, 63, 2, 32, 24
    q = torch.randn(B, T, H, K, dtype=torch.float32, device=device).requires_grad_()
    k = (torch.randn(B, T, H, K, dtype=torch.float32, device=device) * 1.5).requires_grad_()
    v = torch.randn(B, T, H, V, dtype=torch.float32, device=device).requires_grad_()
    g = torch.empty(B, T, H, dtype=torch.float32, device=device).uniform_(-0.2, -0.01).requires_grad_()
    beta = torch.rand(B, T, H, dtype=torch.float32, device=device).requires_grad_()
    h0 = torch.randn(B, H, K, V, dtype=torch.float32, device=device).requires_grad_()
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    ref, ref_ht = naive_recurrent_conceptual_delta_rule(
        q=q,
        k=k,
        v=v,
        beta=beta,
        g=g,
        eta=eta,
        initial_state=h0,
        output_final_state=True,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_grads = [x.grad.clone() for x in (q, k, v, g, beta, h0)]
    for x in (q, k, v, g, beta, h0):
        x.grad = None

    tri, tri_ht = chunk_conceptual_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        eta=eta,
        initial_state=h0,
        output_final_state=True,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_grads = [x.grad for x in (q, k, v, g, beta, h0)]

    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)
    for name, ref_grad, tri_grad in zip(("dq", "dk", "dv", "dg", "dbeta", "dh0"), ref_grads, tri_grads):
        assert_close(name, ref_grad, tri_grad, 0.015)


def test_chunk_fused_transforms():
    torch.manual_seed(42)
    B, T, H, K, V, eta = 1, 31, 2, 32, 24, 100.0
    q = torch.randn(B, T, H, K, device=device).requires_grad_()
    k = (torch.randn(B, T, H, K, device=device) * 1.5).requires_grad_()
    v = torch.randn(B, T, H, V, device=device).requires_grad_()
    g = torch.empty(B, T, H, device=device).uniform_(-0.2, -0.01).requires_grad_()
    beta = torch.randn(B, T, H, device=device).requires_grad_()
    h0 = torch.randn(B, H, K, V, device=device).requires_grad_()
    do, dht = torch.randn_like(v), torch.randn_like(h0)

    ref, ref_ht = naive_recurrent_conceptual_delta_rule(
        q=F.normalize(q, p=2, dim=-1),
        k=F.normalize(k, p=2, dim=-1),
        v=v,
        beta=beta.sigmoid(),
        g=g,
        eta=eta,
        initial_state=h0,
        output_final_state=True,
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_grads = [x.grad.clone() for x in (q, k, v, g, beta, h0)]
    for x in (q, k, v, g, beta, h0):
        x.grad = None

    tri, tri_ht = chunk_conceptual_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        eta=eta,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_grads = [x.grad for x in (q, k, v, g, beta, h0)]

    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)
    for name, ref_grad, tri_grad in zip(("dq", "dk", "dv", "dg", "dbeta", "dh0"), ref_grads, tri_grads):
        assert_close(name, ref_grad, tri_grad, 0.015)


@pytest.mark.parametrize("eta", [0.1, 10.0, 100.0])
def test_fused_recurrent(eta: float):
    torch.manual_seed(42)
    B, T, H, K, V = 1, 31, 2, 32, 24
    q = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
    k = torch.randn(B, T, H, K, dtype=torch.float32, device=device) * 1.5
    v = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
    g = torch.empty(B, T, H, dtype=torch.float32, device=device).uniform_(-0.2, -0.01)
    beta = torch.rand(B, T, H, dtype=torch.float32, device=device)
    h0 = torch.randn(B, H, K, V, dtype=torch.float32, device=device)
    kwargs = dict(q=q, k=k, v=v, g=g, beta=beta, eta=eta, initial_state=h0, output_final_state=True)
    ref, ref_ht = naive_recurrent_conceptual_delta_rule(**kwargs)
    tri, tri_ht = fused_recurrent_conceptual_delta_rule(**kwargs)
    assert_close("o", ref, tri, 1e-4)
    assert_close("ht", ref_ht, tri_ht, 1e-4)


def test_chunk_varlen_gva():
    torch.manual_seed(42)
    T, H, HV, K, V, eta = 47, 2, 4, 32, 24, 10.0
    cu_seqlens = torch.tensor([0, 15, T], dtype=torch.long, device=device)
    q = torch.randn(1, T, H, K, device=device).requires_grad_()
    k = (torch.randn(1, T, H, K, device=device) * 1.5).requires_grad_()
    v = torch.randn(1, T, HV, V, device=device).requires_grad_()
    g = torch.empty(1, T, HV, device=device).uniform_(-0.2, -0.01).requires_grad_()
    beta = torch.rand(1, T, HV, device=device).requires_grad_()
    h0 = torch.randn(2, HV, K, V, device=device).requires_grad_()
    do, dht = torch.randn_like(v), torch.randn_like(h0)

    tri, tri_ht = chunk_conceptual_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        eta=eta,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens.cpu(),
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_grads = [x.grad.clone() for x in (q, k, v, g, beta, h0)]
    for x in (q, k, v, g, beta, h0):
        x.grad = None

    refs, ref_hts = [], []
    for i, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        start, end = start.item(), end.item()
        ref, ref_ht = naive_recurrent_conceptual_delta_rule(
            q=q[:, start:end].repeat_interleave(HV // H, dim=2),
            k=k[:, start:end].repeat_interleave(HV // H, dim=2),
            v=v[:, start:end],
            beta=beta[:, start:end],
            g=g[:, start:end],
            eta=eta,
            initial_state=h0[i:i + 1],
            output_final_state=True,
        )
        refs.append(ref)
        ref_hts.append(ref_ht)
    ref, ref_ht = torch.cat(refs, dim=1), torch.cat(ref_hts, dim=0)
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_grads = [x.grad for x in (q, k, v, g, beta, h0)]

    assert_close("o", ref, tri, 0.005)
    assert_close("ht", ref_ht, tri_ht, 0.005)
    for name, ref_grad, tri_grad in zip(("dq", "dk", "dv", "dg", "dbeta", "dh0"), ref_grads, tri_grads):
        assert_close(name, ref_grad, tri_grad, 0.015)


def test_eta_range():
    q = torch.zeros(1, 1, 1, 1, device=device)
    with pytest.raises(ValueError, match="eta"):
        chunk_conceptual_delta_rule(q=q, k=q, v=q, g=q[..., 0], beta=q[..., 0], eta=0.01)
