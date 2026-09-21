# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.utils import input_guard


@triton.jit
def effective_beta_fwd_kernel(
    k,
    beta,
    effective_beta,
    inv_denom,
    eta,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    G: tl.constexpr,
):
    i_bt = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1).to(tl.int64)

    o_k = tl.arange(0, BK)
    p_k = k + (i_bt * H + i_h) * K + o_k
    b_k = tl.load(p_k, mask=o_k < K, other=0.0).to(tl.float32)
    b_norm2 = tl.sum(b_k * b_k, axis=0)

    for i_g in range(G):
        i_hv = i_h * G + i_g
        p_beta = beta + i_bt * HV + i_hv
        p_effective_beta = effective_beta + i_bt * HV + i_hv
        p_inv_denom = inv_denom + i_bt * HV + i_hv

        b_beta = tl.load(p_beta).to(tl.float32)
        b_a = b_beta * eta
        b_x = b_a * b_norm2
        b_denom = tl.fma(b_a, b_norm2, 1.0)
        b_c_direct = b_a / b_denom

        b_a_safe = tl.where(b_a > 0.0, b_a, 1.0)
        b_c_recip = 1.0 / (b_norm2 + 1.0 / b_a_safe)
        b_c = tl.where(b_x <= 1.0, b_c_direct, b_c_recip)
        b_c = tl.where(b_a == 0.0, 0.0, b_c)

        b_inv_direct = 1.0 / b_denom
        b_inv_recip = b_c / b_a_safe
        b_inv = tl.where(b_x <= 1.0, b_inv_direct, b_inv_recip)
        b_inv = tl.where(b_a == 0.0, 1.0, b_inv)

        tl.store(p_effective_beta, b_c)
        tl.store(p_inv_denom, b_inv)


@triton.jit
def effective_beta_bwd_kernel(
    k,
    effective_beta,
    inv_denom,
    deffective_beta,
    dk,
    dbeta,
    eta,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    G: tl.constexpr,
):
    i_bt = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1).to(tl.int64)

    b_coeff = tl.zeros([1], dtype=tl.float32)
    for i_g in range(G):
        i_hv = i_h * G + i_g
        p_c = effective_beta + i_bt * HV + i_hv
        p_inv = inv_denom + i_bt * HV + i_hv
        p_dc = deffective_beta + i_bt * HV + i_hv
        p_dbeta = dbeta + i_bt * HV + i_hv

        b_c = tl.load(p_c).to(tl.float32)
        b_inv = tl.load(p_inv).to(tl.float32)
        b_dc = tl.load(p_dc).to(tl.float32)
        b_coeff += (b_dc * b_c) * b_c
        tl.store(p_dbeta, (b_dc * eta * b_inv * b_inv).to(dbeta.dtype.element_ty))

    o_k = tl.arange(0, BK)
    p_k = k + (i_bt * H + i_h) * K + o_k
    p_dk = dk + (i_bt * H + i_h) * K + o_k
    b_k = tl.load(p_k, mask=o_k < K, other=0.0).to(tl.float32)
    tl.store(p_dk, (-2.0 * b_coeff * b_k).to(dk.dtype.element_ty), mask=o_k < K)


@input_guard
def effective_beta_fwd(
    k: torch.Tensor,
    beta: torch.Tensor,
    eta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K = k.shape
    HV = beta.shape[2]
    BK = triton.next_power_of_2(K)
    effective_beta = torch.empty_like(beta, dtype=torch.float32)
    inv_denom = torch.empty_like(beta, dtype=torch.float32)
    effective_beta_fwd_kernel[(B * T, H)](
        k=k,
        beta=beta,
        effective_beta=effective_beta,
        inv_denom=inv_denom,
        eta=eta,
        H=H,
        HV=HV,
        K=K,
        BK=BK,
        G=HV // H,
    )
    return effective_beta, inv_denom


@input_guard
def effective_beta_bwd(
    k: torch.Tensor,
    effective_beta: torch.Tensor,
    inv_denom: torch.Tensor,
    deffective_beta: torch.Tensor,
    eta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K = k.shape
    HV = effective_beta.shape[2]
    BK = triton.next_power_of_2(K)
    dk = torch.empty_like(k)
    dbeta = torch.empty_like(deffective_beta)
    effective_beta_bwd_kernel[(B * T, H)](
        k=k,
        effective_beta=effective_beta,
        inv_denom=inv_denom,
        deffective_beta=deffective_beta,
        dk=dk,
        dbeta=dbeta,
        eta=eta,
        H=H,
        HV=HV,
        K=K,
        BK=BK,
        G=HV // H,
    )
    return dk, dbeta
