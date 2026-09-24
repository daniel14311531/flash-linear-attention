# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl
from einops import reduce

from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.cumsum import chunk_global_cumsum
from fla.ops.utils.softplus import softplus
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, check_shared_mem, contiguous


@triton.heuristics({
    'USE_G': lambda args: args['g_cumsum'] is not None,
    'USE_WEIGHT_BIAS': lambda args: args['weight_bias'] is not None,
    'USE_SINK_BIAS': lambda args: args['sink_bias'] is not None,
    'USE_WINDOW': lambda args: args['W'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def parallel_flex_attn_fwd_kernel(
    q,
    k,
    v,
    o,
    g_cumsum,
    weight_bias,
    sink_bias,
    normalizer,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    W: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_WEIGHT_BIAS: tl.constexpr,
    USE_SINK_BIAS: tl.constexpr,
    USE_WINDOW: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        i_n = i_b
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
    # [BT]
    o_q = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_q = o_q < T
    p_q = q + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_d[None, :]
    p_o = o + (bos * HQ + i_hq) * V + o_q[:, None] * (HQ*V) + o_v[None, :]
    p_normalizer = normalizer + bos * HQ + i_hq + o_q * HQ

    # the Q block is kept in the shared memory throughout the whole kernel
    # [BT, BK]
    b_q = tl.load(p_q, mask=m_q[:, None] & (o_d[None, :] < K), other=0.0)
    # [BT, BV]
    b_o = tl.zeros([BT, BV], dtype=tl.float32)

    b_acc = tl.zeros([BT], dtype=tl.float32)

    if USE_G:
        p_g = g_cumsum + bos * HQ + i_hq + o_q * HQ
        b_gq = tl.load(p_g, mask=m_q, other=0.0).to(tl.float32)
    else:
        b_gq = None

    if USE_WEIGHT_BIAS:
        b_weight_bias = tl.load(weight_bias + i_hq).to(tl.float32)
    else:
        b_weight_bias = None

    if USE_SINK_BIAS:
        b_sink_bias = tl.load(sink_bias + i_hq).to(tl.float32)
    else:
        b_sink_bias = None

    # for sliding window, skip key blocks that are entirely outside the window.
    # the earliest key position any query in this block needs: max(0, i_t*BT - W + 1)
    i_start = tl.maximum((i_t * BT - W + 1) // BS * BS, 0) if USE_WINDOW else 0

    for i_s in range(i_start, i_t * BT, BS):
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        p_k = k + (bos * H + i_h) * K + o_d[:, None] + o_k[None, :] * (H*K)
        p_v = v + (bos * H + i_h) * V + o_k[:, None] * (H*V) + o_v[None, :]
        # [BK, BS]
        b_k = tl.load(p_k, mask=(o_d[:, None] < K) & m_k[None, :], other=0.0)
        # [BS, BV]
        b_v = tl.load(p_v, mask=m_k[:, None] & (o_v[None, :] < V), other=0.0)
        # [BT, BS]
        b_s = tl.dot(b_q, b_k) * scale

        if USE_G:
            b_gk = tl.load(g_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0).to(tl.float32)
            b_s += b_gq[:, None] - b_gk[None, :]
        if USE_WEIGHT_BIAS:
            b_s += b_weight_bias

        if USE_WINDOW:
            b_s = tl.where((o_q[:, None] - o_k[None, :] < W) & m_k[None, :], b_s, float('-inf'))

        b_p = softplus(b_s)
        # [BT]
        b_acc += tl.sum(b_p, 1)
        # [BT, BV]
        b_o += tl.dot(b_p.to(b_q.dtype), b_v)

    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        # [BS]
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        p_k = k + (bos * H + i_h) * K + o_d[:, None] + o_k[None, :] * (H*K)
        p_v = v + (bos * H + i_h) * V + o_k[:, None] * (H*V) + o_v[None, :]

        # [BK, BS]
        b_k = tl.load(p_k, mask=(o_d[:, None] < K) & m_k[None, :], other=0.0)
        # [BS, BV]
        b_v = tl.load(p_v, mask=m_k[:, None] & (o_v[None, :] < V), other=0.0)
        # [BT, BS]
        b_s = tl.dot(b_q, b_k) * scale

        if USE_G:
            b_gk = tl.load(g_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0).to(tl.float32)
            b_s += b_gq[:, None] - b_gk[None, :]
        if USE_WEIGHT_BIAS:
            b_s += b_weight_bias

        m_s = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        if USE_WINDOW:
            m_s = m_s & (o_q[:, None] - o_k[None, :] < W)
        b_s = tl.where(m_s, b_s, float('-inf'))

        b_p = softplus(b_s)
        # [BT]
        b_acc += tl.sum(b_p, 1)
        # [BT, BV]
        b_o += tl.dot(b_p.to(b_q.dtype), b_v)

    if USE_SINK_BIAS:
        if USE_WEIGHT_BIAS:
            b_sink_bias += b_weight_bias
        b_acc += softplus(b_sink_bias)

    b_o = b_o / b_acc[:, None]
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_q[:, None] & (o_v[None, :] < V))
    if i_v == 0:
        tl.store(p_normalizer, b_acc.to(p_normalizer.dtype.element_ty), mask=m_q)


@triton.jit
def parallel_flex_attn_bwd_kernel_preprocess(
    o,
    do,
    delta,
    B: tl.constexpr,
    V: tl.constexpr,
):
    i_n = tl.program_id(0).to(tl.int64)
    o_d = tl.arange(0, B)
    m_d = o_d < V

    b_o = tl.load(o + i_n * V + o_d, mask=m_d, other=0)
    b_do = tl.load(do + i_n * V + o_d, mask=m_d, other=0).to(tl.float32)
    b_delta = tl.sum(b_o * b_do)

    tl.store(delta + i_n, b_delta.to(delta.dtype.element_ty))


@triton.heuristics({
    'USE_G': lambda args: args['g_cumsum'] is not None,
    'USE_WEIGHT_BIAS': lambda args: args['weight_bias'] is not None,
    'USE_WINDOW': lambda args: args['W'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def parallel_flex_attn_bwd_kernel_dq(
    q,
    k,
    v,
    normalizer,
    delta,
    do,
    dq,
    dg_cumsum,
    dweight_bias_rows,
    g_cumsum,
    weight_bias,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    W: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_WEIGHT_BIAS: tl.constexpr,
    USE_WINDOW: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        i_n = i_b
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
    o_q = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_q = o_q < T
    p_q = q + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_d[None, :]
    p_dq = dq + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_d[None, :]
    p_do = do + (bos * HQ + i_hq) * V + o_q[:, None] * (HQ*V) + o_v[None, :]
    p_normalizer = normalizer + bos * HQ + i_hq + o_q * HQ
    p_delta = delta + bos * HQ + i_hq + o_q * HQ

    # [BT, BK]
    b_q = tl.load(p_q, mask=m_q[:, None] & (o_d[None, :] < K), other=0.0)
    # [BT, BV]
    b_do = tl.load(p_do, mask=m_q[:, None] & (o_v[None, :] < V), other=0.0)
    # [BT]
    b_normalizer = tl.load(p_normalizer, mask=m_q, other=1.0)
    b_delta = tl.load(p_delta, mask=m_q, other=0.0)

    # [BT, BK]
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    if USE_G:
        b_dg = tl.zeros([BT], dtype=tl.float32)
        p_gq = g_cumsum + bos * HQ + i_hq + o_q * HQ
        b_gq = tl.load(p_gq, mask=m_q, other=0.0).to(tl.float32)
    else:
        b_gq = None
        b_dg = None
    if USE_WEIGHT_BIAS:
        b_weight_bias = tl.load(weight_bias + i_hq).to(tl.float32)
        b_dweight_bias = tl.zeros([BT], dtype=tl.float32)
    else:
        b_weight_bias = None
        b_dweight_bias = None

    i_start = tl.maximum((i_t * BT - W + 1) // BS * BS, 0) if USE_WINDOW else 0

    for i_s in range(i_start, i_t * BT, BS):
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        p_k = k + (bos * H + i_h) * K + o_d[:, None] + o_k[None, :] * (H*K)
        p_v = v + (bos * H + i_h) * V + o_v[:, None] + o_k[None, :] * (H*V)

        # [BK, BS]
        b_k = tl.load(p_k, mask=(o_d[:, None] < K) & m_k[None, :], other=0.0)
        # [BV, BS]
        b_v = tl.load(p_v, mask=(o_v[:, None] < V) & m_k[None, :], other=0.0)
        # [BT, BS]
        b_s = tl.dot(b_q, b_k) * scale
        if USE_G:
            b_gk = tl.load(g_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0).to(tl.float32)
            b_s += b_gq[:, None] - b_gk[None, :]
        if USE_WEIGHT_BIAS:
            b_s += b_weight_bias

        if USE_WINDOW:
            b_s = tl.where((o_q[:, None] - o_k[None, :] < W) & m_k[None, :], b_s, float('-inf'))
        b_dw = tl.sigmoid(b_s)
        # [BT, BV] @ [BV, BS] -> [BT, BS]
        b_dp = tl.dot(b_do, b_v)
        b_ds = b_dw / b_normalizer[:, None] * (b_dp.to(tl.float32) - b_delta[:, None])
        # [BT, BS] @ [BS, BK] -> [BT, BK]
        b_dq = tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k), b_dq)
        if USE_G:
            b_dg += tl.sum(b_ds, 1)
        if USE_WEIGHT_BIAS:
            b_dweight_bias += tl.sum(b_ds, 1)

    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        # [BS]
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        p_k = k + (bos * H + i_h) * K + o_d[:, None] + o_k[None, :] * (H*K)
        p_v = v + (bos * H + i_h) * V + o_v[:, None] + o_k[None, :] * (H*V)

        # [BK, BS]
        b_k = tl.load(p_k, mask=(o_d[:, None] < K) & m_k[None, :], other=0.0)
        # [BV, BS]
        b_v = tl.load(p_v, mask=(o_v[:, None] < V) & m_k[None, :], other=0.0)
        # [BT, BS]
        b_s = tl.dot(b_q, b_k) * scale

        if USE_G:
            p_gk = g_cumsum + bos * HQ + i_hq + o_k * HQ
            b_gk = tl.load(p_gk, mask=m_k, other=0.0).to(tl.float32)
            b_s += b_gq[:, None] - b_gk[None, :]
        if USE_WEIGHT_BIAS:
            b_s += b_weight_bias
        m_s = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        if USE_WINDOW:
            m_s = m_s & (o_q[:, None] - o_k[None, :] < W)
        b_dw = tl.where(m_s, tl.sigmoid(b_s), 0.0)

        # [BT, BV] @ [BV, BS] -> [BT, BS]
        b_dp = tl.dot(b_do, b_v)
        b_ds = b_dw / b_normalizer[:, None] * (b_dp.to(tl.float32) - b_delta[:, None])
        # [BT, BS] @ [BS, BK] -> [BT, BK]
        b_dq = tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k), b_dq)
        if USE_G:
            b_dg += tl.sum(b_ds, 1)
        if USE_WEIGHT_BIAS:
            b_dweight_bias += tl.sum(b_ds, 1)

    b_dq *= scale
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_q[:, None] & (o_d[None, :] < K))
    if USE_G:
        p_dg = dg_cumsum + bos * HQ + i_hq + o_q * HQ
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=m_q)
    if USE_WEIGHT_BIAS:
        p_dweight_bias = dweight_bias_rows + bos * HQ + i_hq + o_q * HQ
        tl.store(p_dweight_bias, b_dweight_bias, mask=m_q)


@triton.heuristics({
    'USE_G': lambda args: args['g_cumsum'] is not None,
    'USE_WEIGHT_BIAS': lambda args: args['weight_bias'] is not None,
    'USE_WINDOW': lambda args: args['W'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def parallel_flex_attn_bwd_kernel_dkv(
    q,
    k,
    v,
    g_cumsum,
    weight_bias,
    normalizer,
    delta,
    do,
    dk,
    dv,
    dg_cumsum,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    W: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_WEIGHT_BIAS: tl.constexpr,
    USE_WINDOW: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        i_n = i_b
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
    o_k = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < T
    p_k = k + (bos * H + i_h) * K + o_k[:, None] * (H*K) + o_d[None, :]
    p_v = v + (bos * H + i_h) * V + o_k[:, None] * (H*V) + o_v[None, :]
    p_dk = dk + (bos * HQ + i_hq) * K + o_k[:, None] * (HQ*K) + o_d[None, :]
    p_dv = dv + (bos * HQ + i_hq) * V + o_k[:, None] * (HQ*V) + o_v[None, :]

    # [BT, BK]
    b_k = tl.load(p_k, mask=m_k[:, None] & (o_d[None, :] < K), other=0.0)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    # [BT, BV]
    b_v = tl.load(p_v, mask=m_k[:, None] & (o_v[None, :] < V), other=0.0)
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)

    if USE_G:
        p_gk = g_cumsum + bos * HQ + i_hq + o_k * HQ
        b_gk = tl.load(p_gk, mask=m_k, other=0.0).to(tl.float32)
        b_dg = tl.zeros([BT], dtype=tl.float32)
    else:
        b_gk = None
        b_dg = None
    if USE_WEIGHT_BIAS:
        b_weight_bias = tl.load(weight_bias + i_hq).to(tl.float32)
    else:
        b_weight_bias = None

    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        # [BS]
        o_q = i_s + tl.arange(0, BS)
        m_q = o_q < T
        p_q = q + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_d[None, :]
        p_do = do + (bos * HQ + i_hq) * V + o_q[:, None] * (HQ*V) + o_v[None, :]
        p_normalizer = normalizer + bos * HQ + i_hq + o_q * HQ
        p_delta = delta + bos * HQ + i_hq + o_q * HQ

        # [BS, BK]
        b_q = tl.load(p_q, mask=m_q[:, None] & (o_d[None, :] < K), other=0.0)
        # [BS, BV]
        b_do = tl.load(p_do, mask=m_q[:, None] & (o_v[None, :] < V), other=0.0)
        # [BS]
        b_normalizer = tl.load(p_normalizer, mask=m_q, other=1.0)
        b_delta = tl.load(p_delta, mask=m_q, other=0.0)
        # [BT, BS]
        b_s = tl.dot(b_k, tl.trans(b_q)) * scale
        if USE_G:
            p_gq = g_cumsum + bos * HQ + i_hq + o_q * HQ
            b_gq = tl.load(p_gq, mask=m_q, other=0.0).to(tl.float32)
            b_s += b_gq[None, :] - b_gk[:, None]
        if USE_WEIGHT_BIAS:
            b_s += b_weight_bias
        m_s = (o_k[:, None] <= o_q[None, :]) & m_q[None, :]
        if USE_WINDOW:
            m_s = m_s & (o_q[None, :] - o_k[:, None] < W)
        b_w = tl.where(m_s, softplus(b_s), 0.0)
        b_p = b_w / b_normalizer[None, :]
        # [BT, BS] @ [BS, BV] -> [BT, BV]
        b_dv = tl.dot(b_p.to(b_do.dtype), b_do, b_dv)
        # [BT, BV] @ [BV, BS] -> [BT, BS]
        b_dp = tl.dot(b_v, tl.trans(b_do))
        # [BT, BS]
        b_dw = tl.where(m_s, tl.sigmoid(b_s), 0.0)
        b_ds = b_dw / b_normalizer[None, :] * (b_dp - b_delta[None, :])
        # [BT, BS] @ [BS, BK] -> [BT, BK]
        b_dk = tl.dot(b_ds.to(b_q.dtype), b_q, b_dk)
        if USE_G:
            b_dg -= tl.sum(b_ds, 1)

    # for sliding window, limit the range of future query blocks to process.
    # a key at position k_pos can only be attended to by queries at positions [k_pos, k_pos + W - 1].
    # so we only need queries up to (i_t + 1) * BT - 1 + W - 1.
    i_end = min(tl.cdiv(T, BS) * BS, (i_t + 1) * BT + W - 1) if USE_WINDOW else tl.cdiv(T, BS) * BS

    for i_s in range((i_t + 1) * BT, i_end, BS):
        # [BS]
        o_q = i_s + tl.arange(0, BS)
        m_q = o_q < T
        p_q = q + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_d[None, :]
        p_do = do + (bos * HQ + i_hq) * V + o_q[:, None] * (HQ*V) + o_v[None, :]
        p_normalizer = normalizer + bos * HQ + i_hq + o_q * HQ
        p_delta = delta + bos * HQ + i_hq + o_q * HQ

        # [BS, BK]
        b_q = tl.load(p_q, mask=m_q[:, None] & (o_d[None, :] < K), other=0.0)
        # [BS, BV]
        b_do = tl.load(p_do, mask=m_q[:, None] & (o_v[None, :] < V), other=0.0)
        # [BS]
        b_normalizer = tl.load(p_normalizer, mask=m_q, other=1.0)
        b_delta = tl.load(p_delta, mask=m_q, other=0.0)
        # [BT, BS]
        b_s = tl.dot(b_k, tl.trans(b_q)) * scale
        if USE_G:
            p_gq = g_cumsum + bos * HQ + i_hq + o_q * HQ
            b_gq = tl.load(p_gq, mask=m_q, other=0.0).to(tl.float32)
            b_s += b_gq[None, :] - b_gk[:, None]
        if USE_WEIGHT_BIAS:
            b_s += b_weight_bias
        if USE_WINDOW:
            m_s = (o_q[None, :] - o_k[:, None] < W) & m_q[None, :]
        else:
            m_s = m_q[None, :]
        b_w = tl.where(m_s, softplus(b_s), 0.0)
        b_p = b_w / b_normalizer[None, :]
        # [BT, BS] @ [BS, BV] -> [BT, BV]
        b_dv = tl.dot(b_p.to(b_do.dtype), b_do, b_dv)
        # [BT, BV] @ [BV, BS] -> [BT, BS]
        b_dp = tl.dot(b_v, tl.trans(b_do))
        # [BT, BS]
        b_dw = tl.where(m_s, tl.sigmoid(b_s), 0.0)
        b_ds = b_dw / b_normalizer[None, :] * (b_dp - b_delta[None, :])
        # [BT, BS] @ [BS, BK] -> [BT, BK]
        b_dk = tl.dot(b_ds.to(b_q.dtype), b_q, b_dk)
        if USE_G:
            b_dg -= tl.sum(b_ds, 1)

    b_dk = b_dk * scale
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_k[:, None] & (o_d[None, :] < K))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_k[:, None] & (o_v[None, :] < V))
    if USE_G:
        p_dg = dg_cumsum + bos * HQ + i_hq + o_k * HQ
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=m_k)


def parallel_flex_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor | None,
    weight_bias: torch.Tensor | None,
    sink_bias: torch.Tensor | None,
    scale: float,
    window_size: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BT = 128
    if check_shared_mem('hopper', q.device.index):
        BS = min(64, max(16, triton.next_power_of_2(T)))
        BK = min(256, max(16, triton.next_power_of_2(K)))
        BV = min(256, max(16, triton.next_power_of_2(V)))
        num_warps = 8
    elif check_shared_mem('ampere', q.device.index):
        BS = min(32, max(16, triton.next_power_of_2(T)))
        BK = min(256, max(16, triton.next_power_of_2(K)))
        BV = min(128, max(16, triton.next_power_of_2(V)))
        num_warps = 4
    else:
        BS = min(32, max(16, triton.next_power_of_2(T)))
        BK = min(256, max(16, triton.next_power_of_2(K)))
        BV = min(64, max(16, triton.next_power_of_2(V)))
        num_warps = 2
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    assert NK == 1, "The key dimension can not be larger than 256"

    o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
    normalizer = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
    grid = (NV, NT, B * HQ)
    parallel_flex_attn_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        o=o,
        g_cumsum=g_cumsum,
        weight_bias=weight_bias,
        sink_bias=sink_bias,
        normalizer=normalizer,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        W=window_size,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
        num_warps=num_warps,
    )
    return o, normalizer


def parallel_flex_attn_bwd_preprocess(
    o: torch.Tensor,
    do: torch.Tensor,
):
    V = o.shape[-1]
    delta = torch.empty_like(o[..., 0], dtype=torch.float)
    parallel_flex_attn_bwd_kernel_preprocess[(delta.numel(),)](
        o=o,
        do=do,
        delta=delta,
        B=triton.next_power_of_2(V),
        V=V,
    )
    return delta


def parallel_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    g_cumsum: torch.Tensor | None,
    weight_bias: torch.Tensor | None,
    normalizer: torch.Tensor,
    do: torch.Tensor,
    sink_bias: torch.Tensor | None = None,
    scale: float = None,
    window_size: int | None = None,
    chunk_size: int = 128,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    # dq/dk are reduced over the full value dim in one program (no cross-program accumulation),
    # so BV must span all of V (NV == 1). Don't cap it here -- the forward can, the backward can't.
    if check_shared_mem('hopper'):
        BT = 128
        BS = 64
        BK = max(triton.next_power_of_2(K), 16)
        BV = max(triton.next_power_of_2(V), 16)
        num_warps = 8
    elif check_shared_mem('ampere'):
        BS = 32
        BK = max(triton.next_power_of_2(K), 16)
        BV = max(triton.next_power_of_2(V), 16)
        BT = 128 if K <= 64 else 64
        num_warps = 4
    else:
        BT = 64
        BS = 32
        BK = max(triton.next_power_of_2(K), 16)
        BV = max(triton.next_power_of_2(V), 16)
        num_warps = 2

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NV = triton.cdiv(V, BV)

    delta = parallel_flex_attn_bwd_preprocess(o, do)

    dq = torch.empty(B, T, HQ, K, dtype=k.dtype if H == HQ else torch.float, device=q.device)
    dk = torch.empty(B, T, HQ, K, dtype=k.dtype if H == HQ else torch.float, device=q.device)
    dv = torch.empty(B, T, HQ, V, dtype=v.dtype if H == HQ else torch.float, device=q.device)
    grid = (NV, NT, B * HQ)

    dg_cumsum, dg_cumsum_k = None, None
    if g_cumsum is not None:
        dg_cumsum = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
        dg_cumsum_k = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
    dweight_bias_rows = (
        torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
        if weight_bias is not None else None
    )

    parallel_flex_attn_bwd_kernel_dq[grid](
        q=q,
        k=k,
        v=v,
        g_cumsum=g_cumsum,
        weight_bias=weight_bias,
        normalizer=normalizer,
        delta=delta,
        do=do,
        dq=dq,
        dg_cumsum=dg_cumsum,
        dweight_bias_rows=dweight_bias_rows,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        W=window_size,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
        num_warps=num_warps,
    )
    parallel_flex_attn_bwd_kernel_dkv[grid](
        q=q,
        k=k,
        v=v,
        g_cumsum=g_cumsum,
        weight_bias=weight_bias,
        normalizer=normalizer,
        delta=delta,
        do=do,
        dk=dk,
        dv=dv,
        dg_cumsum=dg_cumsum_k,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        W=window_size,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BT=BT,
        BS=BS,
        BK=BK,
        BV=BV,
        num_warps=num_warps,
    )
    dk = reduce(dk, 'b t (h g) k -> b t h k', g=G, reduction='sum')
    dv = reduce(dv, 'b t (h g) v -> b t h v', g=G, reduction='sum')
    if g_cumsum is not None:
        dg_cumsum.add_(dg_cumsum_k)

    dweight_bias = dweight_bias_rows.sum((0, 1)) if weight_bias is not None else None
    dsink_bias = None
    if sink_bias is not None:
        sink_logits = sink_bias + weight_bias if weight_bias is not None else sink_bias
        sink_derivative = torch.sigmoid(sink_logits)
        dsink_bias = -(sink_derivative[None, None, :] / normalizer * delta).sum((0, 1))
        if dweight_bias is not None:
            dweight_bias = dweight_bias + dsink_bias

    return dq, dk, dv, dg_cumsum, dweight_bias, dsink_bias


class ParallelAttentionFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, g, weight_bias, sink_bias, scale, window_size, cu_seqlens, chunk_indices=None):
        ctx.dtype = q.dtype

        g_cumsum = chunk_global_cumsum(g, cu_seqlens=cu_seqlens) if g is not None else None
        o, normalizer = parallel_flex_attn_fwd(
            q=q,
            k=k,
            v=v,
            g_cumsum=g_cumsum,
            weight_bias=weight_bias,
            sink_bias=sink_bias,
            scale=scale,
            window_size=window_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        ctx.save_for_backward(q, k, v, o, g_cumsum, weight_bias, sink_bias, normalizer)
        ctx.scale = scale
        ctx.window_size = window_size
        ctx.cu_seqlens = cu_seqlens
        return o.to(q.dtype)

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do):
        q, k, v, o, g_cumsum, weight_bias, sink_bias, normalizer = ctx.saved_tensors
        dq, dk, dv, dg, dweight_bias, dsink_bias = parallel_attn_bwd(
            q=q,
            k=k,
            v=v,
            o=o,
            g_cumsum=g_cumsum,
            weight_bias=weight_bias,
            normalizer=normalizer,
            do=do,
            sink_bias=sink_bias,
            scale=ctx.scale,
            window_size=ctx.window_size,
            cu_seqlens=ctx.cu_seqlens,
        )
        if dg is not None:
            dg = chunk_global_cumsum(dg, cu_seqlens=ctx.cu_seqlens, reverse=True)

        return dq.to(q), dk.to(k), dv.to(v), dg, dweight_bias, dsink_bias, None, None, None, None


def parallel_flex_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    window_size: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    *,
    weight_bias: torch.Tensor | None = None,
    sink_bias: torch.Tensor | None = None,
    **kwargs
) -> torch.Tensor:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, HQ, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
            GQA will be applied if HQ is divisible by H.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (Optional[torch.Tensor]):
            log decay factors of shape `[B, T, HQ]`.
        scale (Optional[float]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        window_size (Optional[int]):
            Sliding window size. If provided, each query at position i only attends to
            keys in `[i - window_size + 1, i]`. If `None`, full causal attention is used.
            Default: `None`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        weight_bias (Optional[torch.Tensor]):
            Per-query-head bias of shape `[HQ]` added to every softplus weight logit.
        sink_bias (Optional[torch.Tensor]):
            Per-query-head attention-sink bias logits of shape `[HQ]` — one
            learnable scalar per query head, as introduced by GPT-OSS.

            Augments the softplus-weight denominator with
            `softplus(sink_bias[h] + weight_bias[h])` without
            adding a corresponding key/value entry, so the model can route
            attention mass to a learnable "no-op" target:
                p_i    = softplus(s_i + weight_bias[h]) / normalizer
                o      = sum_i p_i * v_i   # sink slot contributes no value
            Reserved name: the future
            `sink_tokens_*` kwargs will support Xiao 2024-style K/V sink tokens
            and may be combined with `sink_bias`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HQ, V]`.
    """
    if 'head_first' in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
            f"Please flatten variable-length inputs before processing.",
        )
    if sink_bias is not None:
        assert sink_bias.shape == (q.shape[2],), "sink_bias must have shape [HQ]"
    if weight_bias is not None:
        assert weight_bias.shape == (q.shape[2],), "weight_bias must have shape [HQ]"

    o = ParallelAttentionFunction.apply(
        q, k, v, g, weight_bias, sink_bias, scale, window_size, cu_seqlens, chunk_indices
    )
    return o
