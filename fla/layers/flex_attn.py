# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange

from fla.layers.utils import pad_input, unpad_input
from fla.modules import RMSNorm, RotaryEmbedding
from fla.ops.utils.index import prepare_lens_from_mask

if TYPE_CHECKING:
    from fla.models.utils import Cache


class _FlexAttention(nn.Module):
    """Multi-head attention with a normalized positive attention weight function.

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        num_heads (int, Optional):
            The number of query heads. Default: 32.
        num_kv_heads (int, Optional):
            The number of key/value heads, equal to `num_heads` if `None`. Default: `None`.
        qkv_bias (bool, Optional):
            Whether to use bias in the query, key, and value projections. Default: `False`.
        qk_norm (bool, Optional):
            Whether to apply RMSNorm to queries and keys. Default: `False`.
        window_size (int, Optional):
            The sliding-window size, or `None` for full causal attention. Default: `None`.
        use_weight_bias (bool, Optional):
            Whether to learn one attention-weight logit bias per query head. Default: `True`.
        weight_bias_init (float, Optional):
            The initial value of each attention-weight logit bias. Default: 0.0.
        rope_theta (float, Optional):
            The RoPE base. Default: 10000.0.
        max_position_embeddings (int, Optional):
            The maximum supported position, or `None` to infer it from the input. Default: `None`.
        layer_idx (int, Optional):
            The layer index used to address the KV cache. Default: `None`.
        parallel_attention (Callable):
            The parallel attention operator used for training and prefill.
        decoding_attention (Callable):
            The single-token attention operator used for cached decoding.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: int | None = None,
        use_weight_bias: bool = True,
        weight_bias_init: float = 0.0,
        rope_theta: float | None = 10000.,
        max_position_embeddings: int | None = None,
        layer_idx: int | None = None,
        *,
        parallel_attention: Callable,
        decoding_attention: Callable,
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size must be divisible by num_heads, got {hidden_size} and {num_heads}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads must be divisible by num_kv_heads, got {self.num_heads} and {self.num_kv_heads}"
            )
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.window_size = window_size
        self.use_weight_bias = use_weight_bias
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx
        self.parallel_attention = parallel_attention
        self.decoding_attention = decoding_attention

        if use_weight_bias:
            self.weight_bias = nn.Parameter(torch.full((self.num_heads,), weight_bias_init, dtype=torch.float32))
        else:
            self.register_parameter('weight_bias', None)

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, dtype=torch.float32)
            self.k_norm = RMSNorm(self.head_dim, dtype=torch.float32)

        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError(
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] for padding purposes."
            )

        batch_size, q_len, _ = hidden_states.shape
        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        cu_seqlens = kwargs.get('cu_seqlens')
        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q_len + seqlen_offset
            if attention_mask is not None:
                seqlen_offset = seqlen_offset + prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]
                max_seqlen = q_len + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

        cache_has_content = False
        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            if cu_seqlens is not None and cache_has_content:
                raise NotImplementedError("cu_seqlens cannot be combined with a non-empty KV cache")
            k_cached, v_cached = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size),
            )['attn_state']
            if cache_has_content:
                k = rearrange(k_cached, '... (h d) -> ... h d', d=self.head_dim)
                v = rearrange(v_cached, '... (h d) -> ... h d', d=self.head_dim)

        if cu_seqlens is not None:
            o = self.parallel_attention(
                rearrange(q, 'b l ... -> 1 (b l) ...').contiguous(),
                rearrange(k, 'b l ... -> 1 (b l) ...').contiguous(),
                rearrange(v, 'b l ... -> 1 (b l) ...').contiguous(),
                cu_seqlens=cu_seqlens,
                window_size=self.window_size,
                weight_bias=self.weight_bias,
            )
            o = rearrange(o, '1 (b l) h d -> b l h d', b=batch_size, l=q_len)
        elif q_len == 1:
            if attention_mask is not None:
                if self.window_size is not None:
                    attention_mask = attention_mask[:, -self.window_size:]
                q, (k, v), indices_q, cu_seqlens, _ = unpad_input(q, (k, v), attention_mask, q_len)
                _, cu_seqlens_k = cu_seqlens
                o = self.decoding_attention(
                    q.unsqueeze(0),
                    k.unsqueeze(0),
                    v.unsqueeze(0),
                    cu_seqlens=cu_seqlens_k,
                    weight_bias=self.weight_bias,
                ).squeeze(0)
                o = pad_input(o, indices_q, batch_size, q_len)
            else:
                kv_len = k.shape[1]
                cu_seqlens = torch.arange(0, batch_size * kv_len + 1, kv_len, dtype=torch.long, device=q.device)
                o = self.decoding_attention(
                    rearrange(q, 'b l ... -> l b ...').contiguous(),
                    k.flatten(0, 1).unsqueeze(0).contiguous(),
                    v.flatten(0, 1).unsqueeze(0).contiguous(),
                    cu_seqlens=cu_seqlens,
                    weight_bias=self.weight_bias,
                )
                o = rearrange(o, 'l b ... -> b l ...')
        elif attention_mask is not None and not cache_has_content:
            q, (k, v), indices_q, cu_seqlens, _ = unpad_input(q, (k, v), attention_mask, q_len)
            cu_seqlens_q, _ = cu_seqlens
            o = self.parallel_attention(
                q.unsqueeze(0),
                k.unsqueeze(0),
                v.unsqueeze(0),
                cu_seqlens=cu_seqlens_q,
                window_size=self.window_size,
                weight_bias=self.weight_bias,
            ).squeeze(0)
            o = pad_input(o, indices_q, batch_size, q_len)
        else:
            if cache_has_content:
                cached_len = k.shape[1] - q_len
                q = torch.cat([q.new_zeros(batch_size, cached_len, self.num_heads, self.head_dim), q], dim=1)
                o = self.parallel_attention(
                    q,
                    k,
                    v,
                    window_size=self.window_size,
                    weight_bias=self.weight_bias,
                )[:, cached_len:]
            else:
                o = self.parallel_attention(
                    q,
                    k,
                    v,
                    window_size=self.window_size,
                    weight_bias=self.weight_bias,
                )

        o = self.o_proj(o.reshape(batch_size, q_len, -1))
        return o, None, past_key_values
