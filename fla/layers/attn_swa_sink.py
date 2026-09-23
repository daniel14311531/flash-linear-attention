# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from einops import rearrange

from fla.layers.utils import pad_input, unpad_input
from fla.modules import RMSNorm, RotaryEmbedding
from fla.ops.attn.decoding import attn_decoding_one_step
from fla.ops.attn.parallel import parallel_attn
from fla.ops.utils.index import prepare_lens_from_mask

if TYPE_CHECKING:
    from fla.models.utils import Cache


class AttentionSWASink(nn.Module):
    """Multi-head attention with a sliding window and a learnable attention sink.

    Unlike `fla.layers.attn.Attention` (backed by the `flash_attn` package), this layer
    routes compute through FLA's Triton kernels (`fla.ops.attn`), which natively support
    GPT-OSS style per-query-head sink bias logits added to the softmax denominator:
        p_i = exp(s_i) / (sum_j exp(s_j) + exp(sink_bias[h]))
    The sink slot absorbs probability mass but contributes no value, letting the model
    dump attention mass outside the sliding window.

    Args:
        hidden_size (int, Optional):
            Hidden dimension of the model. Default: 2048.
        num_heads (int, Optional):
            Number of query heads. Default: 32.
        num_kv_heads (int, Optional):
            Number of key/value heads (GQA if fewer than `num_heads`). Default: None.
        qkv_bias (bool, Optional):
            Whether to use biases for q/k/v projections. Default: False.
        qk_norm (bool, Optional):
            Whether to apply RMSNorm to q/k after projection. Default: False.
        window_size (int, Optional):
            Sliding window size; each query at position i attends to keys in
            [i - window_size + 1, i]. None means full causal attention. Default: None.
        attention_sink (bool, Optional):
            Whether to attach a learnable per-head sink bias. Default: True.
        sink_init (float, Optional):
            Initial value of the sink bias logits. Default: 0.0.
        rope_theta (float, Optional):
            Base of RoPE. Default: 10000.
        max_position_embeddings (int, Optional):
            Maximum supported position. Default: None.
        layer_idx (int, Optional):
            Layer index for cache handling. Default: None.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: int | None = None,
        attention_sink: bool = True,
        sink_init: float = 0.0,
        rope_theta: float | None = 10000.,
        max_position_embeddings: int | None = None,
        layer_idx: int = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm

        self.window_size = window_size
        self.attention_sink = attention_sink
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx

        if attention_sink:
            # [HQ] sink bias logits, one learnable scalar per query head
            self.sink_bias = nn.Parameter(torch.full((self.num_heads,), sink_init, dtype=torch.float32))
        else:
            self.sink_bias = None

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
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.size()

        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        # equivalent to cu_seqlens in `flash_attn`
        cu_seqlens = kwargs.get('cu_seqlens')

        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                # to eliminate the offsets of padding tokens
                seqlen_offset = seqlen_offset + prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]
                max_seqlen = q.shape[1] + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

        cache_has_content = False
        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            k_cached, v_cached = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size),
            )['attn_state']
            if cache_has_content:
                k, v = k_cached, v_cached
                k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
                v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        sink_bias = self.sink_bias

        if cu_seqlens is not None:
            # packed varlen training: parallel kernel shares one timeline for q and k
            if cache_has_content:
                raise NotImplementedError(
                    "AttentionSWASink does not support `cu_seqlens` together with a non-empty cache."
                )
            o = parallel_attn(
                rearrange(q, 'b l ... -> 1 (b l) ...').contiguous(),
                rearrange(k, 'b l ... -> 1 (b l) ...').contiguous(),
                rearrange(v, 'b l ... -> 1 (b l) ...').contiguous(),
                cu_seqlens=cu_seqlens,
                window_size=self.window_size,
                sink_bias=sink_bias,
            )
            o = rearrange(o, '1 (b l) h d -> b l h d', b=batch_size, l=q_len)
        elif q_len == 1:
            # single-token incremental decoding through the dedicated kernel
            if attention_mask is not None:
                if self.window_size is not None:
                    attention_mask = attention_mask[:, -self.window_size:]
                q, (k, v), indices_q, cu_seqlens, max_seq_lens = unpad_input(q, (k, v), attention_mask, q_len)
                cu_seqlens_q, cu_seqlens_k = cu_seqlens
                o = attn_decoding_one_step(
                    q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                    cu_seqlens=cu_seqlens_k,
                    sink_bias=sink_bias,
                ).squeeze(0)
                o = pad_input(o, indices_q, batch_size, q_len)
            else:
                # flatten [B, T, ...] -> [1, B*T, ...] in sequence-major order to match cu_seqlens
                T = k.shape[1]
                cu = torch.arange(0, batch_size * T + 1, T, dtype=torch.long, device=q.device)
                o = attn_decoding_one_step(
                    rearrange(q, 'b l ... -> l b ...').contiguous(),
                    k.flatten(0, 1).unsqueeze(0).contiguous(),
                    v.flatten(0, 1).unsqueeze(0).contiguous(),
                    cu_seqlens=cu,
                    sink_bias=sink_bias,
                )
                o = rearrange(o, 'l b ... -> b l ...')
        elif attention_mask is not None and not cache_has_content:
            # packed varlen prefill with padding
            q, (k, v), indices_q, cu_seqlens, max_seq_lens = unpad_input(q, (k, v), attention_mask, q_len)
            cu_seqlens_q, cu_seqlens_k = cu_seqlens
            o = parallel_attn(
                q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                cu_seqlens=cu_seqlens_q,
                window_size=self.window_size,
                sink_bias=sink_bias,
            ).squeeze(0)
            o = pad_input(o, indices_q, batch_size, q_len)
        else:
            # dense path; with a non-empty cache the parallel kernel needs q and k on the
            # same timeline, so pad the query with dummy prefix positions (outputs discarded)
            if cache_has_content:
                cached_len = k.shape[1] - q_len
                q = torch.cat([q.new_zeros(*q.shape[:1], cached_len, *q.shape[2:]), q], dim=1)
                o = parallel_attn(q, k, v, window_size=self.window_size, sink_bias=sink_bias)
                o = o[:, cached_len:]
            else:
                o = parallel_attn(q, k, v, window_size=self.window_size, sink_bias=sink_bias)

        o = o.reshape(batch_size, q_len, -1)
        o = self.o_proj(o)

        if not output_attentions:
            attentions = None

        return o, attentions, past_key_values
