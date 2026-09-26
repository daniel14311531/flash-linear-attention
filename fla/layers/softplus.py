# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.layers.flex_attn import _FlexAttention
from fla.ops.flex_attn.softplus.decoding import flex_attn_decoding_one_step
from fla.ops.flex_attn.softplus.parallel import parallel_flex_attn


class SoftplusAttention(_FlexAttention):
    """Causal attention with normalized softplus weights."""

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            parallel_attention=parallel_flex_attn,
            decoding_attention=flex_attn_decoding_one_step,
            **kwargs,
        )
