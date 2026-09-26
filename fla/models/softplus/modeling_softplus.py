# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch.nn as nn

from fla.layers import SoftplusAttention
from fla.models.softplus.configuration_softplus import SoftplusConfig
from fla.models.transformer.modeling_transformer import (
    TransformerBlock,
    TransformerForCausalLM,
    TransformerModel,
    TransformerPreTrainedModel,
)


class SoftplusBlock(TransformerBlock):

    @staticmethod
    def _build_attention(config: SoftplusConfig, layer_idx: int) -> nn.Module:
        return SoftplusAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            qkv_bias=config.qkv_bias,
            qk_norm=config.qk_norm,
            window_size=config.window_size,
            use_weight_bias=config.use_weight_bias,
            weight_bias_init=config.weight_bias_init,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            layer_idx=layer_idx,
        )


class SoftplusPreTrainedModel(TransformerPreTrainedModel):

    config_class = SoftplusConfig
    _no_split_modules = ['SoftplusBlock']


class SoftplusModel(SoftplusPreTrainedModel, TransformerModel):

    _block_class = SoftplusBlock


class SoftplusForCausalLM(SoftplusPreTrainedModel, TransformerForCausalLM):

    _model_class = SoftplusModel
