# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch.nn as nn

from fla.layers import SigmoidAttention
from fla.models.sigmoid.configuration_sigmoid import SigmoidConfig
from fla.models.transformer.modeling_transformer import (
    TransformerBlock,
    TransformerForCausalLM,
    TransformerModel,
    TransformerPreTrainedModel,
)


class SigmoidBlock(TransformerBlock):

    @staticmethod
    def _build_attention(config: SigmoidConfig, layer_idx: int) -> nn.Module:
        return SigmoidAttention(
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


class SigmoidPreTrainedModel(TransformerPreTrainedModel):

    config_class = SigmoidConfig
    _no_split_modules = ['SigmoidBlock']


class SigmoidModel(SigmoidPreTrainedModel, TransformerModel):

    _block_class = SigmoidBlock


class SigmoidForCausalLM(SigmoidPreTrainedModel, TransformerForCausalLM):

    _model_class = SigmoidModel
