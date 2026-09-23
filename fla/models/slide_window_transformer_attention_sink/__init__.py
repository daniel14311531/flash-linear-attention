# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.slide_window_transformer_attention_sink.configuration_slide_window_transformer_attention_sink import \
    SlideWindowTransformerAttentionSinkConfig
from fla.models.slide_window_transformer_attention_sink.modeling_slide_window_transformer_attention_sink import (
    SlideWindowTransformerAttentionSinkForCausalLM,
    SlideWindowTransformerAttentionSinkModel,
)

AutoConfig.register(
    SlideWindowTransformerAttentionSinkConfig.model_type,
    SlideWindowTransformerAttentionSinkConfig,
    exist_ok=True,
)
AutoModel.register(SlideWindowTransformerAttentionSinkConfig, SlideWindowTransformerAttentionSinkModel, exist_ok=True)
AutoModelForCausalLM.register(
    SlideWindowTransformerAttentionSinkConfig,
    SlideWindowTransformerAttentionSinkForCausalLM,
    exist_ok=True,
)

__all__ = [
    'SlideWindowTransformerAttentionSinkConfig',
    'SlideWindowTransformerAttentionSinkForCausalLM',
    'SlideWindowTransformerAttentionSinkModel',
]
