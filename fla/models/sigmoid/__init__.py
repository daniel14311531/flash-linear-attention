# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.sigmoid.configuration_sigmoid import SigmoidConfig
from fla.models.sigmoid.modeling_sigmoid import SigmoidForCausalLM, SigmoidModel

AutoConfig.register(SigmoidConfig.model_type, SigmoidConfig, exist_ok=True)
AutoModel.register(SigmoidConfig, SigmoidModel, exist_ok=True)
AutoModelForCausalLM.register(SigmoidConfig, SigmoidForCausalLM, exist_ok=True)


__all__ = ['SigmoidConfig', 'SigmoidForCausalLM', 'SigmoidModel']
