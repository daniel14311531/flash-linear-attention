# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.softplus.configuration_softplus import SoftplusConfig
from fla.models.softplus.modeling_softplus import SoftplusForCausalLM, SoftplusModel

AutoConfig.register(SoftplusConfig.model_type, SoftplusConfig, exist_ok=True)
AutoModel.register(SoftplusConfig, SoftplusModel, exist_ok=True)
AutoModelForCausalLM.register(SoftplusConfig, SoftplusForCausalLM, exist_ok=True)


__all__ = ['SoftplusConfig', 'SoftplusForCausalLM', 'SoftplusModel']
