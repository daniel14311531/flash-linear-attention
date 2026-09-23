# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.mesa_net_diy.configuration_mesa_net_diy import MesaNetDIYConfig
from fla.models.mesa_net_diy.modeling_mesa_net_diy import MesaNetDIYForCausalLM, MesaNetDIYModel

AutoConfig.register(MesaNetDIYConfig.model_type, MesaNetDIYConfig, exist_ok=True)
AutoModel.register(MesaNetDIYConfig, MesaNetDIYModel, exist_ok=True)
AutoModelForCausalLM.register(MesaNetDIYConfig, MesaNetDIYForCausalLM, exist_ok=True)

__all__ = ['MesaNetDIYConfig', 'MesaNetDIYForCausalLM', 'MesaNetDIYModel']
