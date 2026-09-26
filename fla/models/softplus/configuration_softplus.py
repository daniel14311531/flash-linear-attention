# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.models.transformer.configuration_transformer import TransformerConfig


class SoftplusConfig(TransformerConfig):

    model_type = 'softplus'

    def __init__(
        self,
        use_weight_bias: bool = True,
        weight_bias_init: float = 0.0,
        **kwargs,
    ):
        self.use_weight_bias = use_weight_bias
        self.weight_bias_init = weight_bias_init
        super().__init__(**kwargs)
