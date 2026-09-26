# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch

from fla.models import SigmoidConfig

from .test_modeling_base import run_test_generation, run_test_model_forward_backward


@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'use_l2warp', 'use_weight_bias', 'dtype'],
    [
        pytest.param(*test, id="L{}-B{}-T{}-H{}-D{}-l2{}-wb{}-{}".format(*test))
        for test in [
            (4, 4, 1024, 4, 64,  True,  True,  torch.bfloat16),
            (4, 4, 1024, 4, 64,  False, False, torch.bfloat16),
            (4, 4, 1024, 4, 128, False, True,  torch.bfloat16),
        ]
    ],
)
def test_modeling(L, B, T, H, D, use_l2warp, use_weight_bias, dtype):
    run_test_model_forward_backward(
        L,
        B,
        T,
        H,
        D,
        SigmoidConfig,
        use_l2warp=use_l2warp,
        use_weight_bias=use_weight_bias,
        dtype=dtype,
    )


@pytest.mark.parametrize(
    ['L', 'B', 'T', 'H', 'D', 'dtype'],
    [pytest.param(2, 4, 2000, 8, 64, torch.float16, id='L2-B4-T2000-H8-D64-float16')],
)
def test_generation(L, B, T, H, D, dtype):
    run_test_generation(L, B, T, H, D, SigmoidConfig, dtype)
