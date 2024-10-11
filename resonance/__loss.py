"""
@Author: Conghao Wong
@Date: 2024-10-10 09:17:14
@LastEditors: Conghao Wong
@LastEditTime: 2024-10-11 11:15:46
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.training.loss import BaseLossLayer


class InterventionLoss(BaseLossLayer):

    def forward(self, outputs: list[torch.Tensor],
                labels: list[torch.Tensor],
                inputs: list[torch.Tensor],
                mask=None, training=None, *args, **kwargs):

        y_interaction_c = outputs[-1]
        return torch.max((y_interaction_c)**2)
