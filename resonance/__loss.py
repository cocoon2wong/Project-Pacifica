"""
@Author: Conghao Wong
@Date: 2024-10-10 09:17:14
@LastEditors: Conghao Wong
@LastEditTime: 2024-10-10 09:23:22
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.training.loss import BaseLossLayer
from qpid.training.loss.__ade import ADE_2D


class InterventionLoss(BaseLossLayer):

    def forward(self, outputs: list[torch.Tensor], 
                labels: list[torch.Tensor], 
                inputs: list[torch.Tensor], 
                mask=None, training=None, *args, **kwargs):
        
        y_interaction_c = outputs[-1]
        return ADE_2D(y_interaction_c, torch.zeros_like(y_interaction_c))
