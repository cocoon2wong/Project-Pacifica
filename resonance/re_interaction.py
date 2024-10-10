"""
@Author: Conghao Wong
@Date: 2024-10-09 20:28:02
@LastEditors: Conghao Wong
@LastEditTime: 2024-10-09 21:37:45
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import numpy as np
import torch

from qpid.args import Args
from qpid.model import layers, transformer
from qpid.model.layers.transfroms import _BaseTransformLayer
from qpid.utils import get_mask

from .__args import ResonanceArgs


class InteractionModel(torch.nn.Module):
    def __init__(self, Args: Args,
                 T_nei_obs: _BaseTransformLayer,
                 iT_nei_pred: _BaseTransformLayer,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.args = Args
        self.re_args = self.args.register_subargs(ResonanceArgs, 're_args')

        # Layers
        # Transform layers (to compute interaction)
        self.T_layer = T_nei_obs
        self.iT_layer = iT_nei_pred

        self.d = self.args.feature_dim
        self.d_id = self.args.noise_depth

        # Shapes
        self.Trsteps_en, self.Trchannels_en = self.T_layer.Tshape
        self.Trsteps_de, self.Trchannels_de = self.iT_layer.Tshape
        self.max_steps = max(self.Trsteps_en, self.re_args.partitions)

        # Noise encoding (fine level)
        self.re_ie = layers.TrajEncoding(self.d_id, self.d//2, torch.nn.Tanh)
        self.concat_fc = layers.Dense(
            self.d + self.d//2, self.d//2, torch.nn.Tanh)

        # Transformer is used as a feature extractor (fine level)
        self.re_T = transformer.Transformer(
            num_layers=4,
            d_model=self.d,
            num_heads=8,
            dff=512,
            input_vocab_size=self.Trchannels_en,
            target_vocab_size=self.Trchannels_de,
            pe_input=self.max_steps,
            pe_target=self.max_steps,
            include_top=False
        )

        # Trainable adj matrix and gcn layer (fine level)
        self.re_ms_fc = layers.Dense(self.d, self.re_args.Kc, torch.nn.Tanh)
        self.re_ms_conv = layers.GraphConv(self.d, self.d)

        # Decoder layers (fine level)
        self.re_decoder_fc1 = layers.Dense(self.d, self.d, torch.nn.Tanh)
        self.re_decoder_fc2 = layers.Dense(self.d,
                                           self.Trsteps_de * self.Trchannels_de)
        
    def forward(self, ego_traj: torch.Tensor,
                f_ego: torch.Tensor,
                f_resonance: torch.Tensor,
                training=None, mask=None, *args, **kwargs):
        
        
        # Concat resonance features with trajectory features
        f_ego_pad = pad(f_ego, self.max_steps)
        f_re_pad = pad(f_resonance, self.max_steps)

        f_behavior = torch.concat([f_ego_pad, f_re_pad], dim=-1)
        f_behavior = self.concat_fc(f_behavior)

        re_predictions = []
        repeats = self.args.K_train if training else self.args.K
        traj_targets = self.T_layer(ego_traj)

        for _ in range(repeats):
            # Assign random ids and embedding -> (batch, steps, d)
            z = torch.normal(mean=0, std=1,
                             size=list(f_behavior.shape[:-1]) + [self.d_id])
            re_f_z = self.re_ie(z.to(ego_traj.device))

            # (batch, steps, 2*d)
            re_f_final = torch.concat([f_behavior, re_f_z], dim=-1)

            # Transformer outputs' shape is (batch, steps, d)
            f_tran, _ = self.re_T(inputs=re_f_final,
                                  targets=traj_targets,
                                  training=training)

            # Multiple generations -> (batch, Kc, d)
            re_adj = self.re_ms_fc(re_f_final)               # (batch, steps, Kc)
            re_adj = torch.transpose(re_adj, -1, -2)
            re_f_multi = self.re_ms_conv(f_tran, re_adj)     # (batch, Kc, d)

            # Forecast keypoints -> (..., Kc, Tsteps_Key, Tchannels)
            re_y = self.re_decoder_fc1(re_f_multi)
            re_y = self.re_decoder_fc2(re_y)
            re_y = torch.reshape(re_y, list(re_y.shape[:-1]) +
                                 [self.Trsteps_de, self.Trchannels_de])

            re_y = self.iT_layer(re_y)
            re_predictions.append(re_y)

        # (batch, K, n_key, dim)
        y_interaction = torch.concat(re_predictions, dim=-3)
        return y_interaction


def pad(input: torch.Tensor, max_steps: int):
    """
    Zero-padding the input tensor (whose shape must be `(batch, steps, dim)`).
    It will pad the input tensor on the `steps` axis if `steps < max_steps`.
    """
    steps = input.shape[-2]
    if steps < max_steps:
        paddings = [0, 0, 0, max_steps - steps, 0, 0]
        return torch.nn.functional.pad(input, paddings)
    else:
        return input
    