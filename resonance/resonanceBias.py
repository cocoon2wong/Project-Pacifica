"""
@Author: Conghao Wong
@Date: 2024-10-09 20:28:02
@LastEditors: Conghao Wong
@LastEditTime: 2024-10-11 11:12:31
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.args import Args
from qpid.model import layers, transformer
from qpid.model.layers.transfroms import _BaseTransformLayer

from .__args import ResonanceArgs


class ResonanceBias(torch.nn.Module):
    """
    Resonance-Bias Layer
    ---
    Predict the resonance-bias trajectory of each ego agent by considering
    its interactions with all neighbors (which is described as the angle-based
    resonance feature).
    """

    def __init__(self, Args: Args,
                 output_units: int,
                 noise_units: int,
                 ego_feature_dim: int,
                 re_feature_dim: int,
                 T_nei_obs: _BaseTransformLayer,
                 iT_nei_pred: _BaseTransformLayer,
                 *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.args = Args
        self.re_args = self.args.register_subargs(ResonanceArgs, 're_args')

        self.d = self.args.feature_dim
        self.d_id = self.args.noise_depth
        self.d_z = noise_units

        # Layers
        # Transform layers (to compute interaction)
        self.T_layer = T_nei_obs
        self.iT_layer = iT_nei_pred

        # Shapes
        self.Trsteps_en, self.Trchannels_en = self.T_layer.Tshape
        self.Trsteps_de, self.Trchannels_de = self.iT_layer.Tshape
        self.max_steps = max(self.Trsteps_en, self.re_args.partitions)

        # Noise encoding (fine level)
        self.ie = layers.TrajEncoding(self.d_id, noise_units, torch.nn.Tanh)
        self.concat_fc = layers.Dense(ego_feature_dim + re_feature_dim,
                                      output_units - noise_units,
                                      activation=torch.nn.Tanh)

        # Transformer is used as a feature extractor (fine level)
        self.re_T = transformer.Transformer(
            num_layers=2,
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

    def forward(self, ego_traj_diff: torch.Tensor,
                f_ego_diff: torch.Tensor,
                f_resonance: torch.Tensor,
                training=None, mask=None, *args, **kwargs):

        # Pad features to keep the compatible tensor shape
        f_ego_pad = pad(f_ego_diff, self.max_steps)
        f_re_pad = pad(f_resonance, self.max_steps)

        # Concat and fuse resonance features with trajectory features
        f_behavior = torch.concat([f_ego_pad, f_re_pad], dim=-1)
        f_behavior = self.concat_fc(f_behavior)

        re_predictions = []
        repeats = self.args.K_train if training else self.args.K
        traj_targets = self.T_layer(ego_traj_diff)
        traj_targets = pad(traj_targets, self.max_steps)

        for _ in range(repeats):
            # Assign random ids and embedding -> (batch, steps, d)
            z = torch.normal(mean=0, std=1,
                             size=list(f_behavior.shape[:-1]) + [self.d_id])
            re_f_z = self.ie(z.to(ego_traj_diff.device))

            # (batch, steps, 2*d)
            re_f_final = torch.concat([f_behavior, re_f_z], dim=-1)

            # Transformer outputs' shape is (batch, steps, d)
            f_tran, _ = self.re_T(inputs=re_f_final,
                                  targets=traj_targets,
                                  training=training)

            # Multiple generations -> (batch, Kc, d)
            # (batch, steps, Kc)
            re_adj = self.re_ms_fc(re_f_final)
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
        y_re_bias = torch.concat(re_predictions, dim=-3)
        return y_re_bias


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
