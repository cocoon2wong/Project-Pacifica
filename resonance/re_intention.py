"""
@Author: Conghao Wong
@Date: 2024-10-09 20:26:00
@LastEditors: Conghao Wong
@LastEditTime: 2024-10-09 21:38:59
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import torch
from .__args import ResonanceArgs

from qpid.args import Args
from qpid.model import layers, transformer
from qpid.model.layers.interpolation import (LinearPositionInterpolation,
                                             LinearSpeedInterpolation)
from qpid.model.layers.transfroms import _BaseTransformLayer


class IntentionModel(torch.nn.Module):
    def __init__(self, Args: Args,
                 T_ego_obs: _BaseTransformLayer,
                 iT_ego_pred: _BaseTransformLayer,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.args = Args
        self.re_args = self.args.register_subargs(ResonanceArgs, 're_args')

        # Layers
        # Transform layers (ego)
        self.t1 = T_ego_obs
        self.it1 = iT_ego_pred
        
        self.d = self.args.feature_dim
        self.d_id = self.args.noise_depth

        # Trajectory encoding (ego)
        self.te = layers.TrajEncoding(self.t1.Oshape[-1], self.d//2,
                                      torch.nn.ReLU,
                                      transform_layer=self.t1)

        # Shapes
        self.Tsteps_en, self.Tchannels_en = self.t1.Tshape
        self.Tsteps_de, self.Tchannels_de = self.it1.Tshape

        # Bilinear structure (outer product + pooling + fc)
        # For trajectories
        self.outer = layers.OuterLayer(self.d//2, self.d//2)
        self.pooling = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten(axes_num=2)
        self.outer_fc = layers.Dense((self.d//4)**2, self.d//2, torch.nn.Tanh)

        # Noise encoding
        self.ie = layers.TrajEncoding(self.d_id, self.d//2, torch.nn.Tanh)

        # Transformer is used as a feature extractor
        self.T = transformer.Transformer(
            num_layers=4,
            d_model=self.d,
            num_heads=8,
            dff=512,
            input_vocab_size=self.Tchannels_en,
            target_vocab_size=self.Tchannels_de,
            pe_input=self.Tsteps_en,
            pe_target=self.Tsteps_en,
            include_top=False
        )

        # Trainable adj matrix and gcn layer
        # See our previous work "MSN: Multi-Style Network for Trajectory Prediction" for detail
        # It is used to generate multiple predictions within one model implementation
        self.ms_fc = layers.Dense(self.d, self.re_args.Kc, torch.nn.Tanh)
        self.ms_conv = layers.GraphConv(self.d, self.d)

        # Decoder layers
        self.decoder_fc1 = layers.Dense(self.d, self.d, torch.nn.Tanh)
        self.decoder_fc2 = layers.Dense(self.d,
                                        self.Tsteps_de * self.Tchannels_de)

        # Interpolation layers
        match self.re_args.interp:
            case 'linear':
                i_layer_type = LinearPositionInterpolation
            case 'speed':
                i_layer_type = LinearSpeedInterpolation
            case _:
                raise ValueError

        self.interp_layer = i_layer_type()

    def forward(self, ego_traj: torch.Tensor,
                output_pred_steps: torch.Tensor,
                training=None, mask=None, *args, **kwargs):

        # Trajectory embedding and encoding
        f = self.te(ego_traj)
        f = self.outer(f, f)
        f = self.pooling(f)
        f = self.flatten(f)
        f_traj = self.outer_fc(f)       # (batch, steps, d/2)

        # Sampling random noise vectors
        all_predictions = []
        repeats = self.args.K_train if training else self.args.K
        traj_targets = self.t1(ego_traj)

        for _ in range(repeats):
            # Assign random ids and embedding -> (batch, steps, d)
            z = torch.normal(mean=0, std=1,
                             size=list(f_traj.shape[:-1]) + [self.d_id])
            f_z = self.ie(z.to(ego_traj.device))

            # (batch, steps, 2*d)
            f_final = torch.concat([f_traj, f_z], dim=-1)

            # Transformer outputs' shape is (batch, steps, d)
            f_tran, _ = self.T(inputs=f_final,
                               targets=traj_targets,
                               training=training)

            # Multiple generations -> (batch, Kc, d)
            adj = self.ms_fc(f_final)               # (batch, steps, Kc)
            adj = torch.transpose(adj, -1, -2)
            f_multi = self.ms_conv(f_tran, adj)     # (batch, Kc, d)

            # Forecast keypoints -> (..., Kc, Tsteps_Key, Tchannels)
            y = self.decoder_fc1(f_multi)
            y = self.decoder_fc2(y)
            y = torch.reshape(y, list(y.shape[:-1]) +
                              [self.Tsteps_de, self.Tchannels_de])

            y = self.it1(y)
            all_predictions.append(y)

        # (batch, K, n_key, dim)
        y_clean = torch.concat(all_predictions, dim=-3)
        y_clean_interp = self.interp(output_pred_steps, y_clean, ego_traj)
        return y_clean_interp, f_traj

    def interp(self, index: torch.Tensor,
               value: torch.Tensor,
               obs_traj: torch.Tensor) -> torch.Tensor:
        """
        Interpolate trajectories according to the predicted keypoints.

        :param index: Indices of predictions, which only includs future points.
        :param value: Predicted future keypoints. It has the same length as the \
            above `index`.
        :param obs_traj: Observed trajectories.
        """
        _i = torch.concat([torch.tensor([-1]).to(index.device), index])
        _obs = torch.repeat_interleave(obs_traj[..., None, -1:, :],
                                       repeats=value.shape[-3],
                                       dim=-3) \
            if value.ndim != obs_traj.ndim else obs_traj[..., -1:, :]
        _v = torch.concat([_obs, value], dim=-2)

        if self.re_args.interp == 'linear':
            return self.interp_layer(_i, _v)
        elif self.re_args.interp == 'speed':
            v0 = obs_traj[..., -1:, :] - obs_traj[..., -2:-1, :]
            v0 = v0[..., None, :, :] if v0.ndim != value.ndim else v0
            return self.interp_layer(_i, _v, init_speed=v0)
        else:
            raise ValueError
