"""
@Author: Conghao Wong
@Date: 2024-10-08 19:18:40
@LastEditors: Conghao Wong
@LastEditTime: 2024-10-09 18:17:44
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import numpy as np
import torch

from qpid.constant import INPUT_TYPES
from qpid.model import Model, layers, transformer
from qpid.model.layers.interpolation import (LinearPositionInterpolation,
                                             LinearSpeedInterpolation)
from qpid.training import Structure
from qpid.utils import get_mask

from .__args import ResonanceArgs


class ResonanceModel(Model):

    def __init__(self, structure=None, *args, **kwargs):
        super().__init__(structure, *args, **kwargs)

        # Init args
        self.args._set_default('K', 1)
        self.args._set_default('K_train', 1)
        self.re_args = self.args.register_subargs(ResonanceArgs, 're_args')

        # Set model inputs
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                        INPUT_TYPES.NEIGHBOR_TRAJ)

        ########################
        # The Intention Branch #
        ########################
        # Layers
        # Interpolation layers
        match self.re_args.interp:
            case 'linear':
                i_layer_type = LinearPositionInterpolation
            case 'speed':
                i_layer_type = LinearSpeedInterpolation
            case _:
                raise ValueError

        self.interp_layer = i_layer_type()

        # Transform layers (ego)
        tlayer, itlayer = layers.get_transform_layers(self.re_args.T)
        self.t1 = tlayer((self.args.obs_frames, self.dim))
        self.it1 = itlayer((len(self.output_pred_steps), self.dim))

        # Trajectory encoding (ego)
        self.te = layers.TrajEncoding(self.dim, self.d//2,
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

        ##########################
        # The Interaction Branch #
        ##########################
        # Layers
        # Transform layers (to compute interaction)
        trlayer, itrlayer = layers.get_transform_layers(self.re_args.Tr)
        self.tr1 = trlayer((self.args.obs_frames, self.dim))
        self.itr1 = itrlayer((self.args.pred_frames, self.dim))

        # Trajectory encoding (neighbors)
        self.tre = layers.TrajEncoding(self.dim, self.d//2,
                                       torch.nn.ReLU,
                                       transform_layer=self.tr1)

        # Circle encoding
        self.ce = layers.TrajEncoding(2, self.d//2, torch.nn.ReLU)

        # Shapes
        self.Trsteps_en, self.Trchannels_en = self.tr1.Tshape
        self.Trsteps_de, self.Trchannels_de = self.itr1.Tshape
        self.Trsteps_en = max(self.Trsteps_en, self.re_args.partitions)

        # Encoding resonance features
        self.re_fc1 = layers.Dense(
            self.d*self.Trsteps_en, self.d*2, torch.nn.ReLU)
        self.re_fc2 = layers.Dense(self.d*2, self.d*2, torch.nn.ReLU)
        self.re_fc3 = layers.Dense(self.d*2, self.d//2, torch.nn.ReLU)

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
            pe_input=self.Trsteps_en,
            pe_target=self.Trsteps_en,
            include_top=False
        )

        # Trainable adj matrix and gcn layer (fine level)
        self.re_ms_fc = layers.Dense(self.d, self.re_args.Kc, torch.nn.Tanh)
        self.re_ms_conv = layers.GraphConv(self.d, self.d)

        # Decoder layers (fine level)
        self.re_decoder_fc1 = layers.Dense(self.d, self.d, torch.nn.Tanh)
        self.re_decoder_fc2 = layers.Dense(self.d,
                                           self.Trsteps_de * self.Trchannels_de)

    def forward(self, inputs: list[torch.Tensor], training=None, mask=None, *args, **kwargs):
        # Unpack inputs
        # (batch, obs, dim)
        obs = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)

        # (batch, N, obs, dim)
        nei = self.get_input(inputs, INPUT_TYPES.NEIGHBOR_TRAJ)

        ########################
        # The Intention Branch #
        ########################

        # Trajectory embedding and encoding
        f = self.te(obs)
        f = self.outer(f, f)
        f = self.pooling(f)
        f = self.flatten(f)
        f_traj = self.outer_fc(f)       # (batch, steps, d/2)

        # Sampling random noise vectors
        all_predictions = []
        repeats = self.args.K_train if training else self.args.K

        traj_targets = self.t1(obs)

        for _ in range(repeats):
            # Assign random ids and embedding -> (batch, steps, d)
            z = torch.normal(mean=0, std=1,
                             size=list(f_traj.shape[:-1]) + [self.d_id])
            f_z = self.ie(z.to(obs.device))

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
        y_clean_interp = self.interp(self.output_pred_steps, y_clean, obs)

        ##########################
        # The Interaction Branch #
        ##########################

        # Move the last point of trajectories to 0
        nei_r = nei - nei[..., -1:, :]
        obs_r = (obs - obs[..., -1:, :])[..., None, :, :]

        pack_r = torch.concat([nei_r, obs_r], dim=-3)
        f_pack = self.tre(pack_r)

        f_re_obs = f_pack[..., :1, :, :]
        f_re_nei = f_pack[..., 1:, :, :]

        # Compute resonance features (for each neighbor)
        # shape of `f_re`: (batch, N, d/2)
        f_re_obs_repeat = torch.repeat_interleave(f_re_obs, nei.shape[-3], -3)
        f_re_concat = torch.concat([f_re_obs_repeat, f_re_nei], dim=-1)
        f_re_flat = torch.flatten(f_re_concat, start_dim=-2, end_dim=-1)
        f_re_nei = self.re_fc3(self.re_fc2(self.re_fc1(f_re_flat)))

        # Compute meta components in a SocialCircle-like way
        # `nei` are relative values to target agents' last obs step
        nei_position = nei[..., -1, :]
        nei_position = self.picker.get_center(nei_position)

        # Distance factor (each neighbor)
        f_distance = torch.norm(nei_position, dim=-1)

        # Direction factor (each neighbor)
        f_direction = torch.atan2(nei_position[..., 0],
                                  nei_position[..., 1])
        f_direction = f_direction % (2*np.pi)

        # Compute angles
        angle_indices = f_direction / (2*np.pi/self.re_args.partitions)
        angle_indices = angle_indices.to(torch.int32)

        # Mask neighbors
        nei_mask = get_mask(torch.sum(nei, dim=[-1, -2]), torch.int32)
        angle_indices = angle_indices * nei_mask + -1 * (1 - nei_mask)

        scan_circle = []
        resonance_circle = []
        for ang in range(self.re_args.partitions):
            _mask = (angle_indices == ang).to(torch.float32)
            _mask_count = torch.sum(_mask, dim=-1)

            n = _mask_count + 0.0001

            scan_circle.append([])
            scan_circle[-1].append(torch.sum(f_distance * _mask, dim=-1) / n)
            scan_circle[-1].append(torch.sum(f_direction * _mask, dim=-1) / n)

            resonance_circle.append(
                torch.sum(f_re_nei * _mask[..., None], dim=-2) / n[..., None])

        # Stack all partitions
        scan_circle = [torch.stack(i, dim=-1) for i in scan_circle]
        scan_circle = torch.stack(scan_circle, dim=-2)
        resonance_circle = torch.stack(resonance_circle, dim=-2)

        # Encode circle components -> (batch, partition, d/2)
        f_scan = self.ce(scan_circle)

        # Concat resonance features -> (batch, partition, d)
        f_re = torch.concat([resonance_circle, f_scan], dim=-1)

        # Concat resonance features with trajectory features
        max_steps = max(self.re_args.partitions, self.args.obs_frames)
        f_traj_pad = pad(f_traj, max_steps)
        f_re_pad = pad(f_re, max_steps)

        f_behavior = torch.concat([f_traj_pad, f_re_pad], dim=-1)
        f_behavior = self.concat_fc(f_behavior)

        re_predictions = []
        for _ in range(repeats):
            # Assign random ids and embedding -> (batch, steps, d)
            z = torch.normal(mean=0, std=1,
                             size=list(f_behavior.shape[:-1]) + [self.d_id])
            re_f_z = self.re_ie(z.to(obs.device))

            # (batch, steps, 2*d)
            re_f_final = torch.concat([f_behavior, re_f_z], dim=-1)

            # Transformer outputs' shape is (batch, steps, d)
            f_tran, _ = self.re_T(inputs=re_f_final,
                                  targets=traj_targets,
                                  training=training)

            # Multiple generations -> (batch, Kc, d)
            re_adj = self.re_ms_fc(f_final)               # (batch, steps, Kc)
            re_adj = torch.transpose(re_adj, -1, -2)
            re_f_multi = self.re_ms_conv(f_tran, re_adj)     # (batch, Kc, d)

            # Forecast keypoints -> (..., Kc, Tsteps_Key, Tchannels)
            re_y = self.re_decoder_fc1(re_f_multi)
            re_y = self.re_decoder_fc2(re_y)
            re_y = torch.reshape(re_y, list(re_y.shape[:-1]) +
                                 [self.Trsteps_de, self.Trchannels_de])

            re_y = self.itr1(re_y)
            re_predictions.append(re_y)

        y_interaction = torch.concat(
            re_predictions, dim=-3)   # (batch, K, n_key, dim)

        return y_clean_interp + y_interaction

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


class ResonanceStructure(Structure):
    MODEL_TYPE = ResonanceModel


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
