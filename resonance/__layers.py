"""
@Author: Conghao Wong
@Date: 2024-10-15 14:54:50
@LastEditors: Conghao Wong
@LastEditTime: 2025-02-17 16:46:20
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import numpy as np
import torch

from qpid.model import layers
from qpid.model.layers.transfroms import _BaseTransformLayer
from qpid.utils import get_mask


class SocialCircleLayer(torch.nn.Module):
    """
    A layer to compute SocialCircle meta components and then encode
    them to high-dimensional features.

    Supported factors:
    - Velocity;
    - Distance;
    - Direction;
    - Movement Direction (Optional).
    """

    def __init__(self, partitions: int,
                 output_units: int,
                 use_velocity: bool | int = True,
                 use_distance: bool | int = True,
                 use_direction: bool | int = True,
                 use_move_direction: bool | int = False,
                 mu=0.0001,
                 relative_velocity: bool | int = False,
                 *args, **kwargs):
        """
        ## Partition Settings
        :param partitions: The number of partitions in the circle.
        :param output_units: Dimension of the output feature.

        ## SocialCircle Meta Components
        :param use_velocity: Choose whether to use the `velocity` factor.
        :param use_distance: Choose whether to use the `distance` factor.
        :param use_direction: Choose whether to use the `direction` factor.
        :param use_move_direction: Choose whether to use the `move direction` factor.

        ## SocialCircle Options
        :param relative_velocity: Choose whether to use relative velocity or not.
        :param mu: The small number to prevent dividing zero when computing. \
            It only works when `relative_velocity` is set to `True`.
        """
        super().__init__(*args, **kwargs)

        self.partitions = partitions
        self.d = output_units

        self.use_velocity = use_velocity
        self.use_distance = use_distance
        self.use_direction = use_direction

        self.rel_velocity = relative_velocity
        self.use_move_direction = use_move_direction
        self.mu = mu

        # Circle encoding
        self.ce = layers.TrajEncoding(self.dim, self.d, torch.nn.ReLU)

    @property
    def dim(self) -> int:
        """
        The number of SocialCircle factors.
        """
        return int(self.use_velocity) + int(self.use_distance) + \
            int(self.use_direction) + int(self.use_move_direction)

    def forward(self, trajs, nei_trajs, *args, **kwargs):
        # Move vectors -> (batch, ..., 2)
        # `nei_trajs` are relative values to target agents' last obs step
        obs_vector = trajs[..., -1:, :] - trajs[..., 0:1, :]
        nei_vector = nei_trajs[..., -1, :] - nei_trajs[..., 0, :]
        nei_posion_vector = nei_trajs[..., -1, :]

        # Velocity factor
        if self.use_velocity:
            # Calculate velocities
            nei_velocity = torch.norm(nei_vector, dim=-1)    # (batch, n)
            obs_velocity = torch.norm(obs_vector, dim=-1)    # (batch, 1)

            # Speed factor in the SocialCircle
            if self.rel_velocity:
                f_velocity = (nei_velocity + self.mu)/(obs_velocity + self.mu)
            else:
                f_velocity = nei_velocity

        # Distance factor
        if self.use_distance:
            f_distance = torch.norm(nei_posion_vector, dim=-1)

        # Move direction factor
        if self.use_move_direction:
            obs_move_direction = torch.atan2(obs_vector[..., 0],
                                             obs_vector[..., 1])
            nei_move_direction = torch.atan2(nei_vector[..., 0],
                                             nei_vector[..., 1])
            delta_move_direction = nei_move_direction - obs_move_direction
            f_move_direction = delta_move_direction % (2*np.pi)

        # Direction factor
        f_direction = torch.atan2(nei_posion_vector[..., 0],
                                  nei_posion_vector[..., 1])
        f_direction = f_direction % (2*np.pi)

        # Angles (the independent variable \theta)
        angle_indices = f_direction / (2*np.pi/self.partitions)
        angle_indices = angle_indices.to(torch.int32)

        # Mask neighbors
        nei_mask = get_mask(torch.sum(nei_trajs, dim=[-1, -2]), torch.int32)
        angle_indices = angle_indices * nei_mask + -1 * (1 - nei_mask)

        # Compute the SocialCircle
        social_circle = []
        for ang in range(self.partitions):
            _mask = (angle_indices == ang).to(torch.float32)
            _mask_count = torch.sum(_mask, dim=-1)

            n = _mask_count + 0.0001
            social_circle.append([])

            if self.use_velocity:
                _velocity = torch.sum(f_velocity * _mask, dim=-1) / n
                social_circle[-1].append(_velocity)

            if self.use_distance:
                _distance = torch.sum(f_distance * _mask, dim=-1) / n
                social_circle[-1].append(_distance)

            if self.use_direction:
                _direction = torch.sum(f_direction * _mask, dim=-1) / n
                social_circle[-1].append(_direction)

            if self.use_move_direction:
                _move_d = torch.sum(f_move_direction * _mask, dim=-1) / n
                social_circle[-1].append(_move_d)

        # Shape of the final SocialCircle: (batch, p, 3)
        social_circle = [torch.stack(i) for i in social_circle]
        social_circle = torch.stack(social_circle)
        social_circle = torch.permute(social_circle, [2, 0, 1])

        f_sc = self.ce(social_circle)
        return f_sc, social_circle


class PoolingLayer(torch.nn.Module):

    def __init__(self, grids: int,
                 hidden_units: int,
                 output_units: int,
                 transform_layer: _BaseTransformLayer,
                 range_gain: float = 2.0,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.grid_length = int(grids ** 0.5)
        assert self.grid_length % 2 == 0, 'Grid lengths should be even!'

        self.range_gain = range_gain
        self.d_h = hidden_units
        self.d = output_units
        self.T_layer = transform_layer

        # Shapes
        self.Trsteps_en, self.Trchannels_en = self.T_layer.Tshape

        # Trajectory encoding (neighbors)
        self.tre = layers.TrajEncoding(self.T_layer.Oshape[-1], hidden_units,
                                       torch.nn.ReLU,
                                       transform_layer=self.T_layer)

        self.fc1 = layers.Dense(hidden_units*self.Trsteps_en,
                                hidden_units,
                                torch.nn.ReLU)
        self.fc2 = layers.Dense(hidden_units, hidden_units, torch.nn.ReLU)
        self.fc3 = layers.Dense(hidden_units, output_units, torch.nn.ReLU)

    def forward(self, x_ego_2d: torch.Tensor,
                x_nei_2d: torch.Tensor):

        # Move the last point of trajectories to 0
        x_ego_pure = (x_ego_2d - x_ego_2d[..., -1:, :])[..., None, :, :]
        x_nei_pure = x_nei_2d - x_nei_2d[..., -1:, :]

        # Embed trajectories (ego + neighbor) together and then split them
        f_pack = self.tre(torch.concat([x_ego_pure, x_nei_pure], dim=-3))
        f_ego = f_pack[..., :1, :, :]
        f_nei = f_pack[..., 1:, :, :]

        # Compute meta resonance features (for each neighbor)
        # shape of the final output `f_re_meta`: (batch, N, d/2)
        f = f_ego * f_nei   # -> (batch, N, obs, d)
        f = torch.flatten(f, start_dim=-2, end_dim=-1)
        f_re = self.fc3(self.fc2(self.fc1(f)))

        # Compute features in the SocialPooling-like way
        # Compute the length of each grid
        vel = torch.norm(x_ego_2d[..., -1, :] - x_ego_2d[..., 0, :], dim=-1)
        total_length = self.range_gain * vel
        grid_interval = total_length / self.grid_length

        grid_indices = x_nei_2d[:, ..., -1, :] - x_ego_2d[..., -1:, :]
        grid_indices = torch.ceil(grid_indices/grid_interval[..., None, None])

        grids = []
        r = range(-self.grid_length//2, self.grid_length//2 + 1)
        for i in r:
            for j in r:
                if i*j == 0:
                    continue

                _mask = (grid_indices -
                         torch.tensor([[[i, j]]]).to(vel.device))
                _mask = ((_mask[..., 0] == 0) *
                         (_mask[..., 1] == 0)).to(torch.float32)

                feature = torch.sum(f_re * _mask[..., None], dim=-2)
                grids.append(feature)

        # Shape of the final feature: (batch, grids, d)
        grids = torch.stack(grids, dim=-2)

        return grids, f_re
