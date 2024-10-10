"""
@Author: Conghao Wong
@Date: 2024-10-09 19:54:07
@LastEditors: Conghao Wong
@LastEditTime: 2024-10-09 20:27:14
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.model import layers
from qpid.model.layers.transfroms import _BaseTransformLayer


class ResonanceLayer(torch.nn.Module):

    def __init__(self, T_nei_obs: _BaseTransformLayer,
                 hidden_units: int,
                 output_units: int,
                 *args, **kwargs) -> None:
        """
        :param hidden_units: Dimension of the middle representations during \
            the network computation.
        :param output_units: Dimension of the output features.

        """
        super().__init__(*args, **kwargs)

        # Layers
        # Transform layers (to compute interaction)
        self.tr1 = T_nei_obs

        # Shapes
        self.Trsteps_en, self.Trchannels_en = self.tr1.Tshape

        # Trajectory encoding (neighbors)
        self.tre = layers.TrajEncoding(self.tr1.Oshape[-1], hidden_units,
                                       torch.nn.ReLU,
                                       transform_layer=self.tr1)

        self.fc1 = layers.Dense(hidden_units*2*self.Trsteps_en,
                                hidden_units,
                                torch.nn.ReLU)
        self.fc2 = layers.Dense(hidden_units, hidden_units, torch.nn.ReLU)
        self.fc3 = layers.Dense(hidden_units, output_units, torch.nn.ReLU)

    def forward(self, ego_traj: torch.Tensor,
                neighbor_traj: torch.Tensor,
                training=None, mask=None, *args, **kwargs):

        # Move the last point of trajectories to 0
        nei_r = neighbor_traj - neighbor_traj[..., -1:, :]
        ego_r = (ego_traj - ego_traj[..., -1:, :])[..., None, :, :]

        # Embed trajectories (ego + neighbor) together and then split them
        pack_r = torch.concat([nei_r, ego_r], dim=-3)
        f_pack = self.tre(pack_r)

        f_ego = f_pack[..., :1, :, :]
        f_nei = f_pack[..., 1:, :, :]

        # Compute resonance features (for each neighbor)
        # shape of the final output `f`: (batch, N, d/2)
        f = torch.repeat_interleave(f_ego, nei_r.shape[-3], -3)
        f = torch.concat([f, f_nei], dim=-1)
        f = torch.flatten(f, start_dim=-2, end_dim=-1)
        f = self.fc3(self.fc2(self.fc1(f)))
        return f
