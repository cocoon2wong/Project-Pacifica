"""
@Author: Conghao Wong
@Date: 2024-10-10 18:26:32
@LastEditors: Conghao Wong
@LastEditTime: 2024-10-11 10:57:00
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import numpy as np
import torch

from qpid.model import layers
from qpid.model.layers.transfroms import _BaseTransformLayer
from qpid.utils import get_mask


class ResonanceCircle(torch.nn.Module):
    """
    ResonanceCircle
    ---
    Resonance: The "similarity" of trajectory spectra of the ego agent and all
    its neighbors. It supposes that social interactions are related to the
    "frequency" of trajectories, especially the resonance phenomenon when two 
    trajectories are performed under similar frequency portions.
    """

    def __init__(self, partitions: int,
                 hidden_units: int,
                 output_units: int,
                 transform_layer: _BaseTransformLayer,
                 *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.partitions = partitions
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
        self.fc3 = layers.Dense(hidden_units, output_units//2, torch.nn.ReLU)

        # Circle encoding (only for other components)
        self.ce = layers.TrajEncoding(2, output_units//2, torch.nn.ReLU)

    def forward(self, ego_traj_2d: torch.Tensor,
                nei_traj_2d: torch.Tensor):

        # Move the last point of trajectories to 0
        nei_r = nei_traj_2d - nei_traj_2d[..., -1:, :]
        ego_r = (ego_traj_2d - ego_traj_2d[..., -1:, :])[..., None, :, :]

        # Embed trajectories (ego + neighbor) together and then split them
        f_pack = self.tre(torch.concat([ego_r, nei_r], dim=-3))
        f_ego = f_pack[..., :1, :, :]
        f_nei = f_pack[..., 1:, :, :]

        # Compute meta resonance features (for each neighbor)
        # shape of the final output `f_re_meta`: (batch, N, d/2)
        f = f_ego * f_nei   # -> (batch, N, obs, d)
        f = torch.flatten(f, start_dim=-2, end_dim=-1)
        f_re_meta = self.fc3(self.fc2(self.fc1(f)))

        # Compute meta components in a SocialCircle-like way
        # `nei` are relative values to target agents' last obs step
        nei_rel_pos = nei_traj_2d[..., -1, :]

        # Distance factor (each neighbor)
        f_distance = torch.norm(nei_rel_pos, dim=-1)

        # Direction factor (each neighbor)
        f_direction = torch.atan2(nei_rel_pos[..., 0],
                                  nei_rel_pos[..., 1])
        f_direction = f_direction % (2*np.pi)

        # Compute angles
        angle_indices = f_direction / (2*np.pi/self.partitions)
        angle_indices = angle_indices.to(torch.int32)

        # Mask neighbors
        nei_mask = get_mask(torch.sum(nei_traj_2d, dim=[-1, -2]), torch.int32)
        angle_indices = angle_indices * nei_mask + -1 * (1 - nei_mask)

        sc_components = []
        resonance_features = []
        for ang in range(self.partitions):
            _mask = (angle_indices == ang).to(torch.float32)
            _mask_count = torch.sum(_mask, dim=-1)

            n = _mask_count + 0.0001

            sc_components.append([])
            sc_components[-1].append(torch.sum(f_distance * _mask, dim=-1) / n)
            sc_components[-1].append(torch.sum(f_direction *
                                     _mask, dim=-1) / n)

            resonance_features.append(
                torch.sum(f_re_meta * _mask[..., None], dim=-2) / n[..., None])

        # Stack all partitions
        sc_components = [torch.stack(i, dim=-1) for i in sc_components]
        sc_components = torch.stack(sc_components, dim=-2)
        resonance_features = torch.stack(resonance_features, dim=-2)

        # Encode circle components -> (batch, partition, d/2)
        f_scan = self.ce(sc_components)

        # Concat resonance features -> (batch, partition, d)
        f_re = torch.concat([resonance_features, f_scan], dim=-1)

        return f_re
