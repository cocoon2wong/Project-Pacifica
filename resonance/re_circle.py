"""
@Author: Conghao Wong
@Date: 2024-10-09 20:28:02
@LastEditors: Conghao Wong
@LastEditTime: 2024-10-09 21:43:18
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import numpy as np
import torch

from qpid.utils import get_mask
from qpid.model import layers


class CircleLayer(torch.nn.Module):
    def __init__(self, partitions: int,
                 output_units: int,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.partitions = partitions
        self.d = output_units

        # Circle encoding
        self.ce = layers.TrajEncoding(2, self.d//2, torch.nn.ReLU)

    def forward(self, ego_traj_2d: torch.Tensor,
                nei_traj_2d: torch.Tensor,
                f_resonance: torch.Tensor,
                training=None, mask=None, *args, **kwargs):

        # Compute meta components in a SocialCircle-like way
        # `nei` are relative values to target agents' last obs step
        nei_rel_pos = nei_traj_2d - ego_traj_2d[..., None, -1:, :]
        nei_rel_pos = nei_rel_pos[..., -1, :]

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

        scan_circle = []
        resonance_circle = []
        for ang in range(self.partitions):
            _mask = (angle_indices == ang).to(torch.float32)
            _mask_count = torch.sum(_mask, dim=-1)

            n = _mask_count + 0.0001

            scan_circle.append([])
            scan_circle[-1].append(torch.sum(f_distance * _mask, dim=-1) / n)
            scan_circle[-1].append(torch.sum(f_direction * _mask, dim=-1) / n)

            resonance_circle.append(
                torch.sum(f_resonance * _mask[..., None], dim=-2) / n[..., None])

        # Stack all partitions
        scan_circle = [torch.stack(i, dim=-1) for i in scan_circle]
        scan_circle = torch.stack(scan_circle, dim=-2)
        resonance_circle = torch.stack(resonance_circle, dim=-2)

        # Encode circle components -> (batch, partition, d/2)
        f_scan = self.ce(scan_circle)
        
        # Concat resonance features -> (batch, partition, d)
        f_re = torch.concat([resonance_circle, f_scan], dim=-1)

        return f_re
