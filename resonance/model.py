"""
@Author: Conghao Wong
@Date: 2024-10-08 19:18:40
@LastEditors: Conghao Wong
@LastEditTime: 2024-10-16 19:16:08
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.constant import INPUT_TYPES
from qpid.model import Model, layers
from qpid.training import Structure
from qpid.utils import INIT_POSITION

from .__args import ResonanceArgs
from .__layers import SocialCircleLayer
from .linearDiffEncoding import LinearDiffEncoding
from .reSelfBias import ReSelfBias
from .resonanceBias import ResonanceBias
from .resonanceCircle import ResonanceCircle


class ResonanceModel(Model):

    def __init__(self, structure=None, *args, **kwargs):
        super().__init__(structure, *args, **kwargs)

        self.as_final_stage_model = True

        # Init args
        self.args._set_default('K', 1)
        self.args._set_default('K_train', 1)
        self.re_args = self.args.register_subargs(ResonanceArgs, 're_args')

        # Set model inputs
        if not self.re_args.encode_agent_types:
            self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                            INPUT_TYPES.NEIGHBOR_TRAJ)
        else:
            self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                            INPUT_TYPES.NEIGHBOR_TRAJ,
                            INPUT_TYPES.AGENT_TYPES)

        # Layers
        # Transform layers (ego)
        tlayer, itlayer = layers.get_transform_layers(self.re_args.T)
        self.t1 = tlayer((self.args.obs_frames, self.dim))
        self.it1 = itlayer((len(self.output_pred_steps), self.dim))

        # Transform layers (neighbors)
        trlayer, itrlayer = layers.get_transform_layers(self.re_args.Tr)
        self.tr1 = trlayer((self.args.obs_frames, self.dim))
        self.itr1 = itrlayer((self.args.pred_frames, self.dim))

        # Linear Difference Encoding Layer
        self.linear = LinearDiffEncoding(obs_frames=self.args.obs_frames,
                                         pred_frames=self.args.pred_frames,
                                         output_units=self.d//2,
                                         transform_layer=self.t1,
                                         encode_agent_types=self.re_args.encode_agent_types)

        # Self-Bias Layer
        if self.re_args.learn_self_bias:
            self.b1 = ReSelfBias(self.args,
                                 output_units=self.d,
                                 noise_units=self.d//2,
                                 transform_layer=self.t1,
                                 itransform_layer=self.it1)

        if not self.re_args.learn_re_bias:
            return

        # Resonance Circle Layer
        if not self.re_args.use_original_socialcircle:
            self.rc = ResonanceCircle(partitions=self.re_args.partitions,
                                      hidden_units=self.d,
                                      output_units=self.d,
                                      transform_layer=self.tr1)
        else:
            self.rc = SocialCircleLayer(partitions=self.re_args.partitions,
                                        output_units=self.d)

        # Resonance Bias Layer
        self.b2 = ResonanceBias(self.args,
                                output_units=self.d,
                                noise_units=self.d//2,
                                ego_feature_dim=self.d,
                                re_feature_dim=self.d//2,
                                T_nei_obs=self.tr1,
                                iT_nei_pred=self.itr1)

    def _compute(self, ego_traj: torch.Tensor,
                 nei_traj: torch.Tensor,
                 training=None,
                 y_self_bias=None):
        """
        Implement the entire model on the given trajectories (ego + neighbors).
        """

        # Compute linear-difference features
        f_ego_diff, ego_traj_linear, y_linear = self.linear(ego_traj)

        # Predict the self-bias trajectory
        if self.re_args.learn_self_bias and y_self_bias is None:
            # Intention Branch
            y_self_bias = self.b1(ego_traj_linear, f_ego_diff,
                                  self.output_pred_steps, training)
        else:
            y_self_bias = 0

        if self.re_args.learn_re_bias:
            # Compute the ResonanceCircle to each ego agent
            f_re = self.rc(self.picker.get_center(ego_traj),
                           self.picker.get_center(nei_traj))

            # Compute the resonance-bias trajectory
            y_re_bias = self.b2(ego_traj - ego_traj_linear,
                                f_ego_diff, f_re, training)
        else:
            y_re_bias = 0

        y = y_linear[..., None, :, :]

        if training or not self.re_args.no_self_bias:
            y = y + y_self_bias

        if training or not self.re_args.no_re_bias:
            y = y + y_re_bias

        return [y,
                y_self_bias,
                y_re_bias]

    def forward(self, inputs: list[torch.Tensor], training=None, mask=None, *args, **kwargs):
        # Unpack inputs
        # (batch, obs, dim)
        ego_traj = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)

        # (batch, N, obs, dim)
        if self.re_args.no_interaction:
            nei_traj = self.create_empty_neighbors(ego_traj)
        else:
            nei_traj = self.get_input(inputs, INPUT_TYPES.NEIGHBOR_TRAJ)

        # Forward the model
        # Compute linear-difference features
        if self.re_args.encode_agent_types:
            agent_types = self.get_input(inputs, INPUT_TYPES.AGENT_TYPES)
        else:
            agent_types = None

        f_ego_diff, ego_traj_linear, y_linear = self.linear(
            ego_traj, agent_types)

        # Predict the self-bias trajectory
        if self.re_args.learn_self_bias:
            # Intention Branch
            y_self_bias = self.b1(ego_traj_linear, f_ego_diff,
                                  self.output_pred_steps, training)
        else:
            y_self_bias = 0

        if self.re_args.learn_re_bias:
            # Compute the ResonanceCircle to each ego agent
            f_re = self.rc(self.picker.get_center(ego_traj)[..., :2],
                           self.picker.get_center(nei_traj)[..., :2])

            # Compute the resonance-bias trajectory
            y_re_bias = self.b2(ego_traj - ego_traj_linear,
                                f_ego_diff, f_re, training)
        else:
            y_re_bias = 0

        y = y_linear[..., None, :, :]

        if training or not self.re_args.no_self_bias:
            y = y + y_self_bias

        if training or not self.re_args.no_re_bias:
            y = y + y_re_bias

        return y

    def create_empty_neighbors(self, ego_traj: torch.Tensor):
        """
        Create the neighbor trajectory matrix that only contains the ego agent.
        """
        empty = INIT_POSITION * torch.ones([ego_traj.shape[0],
                                            self.args.max_agents - 1,
                                            ego_traj.shape[-2],
                                            ego_traj.shape[-1]]).to(ego_traj.device)
        return torch.concat([ego_traj[..., None, :, :], empty], dim=-3)


class ResonanceStructure(Structure):
    MODEL_TYPE = ResonanceModel
