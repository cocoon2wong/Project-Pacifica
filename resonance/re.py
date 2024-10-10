"""
@Author: Conghao Wong
@Date: 2024-10-08 19:18:40
@LastEditors: Conghao Wong
@LastEditTime: 2024-10-10 09:26:09
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.args import Args
from qpid.base import BaseManager
from qpid.constant import INPUT_TYPES
from qpid.model import Model, layers
from qpid.training import Structure
from qpid.utils import INIT_POSITION
from qpid.training.loss import l2

from .__args import ResonanceArgs
from .__loss import InterventionLoss
from .re_circle import CircleLayer
from .re_intention import IntentionModel
from .re_interaction import InteractionModel
from .re_resonance import ResonanceLayer


class ResonanceModel(Model):

    def __init__(self, structure=None, *args, **kwargs):
        super().__init__(structure, *args, **kwargs)

        self.as_final_stage_model = True

        # Init args
        self.args._set_default('K', 1)
        self.args._set_default('K_train', 1)
        self.re_args = self.args.register_subargs(ResonanceArgs, 're_args')

        # Set model inputs
        self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                        INPUT_TYPES.NEIGHBOR_TRAJ)

        # Layers
        # Transform layers (ego)
        tlayer, itlayer = layers.get_transform_layers(self.re_args.T)
        self.t1 = tlayer((self.args.obs_frames, self.dim))
        self.it1 = itlayer((len(self.output_pred_steps), self.dim))

        # Transform layers (neighbors)
        trlayer, itrlayer = layers.get_transform_layers(self.re_args.Tr)
        self.tr1 = trlayer((self.args.obs_frames, self.dim))
        self.itr1 = itrlayer((self.args.pred_frames, self.dim))

        # Intention Branch
        self.intention_model = IntentionModel(self.args,
                                              T_ego_obs=self.t1,
                                              iT_ego_pred=self.it1)

        # Interaction Branch
        self.resonance_layer = ResonanceLayer(T_nei_obs=self.tr1,
                                              hidden_units=self.d,
                                              output_units=self.d//2)

        self.circle_layer = CircleLayer(partitions=self.re_args.partitions,
                                        output_units=self.d)

        self.interaction_model = InteractionModel(self.args,
                                                  T_nei_obs=self.tr1,
                                                  iT_nei_pred=self.itr1)

    def forward(self, inputs: list[torch.Tensor], training=None, mask=None, *args, **kwargs):
        # Unpack inputs
        # (batch, obs, dim)
        ego = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)

        # (batch, N, obs, dim)
        nei = self.get_input(inputs, INPUT_TYPES.NEIGHBOR_TRAJ)

        # Intention Branch
        y_clean, f_ego = self.intention_model(
            ego, self.output_pred_steps, training)

        # Interaction Branch
        # Compute resonance
        f_resonance_meta = self.resonance_layer(ego, nei, training)

        f_re = self.circle_layer(self.picker.get_center(ego),
                                 self.picker.get_center(nei),
                                 f_resonance=f_resonance_meta,
                                 training=training)

        y_interaction = self.interaction_model(ego, f_ego, f_re, training)

        if training:
            # Compute counterfactual outputs
            nei_c = self.create_empty_neighbors(ego)
            f_resonance_meta_c = self.resonance_layer(ego, nei_c, training)
            f_re_c = self.circle_layer(self.picker.get_center(ego),
                                       self.picker.get_center(nei_c),
                                       f_resonance=f_resonance_meta_c,
                                       training=training)
            y_interaction_c = self.interaction_model(
                ego, f_ego, f_re_c, training)
        else:
            y_interaction_c = ego

        if self.re_args.no_interaction:
            Y = y_clean
        else:
            Y = y_clean + y_interaction

        return Y, torch.mean(f_re, dim=-1, keepdim=True), y_interaction_c

    def create_empty_neighbors(self, ego_traj: torch.Tensor):
        empty = INIT_POSITION * torch.ones([ego_traj.shape[0],
                                            self.args.max_agents - 1,
                                            ego_traj.shape[-2],
                                            ego_traj.shape[-1]]).to(ego_traj.device)
        return torch.concat([ego_traj[..., None, :, :], empty], dim=-3)


class ResonanceStructure(Structure):
    MODEL_TYPE = ResonanceModel

    def __init__(self, args: list[str] | Args | None = None, manager: BaseManager | None = None, name='Train Manager'):
        super().__init__(args, manager, name)

        self.loss.set({l2: 0.5, InterventionLoss: 0.5})
