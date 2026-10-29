# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from ..mdp.symmetry import mirror_left_right

# Left-right symmetry (CLOCK_SCAN_SYMMETRY): "aug" trains every mini-batch also on its mirror; "loss" adds
# MIRROR_LOSS_COEF x the mean squared difference between the actions in the mirrored state and the mirrored actions
# (one update on rare states with huge outputs broke a run 600 iterations into stairs); "0" is off.
SYMMETRY = os.environ.get("CLOCK_SCAN_SYMMETRY", "aug")
MIRROR_LOSS_COEF = float(os.environ.get("CLOCK_SCAN_MIRROR_LOSS_COEF", "1.0"))
CLIP_ACTIONS = float(os.environ.get("CLOCK_SCAN_CLIP_ACTIONS", "10"))


@configclass
class BoundedGaussianDistributionCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    class_name: str = "clock_scan.tasks.locomotion.agents.distribution:BoundedGaussianDistribution"
    bound: float = CLIP_ACTIONS


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = ""  # same as task name
    empirical_normalization = False
    # Actions, and so the last_action input, stay within +-CLOCK_SCAN_CLIP_ACTIONS (10 = +-2.5 rad from the default pose,
    # the whole joint range). At 100, a policy fed its own large last action could grow it 10x per step to ~40000.
    clip_actions = CLIP_ACTIONS
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    # The actor's mean is bound * tanh(x / bound): past the clip it would drift with nothing to pull it back.
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=BoundedGaussianDistributionCfg(init_std=1.0),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=SYMMETRY == "aug",
            use_mirror_loss=SYMMETRY == "loss",
            mirror_loss_coeff=MIRROR_LOSS_COEF,
            data_augmentation_func=mirror_left_right,
        )
        if SYMMETRY in ("aug", "loss")
        else None,
    )
