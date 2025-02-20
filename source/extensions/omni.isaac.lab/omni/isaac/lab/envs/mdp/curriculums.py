# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv, ConstrainedManagerBasedRLEnv


def modify_reward_weight(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, weight: float, num_steps: int):
    """Curriculum that modifies a reward weight a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)

def modify_constraint_max_prob(env: ConstrainedManagerBasedRLEnv, env_ids: Sequence[int]):
    """Curriculum that modifies the maximum probability of a constraint a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the constraint term.
        max_prob: The maximum probability of the constraint term.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > 0:
        max_epochs = env.extras["max_epochs"]
        current_epoch = env.extras["current_epoch"]
        coeff = current_epoch / max_epochs
    else:
        coeff = 0.0
    # Get term cfg
    for term_name in env.constraint_manager.get_names():
        term_cfg = env.constraint_manager.get_term_cfg(term_name)
        init_max_prob = term_cfg.init_max_p
        final_max_prob = term_cfg.final_max_p
        
        # Compute the current max prob
        max_prob = init_max_prob + coeff * (final_max_prob - init_max_prob)
        max_prob = min(max_prob, final_max_prob)

        # Set the new max prob
        term_cfg.max_p = max_prob
        env.constraint_manager.set_term_cfg(term_name, term_cfg)