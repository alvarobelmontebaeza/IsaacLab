# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.terrains import TerrainImporter

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`omni.isaac.lab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())

def target_pose_ranges(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance between the robot and the target pose.

    This term is used to increase the difficulty of the target pose when the robot reaches the target pose and decrease
    the difficulty when the robot is far from the target pose.

    Returns:
        The mean target pose range for the given environment ids.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # Extract current command and command term object
    command = env.command_manager.get_command(command_name)
    command_term = env.command_manager.get_term(command_name)
    # obtain the desired and current poses    
    des_pos_w, des_quat_w = command[:, :3], command[:, 3:]
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    # Compute the position and orientation errors
    pos_error = torch.sum(torch.square(curr_pos_w - des_pos_w), dim=1).sqrt()
    rot_error = quat_error_magnitude(curr_quat_w, des_quat_w)

    # Obtain the sigma values for position and orientation
    sigma_pos, sigma_rot = 0.05, 1.0#_get_sigmas(pos_error, rot_error)

    pos_rew = torch.exp(-(pos_error**2) / sigma_pos)
    rot_rew = torch.exp(-rot_error / sigma_rot)

    return pos_rew * rot_rew
