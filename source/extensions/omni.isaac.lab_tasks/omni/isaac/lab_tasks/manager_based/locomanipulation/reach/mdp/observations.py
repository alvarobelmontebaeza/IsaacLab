# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`omni.isaac.lab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import pytorch3d.transforms as pt3d
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.sensors import ContactSensor
if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def pose_command_cartesian_6d_rotation(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    # Get the pose command from the command manager
    command = env.command_manager.get_command(command_name)
    # Separate the position and rotation components
    pos = command[:, :3]
    rot_quat = command[:, 3:]
    # Convert the rotation to 6d orientation representation that favours learning
    rot_mat = pt3d.quaternion_to_matrix(rot_quat)
    rot_6d = pt3d.matrix_to_rotation_6d(rot_mat)

    return torch.cat([pos, rot_6d], dim=1)

def body_pose_cartesian_6d_rotation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    # Get body idx from asset cfg
    asset: RigidObject = env.scene[asset_cfg.name]
    body_pose= asset.data.body_state_w[:, asset_cfg.body_ids[0], :7]  # type: ignore
    # Separate the position and rotation components
    pos = body_pose[:, :3]
    rot_quat = body_pose[:, 3:]
    # Convert the rotation to 6d orientation representation that favours learning
    rot_mat = pt3d.quaternion_to_matrix(rot_quat)
    rot_6d = pt3d.matrix_to_rotation_6d(rot_mat)

    return torch.cat([pos, rot_6d], dim=1)


def feet_contacts(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    print(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :])
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    print(contacts)
    return contacts