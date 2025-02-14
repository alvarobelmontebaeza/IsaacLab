# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable reward functions.

The functions can be passed to the :class:`omni.isaac.lab.managers.RewardTermCfg` object to include
the reward introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import ConstraintTermCfg
from omni.isaac.lab.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ConstrainedManagerBasedRLEnv

"""
Body constraints.
"""

def cstr_flat_orientation(env: ConstrainedManagerBasedRLEnv, limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    base_orientation = asset.data.projected_gravity_b[:, :2]

    return torch.norm(base_orientation, dim=1) - limit


def cstr_base_height(
    env: ConstrainedManagerBasedRLEnv, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        Currently, it assumes a flat terrain, i.e. the target height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    return asset.data.root_pos_w[:, 2] - target_height

"""
Joint constraints.
"""

def cstr_joint_vel_limits(env: ConstrainedManagerBasedRLEnv, limits: float | dict[str, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Constrain the joint velocities to be held within limits

    The termination probability of the constraint will increase with how much the velocity exceeds the limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    velocities = asset.data.joint_vel[:, asset_cfg.joint_ids]
    if isinstance(limits, float):
        velocity_limits = torch.ones_like(velocities) * limits
        cstr_velocity = torch.abs(velocities) - limits
    elif isinstance(limits, dict):
        velocity_limits = torch.tensor([limits[joint] for joint in asset_cfg.joint_ids], device=velocities.device)
        cstr_velocity = torch.abs(velocities) - velocity_limits
    else:
        raise ValueError("Invalid type for limits")
    
    return cstr_velocity.clip(min=0.0)

def cstr_joint_acc_limits(env: ConstrainedManagerBasedRLEnv, limits: float | dict[str, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Constrain the joint accelerations to be held within limits

    The termination probability of the constraint will increase with how much the acceleration exceeds the limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    accelerations = asset.data.joint_acc[:, asset_cfg.joint_ids]
    if isinstance(limits, float):
        acceleration_limits = torch.ones_like(accelerations) * limits
        cstr_acceleration = torch.abs(accelerations) - limits
    elif isinstance(limits, dict):
        acceleration_limits = torch.tensor([limits[joint] for joint in asset_cfg.joint_ids], device=accelerations.device)
        cstr_acceleration = torch.abs(accelerations) - acceleration_limits
    else:
        raise ValueError("Invalid type for limits")
    
    return cstr_acceleration.clip(min=0.0)

def cstr_joint_torque_limits(env: ConstrainedManagerBasedRLEnv, limits: float | dict[str, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Constrain the torques to be held within limits

    The termination probability of the constraint will increase with how much the torque exceeds the limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    if isinstance(limits, float):
        torque_limits = torch.ones_like(torques) * limits
        cstr_torque = torch.abs(torques) - limits
    elif isinstance(limits, dict):
        torque_limits = torch.tensor([limits[joint] for joint in asset_cfg.joint_ids], device=torques.device)
        cstr_torque = torch.abs(torques) - torque_limits
    else:
        raise ValueError("Invalid type for limits")
    
    return cstr_torque


"""
Action penalties.
"""

def cstr_action_rate(env: ConstrainedManagerBasedRLEnv, limit: float) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.abs(env.action_manager.action - env.action_manager.prev_action) - limit


"""
Contact sensor.
"""


def cstr_undesired_contacts(env: ConstrainedManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)


def cstr_contact_forces(env: ConstrainedManagerBasedRLEnv, limit: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # compute the violation
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - limit
    # compute the penalty
    return violation