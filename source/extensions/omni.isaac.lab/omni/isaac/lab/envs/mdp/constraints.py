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

def cstr_body_orientation_axis(env: ConstrainedManagerBasedRLEnv, axis: str, limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    base_orientation = asset.data.projected_gravity_b[:, :2]
    if axis == "x":
        return torch.abs(base_orientation[:, 0]) - limit
    elif axis == "y":
        return torch.abs(base_orientation[:, 1]) - limit
    else:
        raise ValueError(f"Invalid axis: {axis}. Only use 'x' or 'y'.")


def cstr_min_base_height(
    env: ConstrainedManagerBasedRLEnv, min_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        Currently, it assumes a flat terrain, i.e. the target height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    return min_height - asset.data.root_pos_w[:, 2]

def cstr_max_base_velocity(env: ConstrainedManagerBasedRLEnv, max_velocity: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the base velocity exceeding the maximum allowed value."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    base_velocity = asset.data.root_lin_vel_w

    return torch.norm(base_velocity, dim=1) - max_velocity

"""
Joint constraints.
"""

def cstr_joint_pos_upper_limits(env: ConstrainedManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Constrain the joint positions to be held within limits

    The termination probability of the constraint will increase with how much the position exceeds the limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    positions = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_limits = asset.data.joint_limits[:, asset_cfg.joint_ids]
    # joint_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    upper_lim, lower_lim = joint_limits[:,:,1], joint_limits[:,:,0]
    cstr_position = positions - upper_lim
    
    return cstr_position

def cstr_joint_pos_lower_limits(env: ConstrainedManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Constrain the joint positions to be held within limits

    The termination probability of the constraint will increase with how much the position exceeds the limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    positions = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_limits = asset.data.joint_limits[:, asset_cfg.joint_ids]
    # joint_limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
    upper_lim, lower_lim = joint_limits[:,:,1], joint_limits[:,:,0]
    cstr_position = lower_lim - positions
    
    return cstr_position

def cstr_joint_vel_limits(env: ConstrainedManagerBasedRLEnv, limits: None | float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Constrain the joint velocities to be held within limits

    The termination probability of the constraint will increase with how much the velocity exceeds the limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    velocities = asset.data.joint_vel[:, asset_cfg.joint_ids]
    cstr_velocity = torch.abs(velocities) - limits
    
    return cstr_velocity

def cstr_joint_acc_limits(env: ConstrainedManagerBasedRLEnv, limits: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Constrain the joint accelerations to be held within limits

    The termination probability of the constraint will increase with how much the acceleration exceeds the limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    accelerations = asset.data.joint_acc[:, asset_cfg.joint_ids]
    cstr_acceleration = torch.abs(accelerations) - limits
    
    return cstr_acceleration

def cstr_joint_torque_limits(env: ConstrainedManagerBasedRLEnv, limits: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Constrain the torques to be held within limits

    The termination probability of the constraint will increase with how much the torque exceeds the limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    cstr_torque = torch.abs(torques) - limits
    
    return cstr_torque

"""
Action penalties.
"""

def cstr_action_limits(env: ConstrainedManagerBasedRLEnv, action_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Constrain the actions to be held within limits

    The termination probability of the constraint will increase with how much the action exceeds the limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # Retrieve raw actions applied to the joints
    action_term = env.action_manager.get_term(action_name)
    processed_actions = action_term.processed_actions # Apply constraint to the processed actions, not the raw ones
    action_joint_ids = action_term._joint_ids # type: ignore
    # Retrieve joint limits
    joint_limits = asset.data.joint_limits[:, action_joint_ids]
    upper_lim, lower_lim = joint_limits[:,:,1], joint_limits[:,:,0] # Remove the gripper joints which are not actuated
    # Compute the constraint violation
    cstr_action_lim = torch.max(processed_actions - upper_lim, lower_lim - processed_actions)




def cstr_action_rate(env: ConstrainedManagerBasedRLEnv, limit: float) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    action_diff = torch.abs(env.action_manager.action - env.action_manager.prev_action)
    action_rate = action_diff / env.step_dt
    return action_rate - limit

def cstr_joint_deviation(env: ConstrainedManagerBasedRLEnv, limit: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Constrain the joints to be within a certain distance from their default positions.

    The termination probability of the constraint will increase with how much the joint positions deviate from the default positions.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the distance from default positions
    default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    distance_from_default = torch.abs(joint_pos - default_joint_pos)
    # compute the constraint violation
    cstr_distance = distance_from_default - limit
        
    return cstr_distance

def cstr_no_arm_movement(env: ConstrainedManagerBasedRLEnv, limit: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Compute a constraint that penalizes movement beyond a specified limit.
    Args:
        env (ConstrainedManagerBasedRLEnv): The environment containing the scene and command manager.
        limit (float): The distance limit beyond which movement is penalized.
        command_name (str): The name of the command to retrieve the desired position.
        asset_cfg (SceneEntityCfg, optional): Configuration for the scene entity (default is a robot).
    Returns:
        torch.Tensor: A tensor representing the constraint violation, where movement beyond the limit is penalized.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the distance from the target
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :2] # Desired XY pos of the EE in the base frame
    distance = torch.norm(des_pos_b, dim=1)
    mu = l = 2.0 * limit
    gate = torch.sigmoid(5.0 * (distance - mu)/l).clamp(0.0, 1.0)
    
    # Compute distance from default joint positions
    default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    distance_from_default = torch.abs(joint_pos - default_joint_pos)

    # Compute the constraint violation
    cstr_no_movement = gate.unsqueeze(1) * distance_from_default
        
    return cstr_no_movement


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

def cstr_foot_contact_force(env: ConstrainedManagerBasedRLEnv, limit: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize foot contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # Compute the norm of the forces of each foot
    f_norm = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1)
    f_norm_max = torch.max(f_norm, dim=1)[0]

    return f_norm_max - limit

def cstr_foot_stumble(env: ConstrainedManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, coeff: float = 4.0) -> torch.Tensor:
    """Penalize foot stumble as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # Extract the forces in different axis and compute the norm
    f_xy = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids, :2], dim=-1)
    f_z = torch.abs(net_contact_forces[:, :, sensor_cfg.body_ids, 2])
    # Get the max value in history
    f_xy = torch.max(f_xy, dim=1)[0]
    f_z = torch.max(f_z, dim=1)[0]
    # Compute the constraint
    cstr_stumble = f_xy - (f_z * coeff)

    return cstr_stumble

def cstr_foot_slip(env: ConstrainedManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 1.0) -> torch.Tensor:
    # Extract used quantities
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # Compute the norm of the forces of each foot
    f_norm = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1)
    f_norm_max = torch.max(f_norm, dim=1)[0]
    # Get the foot velocity
    asset: Articulation = env.scene[asset_cfg.name]
    foot_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids]
    foot_vel_xy_norm = torch.norm(foot_vel[:, :, :2], dim=-1)
    # Compute the constraint
    cstr_slip = (f_norm_max * foot_vel_xy_norm) - threshold

    return cstr_slip
