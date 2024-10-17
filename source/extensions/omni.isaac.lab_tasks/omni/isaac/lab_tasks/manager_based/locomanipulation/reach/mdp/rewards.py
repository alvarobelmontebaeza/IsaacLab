# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject, Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.utils.math import quat_rotate_inverse, yaw_quat, combine_frame_transforms, quat_error_magnitude, transform_points

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    return reward



def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

######## JOINT REWARDS ########
def joint_power_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Calculate the L2 norm of the joint power for a given asset in the environment.

    Args:
        env (ManagerBasedRLEnv): The environment containing the asset.
        asset_cfg (SceneEntityCfg, optional): Configuration for the asset. Defaults to SceneEntityCfg("robot").

    Returns:
        torch.Tensor: The L2 norm of the joint power for the specified asset.
    """

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(torques * joint_vel), dim=1)

def joint_power_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Calculate the L1 norm of the joint power for a given asset in the environment.

    The joint power is computed as the absolute value of the product of applied torques and joint velocities,
    summed across all specified joints.

    Args:
        env (ManagerBasedRLEnv): The environment containing the scene and assets.
        asset_cfg (SceneEntityCfg, optional): Configuration for the asset whose joint power is to be calculated.
            Defaults to SceneEntityCfg("robot").

    Returns:
        torch.Tensor: A tensor containing the L1 norm of the joint power for each instance in the batch.
    """

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(torques * joint_vel), dim=1)

######## ACTION REWARDS ########
def leg_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action[:, :12] - env.action_manager.prev_action[:, :12]), dim=1)

def arm_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action[:, 12:] - env.action_manager.prev_action[:, 12:]), dim=1)

######## MANIPULATION REWARDS ########

def _pose_to_keypoints(env: ManagerBasedRLEnv, pose: torch.Tensor, command_name: str) -> torch.Tensor:
        """
        Compute the 3D positions of the keypoints of a cube centered at the body's origin and transform them to the world frame.
        Args:
            env (ManagerBasedRLEnv): The environment instance containing the command manager.
            pose (torch.Tensor): A tensor containing the pose information with shape (N, 7), where N is the batch size.
                                 The first 3 elements are the position (x, y, z) and the next 4 elements are the quaternion (qx, qy, qz, qw).
            command_name (str): The name of the command to retrieve the cube size configuration.
            asset_cfg (SceneEntityCfg): Configuration of the scene entity.
        Returns:
            torch.Tensor: A tensor containing the 3D positions of the keypoints in the world frame with shape (N, 3, 3).
        """
        # compute the 3D positions of the 3 vertices of a cube centered at the body's origin
        # -- x, y, z
        # Initialize keypoints as verices of a cube
        pose_keypoints_b = torch.zeros((pose.shape[0], 3, 3), device=pose.device)
        cube_size = env.command_manager.get_term(command_name).cfg.cube_size
    
        pose_keypoints_b[:, 0, 0] = cube_size / 2.0
        pose_keypoints_b[:, 0, 1] = -cube_size / 2.0
        pose_keypoints_b[:, 0, 2] = cube_size / 2.0

        pose_keypoints_b[:, 1, 0] = -cube_size / 2.0
        pose_keypoints_b[:, 1, 1] = -cube_size / 2.0
        pose_keypoints_b[:, 1, 2] = cube_size / 2.0
        
        pose_keypoints_b[:, 2, 0] = cube_size / 2.0
        pose_keypoints_b[:, 2, 1] = cube_size / 2.0
        pose_keypoints_b[:, 2, 2] = cube_size / 2.0

        # Convert keypoints to world frame
        pose_keypoints_w = transform_points(
            pose_keypoints_b,
            pose[:, :3],
            pose[:, 3:],
        )

        return pose_keypoints_w

def pose_keypoints_command_error_exp(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, sigma: float = float(0.05)) -> torch.Tensor:
    """Reward tracking of the pose keypoints using exponential kernel.

    The function computes the pose error between the desired pose keypoints (from the command) and the
    current pose keypoints of the asset's body (in world frame) and maps it with an exponential kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current pose keypoints
    des_keypoints_w = env.command_manager.get_term(command_name).command_w
    curr_poses_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :7]  # type: ignore
    curr_keypoints_w = _pose_to_keypoints(env, curr_poses_w, command_name)
    # Iterate through each keypoint and compute the error
    for i in range(3):
        des_keypoint = des_keypoints_w[:, i]
        curr_keypoint = curr_keypoints_w[:, i]
        rew = torch.exp(-(1.0 / sigma) * torch.norm(curr_keypoint - des_keypoint, dim=1))
        if i == 0:
            total_rew = rew
        else:
            total_rew += rew
    
    return total_rew

def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)

def _get_sigmas(epsilon_pos, epsilon_orn):
    """
    Function to get sigma values for position and orientation based on position and orientation errors.
    
    Parameters:
    - epsilon_pos (torch.Tensor): Tensor of position errors, shape (N,).
    - epsilon_orn (torch.Tensor): Tensor of orientation errors, shape (N,).
    
    Returns:
    - sigma_pos (torch.Tensor): Tensor of sigma values for position, shape (N,).
    - sigma_orn (torch.Tensor): Tensor of sigma values for orientation, shape (N,).
    """
    # Define thresholds and corresponding sigma values for position
    pos_thresholds = [100.0, 1.0, 0.8, 0.5, 0.4, 0.2, 0.1]
    sigma_pos_values = [2.0, 1.0, 0.5, 0.1, 0.05, 0.01, 0.005]

    # Define thresholds and corresponding sigma values for orientation
    orn_thresholds = [100.0, 1.0, 0.8, 0.6, 0.2]
    sigma_orn_values = [8.0, 4.0, 2.0, 1.0, 0.5]

    # Initialize tensors for sigma values (default to the smallest value)
    sigma_pos = torch.full_like(epsilon_pos, sigma_pos_values[0])
    sigma_orn = torch.full_like(epsilon_orn, sigma_orn_values[0])

    # Assign sigma values for position errors based on thresholds
    for i, threshold in enumerate(pos_thresholds):
        mask = epsilon_pos < threshold
        sigma_pos[mask] = sigma_pos_values[i]

    # Assign sigma values for orientation errors based on thresholds
    for i, threshold in enumerate(orn_thresholds):
        mask = epsilon_orn < threshold
        sigma_orn[mask] = sigma_orn_values[i]

    return sigma_pos, sigma_orn

def pose_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current poses    
    des_pos_w, des_quat_w = command[:, :3], command[:, 3:]
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore

    # Compute the position and orientation errors
    pos_error = torch.sum(torch.square(curr_pos_w - des_pos_w), dim=1).sqrt()
    rot_error = quat_error_magnitude(curr_quat_w, des_quat_w)

    # Obtain the sigma values for position and orientation
    sigma_pos, sigma_rot = _get_sigmas(pos_error, rot_error)

    pos_rew = torch.exp(-pos_error / sigma_pos)
    rot_rew = torch.exp(-rot_error / sigma_rot)

    return pos_rew * rot_rew


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)
