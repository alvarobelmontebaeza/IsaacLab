# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import CommandTerm
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.utils.math import combine_frame_transforms, subtract_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique, quat_inv
import pytorch3d.transforms as pt3d

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

    from .commands_cfg import UniformSphericalPoseCommandCfg


class UniformSphericalPoseCommand(CommandTerm):
    """Command generator for generating pose commands uniformly.

    The command generator generates poses by sampling positions uniformly within specified
    regions in cartesian space. For orientation, it samples uniformly the euler angles
    (roll-pitch-yaw) and converts them into quaternion representation (w, x, y, z).

    The position and orientation commands are generated in the base frame of the robot, and not the
    simulation world frame. This means that users need to handle the transformation from the
    base frame to the simulation world frame themselves.

    .. caution::

        Sampling orientations uniformly is not strictly the same as sampling euler angles uniformly.
        This is because rotations are defined by 3D non-Euclidean space, and the mapping
        from euler angles to rotations is not one-to-one.

    """

    cfg: UniformSphericalPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformSphericalPoseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_spherical_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_spherical_b[:, 3] = 1.0
        self.pose_command_cartesian_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_cartesian_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_cartesian_b)
        self.curr_ee_target_b = torch.zeros_like(self.pose_command_spherical_b)
        # -- initial end-effector pose
        self.init_ee_pose_w = torch.zeros(self.num_envs, 7, device=self.device)
        self.init_ee_pose_b = torch.zeros_like(self.init_ee_pose_w)
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformSphericalPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """
        Returns the spherical pose command.
        This method retrieves the spherical pose command tensor, which is stored in the 
        `pose_command_spherical_b` attribute.
        Returns:
            torch.Tensor: The spherical pose command tensor.
        """
 
        return self.curr_ee_target_b
    
    """
    Implementation specific functions.
    """

    def spherical_to_cartesian(self, spherical_pose: torch.Tensor) -> torch.Tensor:
        cartesian_pose = torch.zeros_like(spherical_pose)
        cartesian_pose[:, 0] = spherical_pose[:, 0] * torch.sin(spherical_pose[:, 1]) * torch.cos(spherical_pose[:, 2])
        cartesian_pose[:, 1] = spherical_pose[:, 0] * torch.sin(spherical_pose[:, 1]) * torch.sin(spherical_pose[:, 2])
        cartesian_pose[:, 2] = spherical_pose[:, 0] * torch.cos(spherical_pose[:, 1])
        cartesian_pose[:, 3:] = spherical_pose[:, 3:]
        return cartesian_pose
    
    def cartesian_to_spherical(self, cartesian_pose: torch.Tensor) -> torch.Tensor:
        spherical_pose = torch.zeros_like(cartesian_pose)
        r = torch.norm(cartesian_pose[:, :3], dim=-1)
        spherical_pose[:, 0] = r
        spherical_pose[:, 1] = torch.acos(cartesian_pose[:, 2] / r)
        spherical_pose[:, 2] = torch.atan2(cartesian_pose[:, 1], cartesian_pose[:, 0])
        spherical_pose[:, 3:] = cartesian_pose[:, 3:]
        return spherical_pose
    
    def _update_metrics(self):
        # transform command from base frame to simulation world frame
        curr_ee_target_cart_b = self.spherical_to_cartesian(self.curr_ee_target_b)
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            curr_ee_target_cart_b[:, :3],
            curr_ee_target_cart_b[:, 3:],
        )
        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_state_w[:, self.body_idx, :3],
            self.robot.data.body_state_w[:, self.body_idx, 3:7],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new pose targets
        # -- spherical coordinates
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_spherical_b[env_ids, 0] = r.uniform_(*self.cfg.sphere_ranges.radius)
        self.pose_command_spherical_b[env_ids, 1] = r.uniform_(*self.cfg.sphere_ranges.polar)
        self.pose_command_spherical_b[env_ids, 2] = r.uniform_(*self.cfg.sphere_ranges.azimuth)
        # -- orientation
        euler_angles = torch.zeros_like(self.pose_command_spherical_b[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.sphere_ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.sphere_ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.sphere_ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        quat = quat_unique(quat) if self.cfg.make_quat_unique else quat
        self.pose_command_spherical_b[env_ids, 3:] = quat
        # Convert the sampled spherical coord to cartesian
        self.pose_command_cartesian_b[env_ids] = self.spherical_to_cartesian(self.pose_command_spherical_b[env_ids])
        self.curr_ee_target_b[env_ids, 3:] = self.pose_command_spherical_b[env_ids, 3:]

        self.init_ee_pose_w[env_ids, :] = self.robot.data.body_state_w[env_ids, self.body_idx, :7].clone()
        # Convert the initial end-effector pose from world frame to base frame
        self.init_ee_pose_b[env_ids, :3], self.init_ee_pose_b[env_ids, 3:] = subtract_frame_transforms(
            self.robot.data.root_pos_w[env_ids],
            self.robot.data.root_quat_w[env_ids],
            self.init_ee_pose_w[env_ids, :3],
            self.init_ee_pose_w[env_ids, 3:],
        )
        self.init_ee_pose_spherical_b = self.cartesian_to_spherical(self.init_ee_pose_b)

    def _update_command(self):
        # Get current time for the command (clipped to 0,1)
        t = torch.clip((self.cfg.resampling_time_range[1] - self.time_left) / self.cfg.resampling_time_range[1], 0, 1).reshape(-1, 1)
        # Interpolate between the current and the target command
        self.curr_ee_target_b[:, :3] = torch.lerp(self.init_ee_pose_spherical_b[:, :3], self.pose_command_spherical_b[:, :3], t)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current body pose
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        # -- current body pose
        body_pose_w = self.robot.data.body_state_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_pose_w[:, :3], body_pose_w[:, 3:7])
