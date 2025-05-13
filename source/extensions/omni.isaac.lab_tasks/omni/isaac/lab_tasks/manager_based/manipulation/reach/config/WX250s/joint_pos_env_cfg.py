# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from omni.isaac.lab.utils import configclass

import omni.isaac.lab_tasks.manager_based.manipulation.reach.mdp as mdp
from omni.isaac.lab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

##
# Pre-defined configs
##
from omni.isaac.lab_assets.interbotix import WX250s_CFG   # isort: skip


##
# Environment configuration
##


@configclass
class WX250sReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to WX250s
        self.scene.robot = WX250s_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["ee_gripper_link"]
        self.rewards.end_effector_position_tracking.weight = -0.
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["ee_gripper_link"]
        self.rewards.end_effector_position_tracking_fine_grained.weight = 0.
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["ee_gripper_link"]
        self.rewards.end_effector_orientation_tracking.weight = -0.

        self.rewards.pose_command_tracking.params["asset_cfg"].body_names = ["ee_gripper_link"]
        self.rewards.pose_command_tracking.weight = 5.0
        
        self.rewards.joint_vel.weight = 0.0
        self.rewards.joint_power.weight = -0.05
        self.rewards.action_rate.weight = -0.1

        self.curriculum.action_rate = None
        self.curriculum.joint_vel = None

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", 
            joint_names=[".*waist",".*shoulder",".*elbow",".*forearm_roll",".*wrist_angle",".*wrist_rotate"],
            scale=0.5,
            use_default_offset=True
        )
        # override observations
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = [".*waist", ".*shoulder", ".*elbow", ".*forearm_roll", ".*wrist_angle", ".*wrist_rotate"]
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = [".*waist", ".*shoulder", ".*elbow", ".*forearm_roll", ".*wrist_angle", ".*wrist_rotate"]

        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = "ee_gripper_link"
        self.commands.ee_pose.ranges.pitch = (-math.pi / 4, math.pi / 4)
        self.commands.ee_pose.ranges.yaw = (-math.pi / 4, math.pi / 4)
        self.commands.ee_pose.ranges.pos_x = (0.1, 0.3)
        self.commands.ee_pose.ranges.pos_y = (-0.2, 0.2)
        self.commands.ee_pose.ranges.pos_z = (0.1, 0.4)

        self.sim.dt = 0.005
        self.decimation = 4
        self.sim.render_interval = self.decimation



@configclass
class WX250sReachEnvCfg_PLAY(WX250sReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
