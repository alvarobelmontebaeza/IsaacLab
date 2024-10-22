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
from omni.isaac.lab_assets import K1_CFG  # isort: skip


##
# Environment configuration
##


@configclass
class K1ReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to K1
        self.scene.robot = K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["link_grasping_frame"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["link_grasping_frame"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["link_grasping_frame"]

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["K1.*"], scale=0.5, use_default_offset=True
        )
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = "link_grasping_frame"
        self.commands.ee_pose.ranges.pitch = (-math.pi / 4, math.pi / 4)
        self.commands.ee_pose.ranges.yaw = (-math.pi / 4, math.pi / 4)
        self.commands.ee_pose.ranges.pos_x = (0.2, 0.5)
        self.commands.ee_pose.ranges.pos_y = (-0.2, 0.2)
        self.commands.ee_pose.ranges.pos_z = (0.1, 0.4)



@configclass
class K1ReachEnvCfg_PLAY(K1ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
