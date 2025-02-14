# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.manager_based.locomanipulation.reach.reach_env_cfg import LocomanipulationReachRoughEnvCfg
from omni.isaac.lab_tasks.manager_based.locomanipulation.reach.constrained_reach_env_cfg import CstrLocomanipulationReachRoughEnvCfg
##
# Pre-defined configs
##
from omni.isaac.lab_assets.unitree import WIDOWGO2_CFG, UNITREE_GO1_CFG  # isort: skip


@configclass
class WidowGo2ReachRoughEnvCfg(LocomanipulationReachRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = WIDOWGO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # COMMANDS CFG ADJUSTEMENTS
        self.commands.ee_pose.body_name = ".*wx250s_ee_gripper_link"

        # ACTIONS CFG ADJUSTEMENTS
        self.actions.arm_joint_pos.joint_names = [
            ".*widow_waist",
            ".*widow_shoulder",
            ".*widow_elbow",
            ".*widow_forearm_roll",
            ".*widow_wrist_angle",
            ".*widow_wrist_rotate",
        ]
        # reduce action scale
        self.actions.leg_joint_pos.scale = 0.25

        # event
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 1.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.add_arm_payload.params["mass_distribution_params"] = (0.0, 0.1)
        self.events.add_arm_payload.params["asset_cfg"].body_names = ".*wx250s_ee_gripper_link"
        
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # REWARDS CFG ADJUSTEMENTS
        self.rewards.arm_dof_power.params["asset_cfg"].joint_names = [
            ".*widow_waist",
            ".*widow_shoulder",
            ".*widow_elbow",
            ".*widow_forearm_roll",
            ".*widow_wrist_angle",
            ".*widow_wrist_rotate",
        ]
        self.rewards.pose_tracking.params["asset_cfg"].body_names = [".*wx250s_ee_gripper_link"]


@configclass
class WidowGo2ReachRoughEnvCfg_PLAY(WidowGo2ReachRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

#### CONSTRAINED CONFIGS ####
@configclass
class CstrWidowGo2ReachRoughEnvCfg(CstrLocomanipulationReachRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = WIDOWGO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # COMMANDS CFG ADJUSTEMENTS
        self.commands.ee_pose.body_name = ".*wx250s_ee_gripper_link"

        # ACTIONS CFG ADJUSTEMENTS
        self.actions.arm_joint_pos.joint_names = [
            ".*widow_waist",
            ".*widow_shoulder",
            ".*widow_elbow",
            ".*widow_forearm_roll",
            ".*widow_wrist_angle",
            ".*widow_wrist_rotate",
        ]
        # reduce action scale
        self.actions.leg_joint_pos.scale = 0.25

        # event
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 1.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.add_arm_payload.params["mass_distribution_params"] = (0.0, 0.1)
        self.events.add_arm_payload.params["asset_cfg"].body_names = ".*wx250s_ee_gripper_link"
        
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # REWARDS CFG ADJUSTEMENTS
        self.rewards.arm_dof_power.params["asset_cfg"].joint_names = [
            ".*widow_waist",
            ".*widow_shoulder",
            ".*widow_elbow",
            ".*widow_forearm_roll",
            ".*widow_wrist_angle",
            ".*widow_wrist_rotate",
        ]
        self.rewards.pose_tracking.params["asset_cfg"].body_names = [".*wx250s_ee_gripper_link"]


@configclass
class CstrWidowGo2ReachRoughEnvCfg_PLAY(CstrWidowGo2ReachRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
