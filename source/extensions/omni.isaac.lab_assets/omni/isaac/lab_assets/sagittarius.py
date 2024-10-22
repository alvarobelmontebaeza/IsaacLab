# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots.

The following configuration parameters are available:

* :obj:`UR10_CFG`: The UR10 arm without a gripper.

Reference: https://github.com/ros-industrial/universal_robot
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_ASSETS_DIR
##
# Configuration
##


K1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DIR}/K1.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "K1_shoulder_pan_joint": 0.0,
            "K1_shoulder_lift_joint": 0.0,
            "K1_elbow_joint": 0.0,
            "K1_wrist_1_joint": 0.0,
            "K1_wrist_2_joint": 0.0,
            "K1_wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "arm_low_torque": ImplicitActuatorCfg(
            joint_names_expr=["K1_shoulder_pan_joint", "K1_wrist_1_joint", "K1_wrist_2_joint", "K1_wrist_3_joint"],
            effort_limit=4.41,
            velocity_limit=3.14,
            stiffness=5.0,
            damping=1.0,
        ),
        "arm_high_torque": ImplicitActuatorCfg(
            joint_names_expr=["K1_shoulder_lift_joint", "K1_elbow_joint"],
            effort_limit=8.33,
            velocity_limit=3.14,
            stiffness=5.0,
            damping=1.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of K1 arm using implicit actuator models."""
