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
from omni.isaac.lab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_ASSETS_DIR
##
# Configuration
##


WX250s_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DIR}/wx250s.usd",
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
            ".*widow_waist": 0.0,
            ".*widow_shoulder": 0.0,
            ".*widow_elbow": 0.0,
            ".*widow_forearm_roll": 0.0,
            ".*widow_wrist_angle": 0.0,
            ".*widow_wrist_rotate": 0.0,
        },
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*widow_waist", ".*widow_shoulder", ".*widow_elbow", ".*widow_forearm_roll", ".*widow_wrist_angle", ".*widow_wrist_rotate"],
            effort_limit={
                ".*widow_waist": 4.0,
                ".*widow_shoulder": 8.0,
                ".*widow_elbow": 8.0,
                ".*widow_forearm_roll": 4.0,
                ".*widow_wrist_angle": 4.0,
                ".*widow_wrist_rotate": 1.5,
            },
            velocity_limit=3.14,
            stiffness=1000.0,
            damping=20.0,
        ),
    },
)
"""Configuration of wx250s arm using implicit actuator models."""
