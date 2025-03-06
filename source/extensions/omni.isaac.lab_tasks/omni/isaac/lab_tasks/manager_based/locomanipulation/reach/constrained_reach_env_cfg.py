# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import omni.isaac.lab_tasks.manager_based.locomanipulation.reach.mdp as mdp

# For constraints
from omni.isaac.lab.envs import ConstrainedManagerBasedRLEnvCfg
from omni.isaac.lab.managers import ConstraintTermCfg as CstrTerm

##
# Pre-defined configs
##
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""    
    ee_pose = mdp.UniformPoseWorldCommandCfg(
        asset_name="robot",
        body_name=MISSING, # Virtual EE frame at the end of the robot arm
        resampling_time_range=(4.0, 4.0),
        ranges=mdp.UniformPoseWorldCommandCfg.Ranges(
            pos_x=(0.2, 1.2),
            pos_y=(-0.5, 0.5),
            pos_z=(-0.3, 0.3),
            roll=(0.0, 0.0),
            pitch= (-math.pi * 0.25, math.pi * 0.5),
            yaw=(-math.pi * 0.5, math.pi * 0.5),
        ),
        debug_vis=True,
    )    
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # Output target joint positions for both the arm and the leg joints
    leg_joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*hip_joint", ".*thigh_joint", ".*calf_joint"], scale=0.25, use_default_offset=True)
    arm_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=MISSING,
        scale=0.5,
        use_default_offset=True,
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        #base_pos_w = ObsTerm(func=mdp.root_pos_w, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_rotation = ObsTerm(func=mdp.root_quat_w, noise=Unoise(n_min=-0.1, n_max=0.1))
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-1.5, n_max=1.5))
        feet_contacts = ObsTerm(func=mdp.feet_contacts, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot")})
        actions = ObsTerm(func=mdp.last_action)
        '''
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        '''
        target_pose = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        # target_pose = ObsTerm(func=mdp.pose_command_cartesian_6d_rotation, params={"command_name": "ee_pose"})
        # current_pose = ObsTerm(func=mdp.body_pose_cartesian_quaternion_base_frame, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*link_grasping_frame")})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    # Randomize the mass of a payload carried by the arm from 0 to max mapyload
    add_arm_payload = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ee_payload"),
            "mass_distribution_params": (0.0, 0.1),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 4.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    pose_tracking = RewTerm(
        func=mdp.pose_command_error_exp_base_frame_radius,
        weight=2.5,
        params={"command_name": "ee_pose", "radius": 0.4, "asset_cfg": SceneEntityCfg("robot", body_names=[".*ee_link"]), "sigmas": "adaptive"}
    )
    leg_low_power = RewTerm(func=mdp.low_power, weight=0.45, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_joint", ".*thigh_joint", ".*calf_joint"]), "max_power": 900.0})
    arm_low_power = RewTerm(func=mdp.low_power, weight=0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*waist", ".*shoulder", ".*elbow", ".*forearm_roll", ".*wrist_angle", ".*wrist_rotate"]), "max_power": 60.0})
    action_rate = RewTerm(func=mdp.action_rate_regularization, weight=0.2)
    # alive = RewTerm(func=mdp.is_alive, weight=0.05)
    # -- penalties
    # arm_dof_power = RewTerm(func=mdp.joint_power_l2, weight=-5e-3, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*arm_joints"])})
    # legs_dof_power = RewTerm(func=mdp.joint_power_l2, weight=-7.5e-6, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_joint", ".*thigh_joint", ".*calf_joint"])})
    # foot_force_l2 = RewTerm(func=mdp.foot_force_z, weight=-1e-4, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot")})
    # base_lin_acc = RewTerm(func=mdp.body_lin_acc_l2, weight=-0.0005, params={"asset_cfg": SceneEntityCfg("robot", body_names=["base"])})
    # base_ang_acc = RewTerm(func=mdp.body_ang_acc_l2, weight=-0.0005, params={"asset_cfg": SceneEntityCfg("robot", body_names=["base"])})
    # dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-4)
    # dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-6)
    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    # leg_action_rate_l2 = RewTerm(func=mdp.leg_action_rate_l2, weight=-0.01)
    # arm_action_rate_l2 = RewTerm(func=mdp.arm_action_rate_l2, weight=-0.02)
    # hip_action_l2 = RewTerm(func=mdp.hip_action_l2, weight=-0.1)
    # even_mass_usage = RewTerm(func=mdp.feet_force_std, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot")})
    
    # -- constraints
    # root_height = RewTerm(func=mdp.root_height_below_minimum, weight=-1.0, params={"minimum_height": 0.25})
    # -- optional penalties
    # dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    # base_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5, params={"asset_cfg": SceneEntityCfg("robot", body_names="base")})
    # default_dof_pos = RewTerm(func=mdp.joint_deviation_l1, weight=-0.05, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_joint", ".*thigh_joint", ".*calf_joint"])})

@configclass
class ConstraintsCfg:
    """Constraint terms for the MDP."""

    # -- SOFT CONSTRAINTS
    # -- Legs
    cstr_leg_joint_pos_upper_limits = CstrTerm(init_max_p=0.05, final_max_p=0.9, func=mdp.cstr_joint_pos_upper_limits, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_joint", ".*thigh_joint", ".*calf_joint"])})
    cstr_leg_joint_pos_lower_limits = CstrTerm(init_max_p=0.05, final_max_p=0.9, func=mdp.cstr_joint_pos_lower_limits, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_joint", ".*thigh_joint", ".*calf_joint"])})
    cstr_leg_joint_vel_limits = CstrTerm(init_max_p=0.05, final_max_p=0.9, func=mdp.cstr_joint_vel_limits, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_joint", ".*thigh_joint", ".*calf_joint"]), 
        "limits": 30.0})
    cstr_leg_joint_torque_limits = CstrTerm(init_max_p=0.05, final_max_p=0.25, func=mdp.cstr_joint_torque_limits, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_joint", ".*thigh_joint"]), 
        "limits": 30.0})
    cstr_leg_calf_joint_torque_limits = CstrTerm(init_max_p=0.05, final_max_p=0.25, func=mdp.cstr_joint_torque_limits, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*calf_joint"]), 
        "limits": 40.0})
    cstr_joint_leg_acc_limits = CstrTerm(init_max_p=0.05, final_max_p=0.25, func=mdp.cstr_joint_acc_limits, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_joint", ".*thigh_joint", ".*calf_joint"]), 
        "limits": 2000.0})
    
    # -- Arm
    cstr_arm_joint_pos_upper_limits = CstrTerm(init_max_p=0.05, final_max_p=0.9, func=mdp.cstr_joint_pos_upper_limits, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*waist", ".*shoulder", ".*elbow", ".*forearm_roll", ".*wrist_angle", ".*wrist_rotate"])})
    cstr_arm_joint_pos_lower_limits = CstrTerm(init_max_p=0.05, final_max_p=0.9, func=mdp.cstr_joint_pos_lower_limits, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*waist", ".*shoulder", ".*elbow", ".*forearm_roll", ".*wrist_angle", ".*wrist_rotate"])})
    cstr_arm_joint_vel_limits = CstrTerm(init_max_p=0.05, final_max_p=0.9, func=mdp.cstr_joint_vel_limits, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*widow_waist", ".*widow_shoulder", ".*widow_elbow", ".*widow_forearm_roll", ".*widow_wrist_angle", ".*widow_wrist_rotate"]), 
        "limits": 3.14})
    cstr_arm_joint_torque_limits_waist = CstrTerm(init_max_p=0.05, final_max_p=0.25, func=mdp.cstr_joint_torque_limits, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*widow_waist"]), 
        "limits": 10.0,
    })
    cstr_arm_joint_torque_limits_shoulder = CstrTerm(init_max_p=0.05, final_max_p=0.25, func=mdp.cstr_joint_torque_limits, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*widow_shoulder"]), 
        "limits": 20.0,
    })
    cstr_arm_joint_torque_limits_elbow = CstrTerm(init_max_p=0.05, final_max_p=0.25, func=mdp.cstr_joint_torque_limits, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*widow_elbow"]), 
        "limits": 15.0,
    })
    cstr_arm_joint_torque_limits_forearm_roll = CstrTerm(init_max_p=0.05, final_max_p=0.25, func=mdp.cstr_joint_torque_limits, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*widow_forearm_roll"]), 
        "limits": 2.0,
    })
    cstr_arm_joint_torque_limits_wrist_angle = CstrTerm(init_max_p=0.05, final_max_p=0.25, func=mdp.cstr_joint_torque_limits, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*widow_wrist_angle"]), 
        "limits": 5.0,
    })
    cstr_arm_joint_torque_limits_wrist_rotate = CstrTerm(init_max_p=0.05, final_max_p=0.25, func=mdp.cstr_joint_torque_limits, params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=[".*widow_wrist_rotate"]), 
        "limits": 1.0,
    })
    
    # -- Actions
    cstr_leg_action_limits = CstrTerm(init_max_p=0.05, final_max_p=0.9, func=mdp.cstr_action_limits, params={"action_name": "leg_joint_pos", "asset_cfg": SceneEntityCfg("robot")})
    cstr_arm_action_limits = CstrTerm(init_max_p=0.05, final_max_p=0.9, func=mdp.cstr_action_limits, params={"action_name": "arm_joint_pos", "asset_cfg": SceneEntityCfg("robot")})
    # cstr_action_rate = CstrTerm(init_max_p=0.05, final_max_p=0.25, func=mdp.cstr_action_rate, params={"limit": 150.0})

    # -- HARD CONSTRAINTS
    cstr_base_knee_contact = CstrTerm(init_max_p=1.0, final_max_p=1.0, func=mdp.cstr_undesired_contacts,
                                      params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*base", ".*Head.*", ".*hip", ".*thigh", ".*calf"])})
    cstr_arm_contact = CstrTerm(init_max_p=1.0, final_max_p=1.0, func=mdp.cstr_undesired_contacts,
                                params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*arm_links"])})
    cstr_foot_contact_force = CstrTerm(init_max_p=1.0, final_max_p=1.0, func=mdp.cstr_foot_contact_force,
                                       params={"limit": 900.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*foot"])})
    cstr_min_base_height = CstrTerm(init_max_p=1.0, final_max_p=1.0, func=mdp.cstr_min_base_height,
                                    params={"min_height": 0.15, "asset_cfg": SceneEntityCfg("robot", body_names=[".*base"])})
    # -- STYLE CONSTRAINTS
    cstr_joint_deviation = CstrTerm(init_max_p=0.05, final_max_p=0.25, func=mdp.cstr_joint_deviation,
                                    params={"limit": 0.2, "asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_joint"])})
    cstr_base_orientation = CstrTerm(init_max_p=0.05, final_max_p=0.25, func=mdp.cstr_flat_orientation,
                                     params={"limit": 0.2, "asset_cfg": SceneEntityCfg("robot", body_names=[".*base"])})
    cstr_body_orientation_pitch = CstrTerm(init_max_p=0.05, final_max_p=0.25, func=mdp.cstr_body_orientation_axis,
                                          params={"axis": "y", "limit": 0.1, "asset_cfg": SceneEntityCfg("robot", body_names=[".*base"])})
    # cstr_no_move = CstrTerm(init_max_p=0.05, final_max_p=0.25, func=mdp.cstr_no_movement, params={"limit": 0.3, "command_name": "ee_pose", "asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_joint", ".*thigh_joint", ".*calf_joint"])})
    cstr_foot_stumble = CstrTerm(init_max_p=0.05, final_max_p=0.25, func=mdp.cstr_foot_stumble,
                                 params={"coeff": 2.0, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*foot"])})
    cstr_foot_slip = CstrTerm(init_max_p=0.05, final_max_p=0.9, func=mdp.cstr_foot_slip,
                              params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*foot"]), "asset_cfg": SceneEntityCfg("robot", body_names=[".*foot"])})
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    '''
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*base", ".*hip", ".*thigh"]), "threshold": 1.0},
    )
    '''
    # bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.5, "asset_cfg": SceneEntityCfg("robot", body_names=["base"])})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    constraint_max_probs = CurrTerm(func=mdp.modify_constraint_max_prob, params={"max_p_epoch_factor": 0.6}) # must be in (0,1)
    #TODO: Add curriculum for the arm pose targets


##
# Environment configuration
##


@configclass
class CstrLocomanipulationReachRoughEnvCfg(ConstrainedManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    constraints: ConstraintsCfg = ConstraintsCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 8.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
