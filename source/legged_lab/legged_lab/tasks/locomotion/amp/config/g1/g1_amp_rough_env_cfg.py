import math
import os

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.sensors.volume_points import Grid3dPointsGeneratorCfg, VolumePointsCfg

##
# Pre-defined configs
##
from legged_lab.assets.unitree import UNITREE_G1_29DOF_CFG
from legged_lab.tasks.locomotion.amp.ame_cfg import (
    G1_AME_HEIGHT_OFFSET,
    G1_AME_MAP_SCAN_HISTORY_LENGTH,
    G1_AME_SCAN_POSITION_OFFSET,
    G1_AME_SCAN_RESOLUTION,
    G1_AME_SCAN_SIZE,
)
from legged_lab.tasks.locomotion.amp.amp_env_cfg import LocomotionAmpEnvCfg
from legged_lab.terrains.config.rough import ROUGH_PERLIN_TERRAINS_CFG  # isort: skip

# The order must align with the retarget config file scripts/tools/retarget/config/g1_29dof.yaml
KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
]  # if changed here and symmetry is enabled, remember to update amp.mdp.symmetry.g1 as well!
ANIMATION_TERM_NAME = "animation"
AMP_NUM_STEPS = 4


@configclass
class G1AmpRewards:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )

    # -- penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])},
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.25e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )

    joint_deviation_hip = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-0.1,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"]),
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-0.05,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*_joint",
                ],
            ),
        },
    )
    joint_deviation_waist = RewTerm(
        func=mdp.stand_still_joint_deviation_l1,
        weight=-0.1,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names="waist_.*_joint"),
        },
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    #pengzhang
    # foot_stair_intrusion = RewTerm(
    #     func=mdp.volume_points_penetration_feet,
    #     weight=-1.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("foot_volume_points"),
    #         "tolerance": 0.0,
    #         "normal_penalty_scale": 1.0,
    #         "transition_penalty_scale": 2.0,
    #         "enable_terrain_foot_weights": True,
    #         "stairs_weight_min": 0.2,
    #         "stairs_weight_max": 1.0,
    #         "heel_weight_scale": 1.0,
    #     },
    # )

    # Soft collision penalty for arm links (wrists/elbows). Previously these were in the
    # base_contact termination, which forced the policy into weird arm postures to avoid ever
    # touching them (any contact => is_terminated => -200). Relaxed here to a per-step penalty
    # that discourages, but does not forbid, arm self/ground contact. Counts the number of arm
    # bodies whose net contact force exceeds the threshold (force magnitude ignored).
    undesired_arm_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_elbow_link", ".*_wrist_.*"]),
            "threshold": 1.0,
        },
    )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class G1AmpRoughEnvCfg(LocomotionAmpEnvCfg):
    """Configuration for the G1 AMP environment on rough (generator) terrain.

    This is the base config for the G1 AMP task family; ``G1AmpFlatEnvCfg`` inherits
    from it and strips the rough-only pieces (mirrors the official IsaacLab velocity
    example, where ``flat_env_cfg`` derives from ``rough_env_cfg``).

    Uses ``ROUGH_PERLIN_TERRAINS_CFG`` generator terrain (the official ``ROUGH_TERRAINS_CFG``
    with the foot-catching height-field tiles — ``random_rough`` and the two ``hf_pyramid_slope``
    tiles — swapped for smooth Perlin ports) with a ``terrain_levels_vel`` difficulty
    curriculum, a torso-mounted height scanner feeding ``height_scan`` into the policy and
    critic groups (never the discriminator — the reference motions have no terrain channel),
    and the ``reset_from_ref`` height-alignment fix (``align_xy_to_origin=True``) so the
    robot spawns aligned to the sub-terrain ground height without any raycast.
    """

    rewards: G1AmpRewards = G1AmpRewards()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ------------------------------------------------------
        # Terrain (rough) — override the plane terrain from AmpSceneCfg
        # ------------------------------------------------------
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_PERLIN_TERRAINS_CFG
        # terrain_levels curriculum (added below) enables generator.curriculum in the base
        # __post_init__; max_init_terrain_level=5 lets envs start spread across difficulties
        # and progress via the curriculum.
        self.scene.terrain.max_init_terrain_level = 5
        edge_obstacle = self.scene.terrain.virtual_obstacles.get("stair_edges")
        if edge_obstacle is not None and self.scene.terrain.terrain_generator is not None:
            terrain_generator = self.scene.terrain.terrain_generator
            edge_obstacle.terrain_tile_size = terrain_generator.size
            edge_obstacle.terrain_grid_shape = (terrain_generator.num_rows, terrain_generator.num_cols)
            edge_obstacle.sub_terrain_border_width = max(
                (
                    getattr(sub_terrain_cfg, "border_width", 0.0)
                    for sub_terrain_cfg in terrain_generator.sub_terrains.values()
                ),
                default=0.0,
            )

        # ------------------------------------------------------
        # Height scanner + height_scan observation (rough only)
        # ------------------------------------------------------
        # AME terrain map from the reference G1 setup: a 16x11 xyz grid covering
        # [0.0, 0.6] m forward and [-0.2, 0.2] m laterally from the torso.
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(pos=G1_AME_SCAN_POSITION_OFFSET),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=G1_AME_SCAN_RESOLUTION, size=G1_AME_SCAN_SIZE),
            debug_vis=True,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.foot_volume_points = VolumePointsCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
            points_generator=Grid3dPointsGeneratorCfg(
                x_min=-0.05,
                x_max=0.13,
                x_num=19,
                y_min=-0.03,
                y_max=0.03,
                y_num=7,
                z_min=-0.04,
                z_max=-0.02,
                z_num=3,
            ),
            debug_vis=True,
        )
        # The xyz map goes into policy + critic only. A 0.5 m vertical bias matches the
        # reference AME normalization; the discriminator never receives terrain data.
        self.observations.policy.height_scan = ObsTerm(
            func=mdp.height_scan_xyz,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": G1_AME_HEIGHT_OFFSET},
            noise=Unoise(n_min=-0.03, n_max=0.03),
            clip=(-1.0, 1.0),
            history_length=G1_AME_MAP_SCAN_HISTORY_LENGTH,
            flatten_history_dim=True,
        )
        self.observations.critic.height_scan = ObsTerm(
            func=mdp.height_scan_xyz,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": G1_AME_HEIGHT_OFFSET},
            clip=(-1.0, 1.0),
            history_length=G1_AME_MAP_SCAN_HISTORY_LENGTH,
            flatten_history_dim=True,
        )

        # ------------------------------------------------------
        # Curriculum — terrain difficulty progression
        # ------------------------------------------------------
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
        # The base __post_init__ toggles terrain_generator.curriculum from terrain_levels,
        # but it ran before we set terrain_levels above (super().__post_init__ runs first),
        # so enable it explicitly here now that the curriculum term exists.
        self.scene.terrain.terrain_generator.curriculum = True

        # ------------------------------------------------------
        # motion data
        # ------------------------------------------------------
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "g1_29dof", "amp", "walk_and_run"
        )
        # Pure-walking dataset: all C-series run clips are excluded so the discriminator only
        # ever rewards walking kinematics. MotionDataManager loads ONLY the files listed here
        # (extra .pkl files in the directory are ignored), so no data files need to be moved.
        self.motion_data.motion_dataset.motion_data_weights = {
            "B10_-__Walk_turn_left_45_stageii": 1.0,
            "B11_-__Walk_turn_left_135_stageii": 1.0,
            "B13_-__Walk_turn_right_90_stageii": 1.0,
            "B14_-__Walk_turn_right_45_t2_stageii": 1.0,
            "B15_-__Walk_turn_around_stageii": 1.0,
            "B22_-__side_step_left_stageii": 1.0,
            "B23_-__side_step_right_stageii": 1.0,
            "B4_-_Stand_to_Walk_backwards_stageii": 1.0,
            "B9_-__Walk_turn_left_90_stageii": 1.0,
            "Walk_B10_-_Walk_turn_left_45_stageii": 1.0,
            "Walk_B13_-_Walk_turn_right_45_stageii": 1.0,
            "Walk_B15_-_Walk_turn_around_stageii": 1.0,
            "Walk_B16_-_Walk_turn_change_stageii": 1.0,
            "Walk_B22_-_Side_step_left_stageii": 1.0,
            "Walk_B23_-_Side_step_right_stageii": 1.0,
            "Walk_B4_-_Stand_to_Walk_Back_stageii": 1.0,
        }

        # ------------------------------------------------------
        # animation
        # ------------------------------------------------------
        self.animation.animation.num_steps_to_use = AMP_NUM_STEPS

        # -----------------------------------------------------
        # Observations
        # -----------------------------------------------------
        self.terminal_obs_groups = ("disc",)

        # critic observations
        self.observations.critic.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(name="robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        }

        # discriminator observations
        self.observations.disc.history_length = AMP_NUM_STEPS

        # discriminator demonstration observations
        # ref_root_local_rot_tan_norm removed from disc_demo (paired with the disc-side
        # root_local_rot_tan_norm removal in amp_env_cfg) — no animation param to wire up.
        self.observations.disc_demo.ref_root_ang_vel_b.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_pos.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_vel.params["animation"] = ANIMATION_TERM_NAME

        # ------------------------------------------------------
        # Events
        # ------------------------------------------------------
        self.events.add_base_mass.params["asset_cfg"].body_names = "torso_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.register_virtual_obstacles = EventTerm(
            func=mdp.register_virtual_obstacle_to_sensor,
            mode="startup",
            params={"sensor_cfgs": SceneEntityCfg("foot_volume_points")},
        )
        # align_xy_to_origin=True: drop the reference motion's absolute xy so the robot
        # spawns at the sub-terrain origin, whose env_origins.z is the known ground height
        # on generator terrain — keeps root height aligned with no raycast. AMP does not
        # track root position, so the reference xy carries no useful signal. See the
        # reset_from_ref docstring for the full rationale.
        self.events.reset_from_ref.params = {
            "animation": ANIMATION_TERM_NAME,
            "height_offset": 0.1,
            "align_xy_to_origin": True,
        }

        # ------------------------------------------------------
        # Commands
        # ------------------------------------------------------
        # Ranges chosen to match what the walk-only AMP data can actually *sustain*, not
        # its instantaneous tails (body frame, measured offline on the reference clips):
        #   v_x lower bound -0.3: the raw p1 is ~-2.0, but those fast-backward frames are
        #     transient turn/pivot instants -- NO clip sustains straight (|w_z|<0.5) backward
        #     v_x<-1.0 for even 0.5 s. The only sustained straight backward walking is in the
        #     two B4 stand-to-walk-back clips, which sustain v_x ~ -0.3..-0.45. Backward
        #     commands are enabled but clamped to -0.3 (conservative end of the demonstrated
        #     range): anything faster has no demo coverage and the policy falls back to a
        #     discriminator-less backward-hop. (was -2.0, then 0.0)
        #   v_y (-0.5, 0.5): matches the side-step clips (p1/p99 ~ -0.48 / 0.53).
        #   w_z (-1.5, 1.5): covers the fast turn-around / 135deg clips (p1/p99 ~ -1.55 / 1.65).
        # The reset-aligned command in AmpVelocityCommand is clamped to these same ranges.
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

        # ------------------------------------------------------
        # terminations
        # ------------------------------------------------------
        # Base-height fall check: make it terrain-relative on rough ground by feeding it the
        # height_scanner RayCaster (the base config leaves sensor_cfg=None = absolute world-z,
        # which misfires as the generator ground rises). This measures how far the base sank
        # relative to the terrain underneath it.
        self.terminations.base_height.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
        # Fall-recovery fallback: terminate on torso/pelvis contact (matches the official G1
        # velocity rough config, which uses illegal_contact on torso_link). These links carry
        # large collision meshes high on the body, so normal gait never touches them — only a
        # real fall does. The "push-up pose" loophole (propping up on hands/forearms to keep the
        # torso off the ground) is caught by the terrain-relative base_height term above, so
        # wrist/elbow contact bodies are no longer terminated here — arm contact is instead
        # discouraged softly via the ``undesired_arm_contacts`` reward penalty, which avoids the
        # weird arm postures the policy learned to guarantee zero arm contact.
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "torso_link",
            "pelvis",
        ]


@configclass
class G1AmpRoughEnvCfg_PLAY(G1AmpRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()


        # self.scene.terrain.debug_vis = True

        # if self.scene.foot_volume_points is not None:
        #     self.scene.foot_volume_points.debug_vis = True

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5
        # spawn on the flattest tiles and shrink the terrain grid to save memory
        self.scene.terrain.max_init_terrain_level = 0
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 10
            self.scene.terrain.terrain_generator.num_cols = 10
            self.scene.terrain.terrain_generator.curriculum = False
            # self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion = 0.2
            # self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 0.2
            # self.scene.terrain.terrain_generator.sub_terrains["boxes"].proportion = 0.2
            # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].proportion = 0.0
            # self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.0
            # self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.0
            

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
        self.commands.base_velocity.ranges.heading =  (-math.pi, math.pi)

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.push_robot = None

        # Keep reset_from_ref active during play. It is the ONLY reset-mode event that actually
        # repositions the robot (writes root pose + joint state from a reference frame); the other
        # reset event (base_external_force_torque) applies zero force and is a no-op. IsaacLab's
        # _reset_idx / scene.reset() does not reposition the articulation by itself. If this were
        # None, a fired termination (e.g. base_height when a robot falls) would run an empty reset
        # and leave the robot lying in place — terminations would look like they never trigger.
