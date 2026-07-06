import math
import os

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab import LEGGED_LAB_ROOT_DIR

##
# Pre-defined configs
##
from legged_lab.assets.unitree import UNITREE_G1_29DOF_CFG
from legged_lab.tasks.locomotion.amp.amp_env_cfg import LocomotionAmpEnvCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

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
        weight=-1.0e-7,
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
        weight=0.5,
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

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class G1AmpRoughEnvCfg(LocomotionAmpEnvCfg):
    """Configuration for the G1 AMP environment on rough (generator) terrain.

    This is the base config for the G1 AMP task family; ``G1AmpFlatEnvCfg`` inherits
    from it and strips the rough-only pieces (mirrors the official IsaacLab velocity
    example, where ``flat_env_cfg`` derives from ``rough_env_cfg``).

    Uses ``ROUGH_TERRAINS_CFG`` generator terrain with a ``terrain_levels_vel`` difficulty
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
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        # terrain_levels curriculum (added below) enables generator.curriculum in the base
        # __post_init__; max_init_terrain_level=5 lets envs start spread across difficulties
        # and progress via the curriculum.
        self.scene.terrain.max_init_terrain_level = 5

        # ------------------------------------------------------
        # Height scanner + height_scan observation (rough only)
        # ------------------------------------------------------
        # RayCaster grid under the torso (G1's base body). Copied from the official velocity
        # example (GridPatternCfg 0.1m / 1.6x1.0m), with prim_path on torso_link.
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        # height_scan goes into policy + critic ONLY — never disc/disc_demo (the reference
        # motions have no terrain channel). Per-term history_length=3 (proprio terms use 5);
        # this relies on the policy/critic groups using per-term history (group history=None).
        self.observations.policy.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
            history_length=3,
            flatten_history_dim=True,
        )
        self.observations.critic.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            history_length=3,
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
            "C11_-_run_turn_left_90_stageii": 1.0,
            "C12_-_run_turn_left_45_stageii": 1.0,
            "C13_-_run_turn_left_135_stageii": 1.0,
            "C14_-_run_turn_right_90_stageii": 1.0,
            "C15_-_run_turn_right_45_stageii": 1.0,
            "C16_-_run_turn_right_135_stageii": 1.0,
            "C17_-_run_change_direction_stageii": 1.0,
            "C1_-_stand_to_run_stageii": 1.0,
            "C3_-_run_stageii": 1.0,
            "C4_-_run_to_walk_a_stageii": 1.0,
            "C5_-_walk_to_run_stageii": 1.0,
            "C6_-_stand_to_run_backwards_stageii": 1.0,
            "C8_-_run_backwards_to_stand_stageii": 1.0,
            "C9_-_run_backwards_turn_run_forward_stageii": 1.0,
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
        self.observations.disc_demo.ref_root_ang_vel_b.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_pos.params["animation"] = ANIMATION_TERM_NAME
        self.observations.disc_demo.ref_joint_vel.params["animation"] = ANIMATION_TERM_NAME

        # ------------------------------------------------------
        # Events
        # ------------------------------------------------------
        self.events.add_base_mass.params["asset_cfg"].body_names = "torso_link"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
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
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
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
        # torso off the ground) is now caught by the terrain-relative base_height term above, so
        # wrist/elbow contact bodies are no longer needed here — keep only torso and pelvis.
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "torso_link",
            "pelvis",
        ]


@configclass
class G1AmpRoughEnvCfg_PLAY(G1AmpRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5
        # spawn on the flattest tiles and shrink the terrain grid to save memory
        self.scene.terrain.max_init_terrain_level = 0
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.push_robot = None

        self.events.reset_from_ref = None
