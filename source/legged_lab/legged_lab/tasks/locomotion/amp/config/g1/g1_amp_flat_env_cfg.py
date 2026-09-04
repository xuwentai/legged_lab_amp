import copy

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import FlatPatchSamplingCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils.configclass import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp
import legged_lab.terrains as terrain_gen
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_rough_env_cfg import (
    G1AmpRoughEnvCfg,
    G1AmpRoughEnvCfg_PLAY,
)

# The order must align with the retarget config file scripts/tools/retarget/config/g1_29dof.yaml
ANIMATION_TERM_NAME = "animation"

FLAT_SLOPE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=60.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.PerlinPlaneTerrainCfg(
            proportion=0.60,
            noise_scale=0.00,
            noise_frequency=20,
            fractal_octaves=2,
            fractal_lacunarity=2.0,
            fractal_gain=0.25,
            centering=True,
            border_width=0.25,
            wall_prob=[0.0, 0.0, 0.0, 0.0],
            wall_height=5.0,
            wall_thickness=0.05,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                ),
            },
        ),
        "hf_pyramid_slope": terrain_gen.PerlinPyramidSlopedTerrainCfg(
            proportion=0.40,
            slope_range=(0.0, 0.7),
            platform_width=1.5,
            border_width=1.0,
            wall_prob=[0.3, 0.3, 0.3, 0.3],
            wall_height=5.0,
            wall_thickness=0.05,
            perlin_cfg=terrain_gen.PerlinPlaneTerrainCfg(
                noise_scale=0.00,
                noise_frequency=20,
                fractal_octaves=2,
                fractal_lacunarity=2.0,
                fractal_gain=0.25,
                centering=True,
            ),
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    num_patches=50,
                    patch_radius=[0.05, 0.10, 0.15, 0.20],
                    max_height_diff=0.05,
                ),
            },
        ),
        # "hf_pyramid_slope_in": terrain_gen.PerlinInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.10,
        #     slope_range=(0.0, 0.7),
        #     platform_width=1.5,
        #     border_width=1.0,
        #     wall_prob=[0.3, 0.3, 0.3, 0.3],
        #     wall_height=5.0,
        #     wall_thickness=0.05,
        #     perlin_cfg=terrain_gen.PerlinPlaneTerrainCfg(
        #         noise_scale=0.00,
        #         noise_frequency=20,
        #         fractal_octaves=2,
        #         fractal_lacunarity=2.0,
        #         fractal_gain=0.25,
        #         centering=True,
        #     ),
        #     flat_patch_sampling={
        #         "target": FlatPatchSamplingCfg(
        #             num_patches=50,
        #             patch_radius=[0.05, 0.10, 0.15, 0.20],
        #             max_height_diff=0.05,
        #         ),
        #     },
        # ),
    },
)


@configclass
class G1AmpFlatEnvCfg(G1AmpRoughEnvCfg):
    """Configuration for the G1 AMP environment on flat and sloped terrain."""

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------------------------------
        # Terrain (flat + slopes) — keep only flat/uphill/downhill tiles from the rough base
        # ------------------------------------------------------
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = copy.deepcopy(FLAT_SLOPE_TERRAINS_CFG)
        self.scene.terrain.max_init_terrain_level = 5
        self.rewards.lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-3.0)
        self.commands.base_velocity.reset_standing_from_default = True
        # Blind walking: keep the internal height scanner for terrain-relative base-height
        # checks, but do not feed the height map to policy/critic observations.
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # No stair/edge obstacles in this flat+slope preset.
        self.scene.foot_volume_points = None
        self.rewards.foot_stair_intrusion = None
        # Flat-only standing cleanup:
        # 1) keep ankle joints near their default pose when command is near zero to suppress
        #    in-place stepping during standstill;
        # 2) keep the feet level on contact to reduce heel-only contact / floating toes.
        self.rewards.joint_deviation_ankles = RewTerm(
            func=mdp.stand_still_joint_deviation_l1,
            weight=-0.2,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.08,
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"]
                ),
            },
        )
        self.rewards.feet_orientation = RewTerm(
            func=mdp.feet_orientation_l2,
            weight=-2.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            },
        )
        self.events.register_virtual_obstacles = None


@configclass
class G1AmpFlatEnvCfg_PLAY(G1AmpRoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        # Use the same flat+slope generator as training, but keep the PLAY grid small.
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = copy.deepcopy(FLAT_SLOPE_TERRAINS_CFG)
        self.scene.terrain.max_init_terrain_level = 0
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 10
            self.scene.terrain.terrain_generator.num_cols = 10
            self.scene.terrain.terrain_generator.curriculum = False
        self.scene.foot_volume_points = None
        self.rewards.foot_stair_intrusion = None
        self.rewards.lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-3.0)
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.rewards.joint_deviation_ankles = RewTerm(
            func=mdp.stand_still_joint_deviation_l1,
            weight=-0.2,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.08,
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"]
                ),
            },
        )
        self.rewards.feet_orientation = RewTerm(
            func=mdp.feet_orientation_l2,
            weight=-2.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            },
        )
        self.events.register_virtual_obstacles = None
        self.commands.base_velocity.reset_standing_from_default = True
