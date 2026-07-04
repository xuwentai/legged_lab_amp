from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_rough_env_cfg import (
    G1AmpRoughEnvCfg,
    G1AmpRoughEnvCfg_PLAY,
)

# The order must align with the retarget config file scripts/tools/retarget/config/g1_29dof.yaml
ANIMATION_TERM_NAME = "animation"


@configclass
class G1AmpFlatEnvCfg(G1AmpRoughEnvCfg):
    """Configuration for the G1 AMP environment on flat terrain.

    Inherits the full rough config and strips the rough-only pieces (mirrors the official
    IsaacLab velocity ``flat_env_cfg`` deriving from ``rough_env_cfg``): reverts the terrain
    back to an infinite plane and disables the terrain curriculum. This reproduces the
    original flat AMP behavior. Once milestone B adds height_scan / terrain_levels to the
    rough base, this class must also null them out (see TODOs).
    """

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------------------------------
        # Terrain (flat) — revert the generator terrain from the rough base
        # ------------------------------------------------------
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # On plane terrain env_origins.z == 0 and the reference motion's absolute xy is
        # valid ground, so restore the original DeepMimic-style reset (keep reference xy).
        self.events.reset_from_ref.params = {
            "animation": ANIMATION_TERM_NAME,
            "height_offset": 0.1,
            "align_xy_to_origin": False,
        }
        # base_height termination is valid again on flat ground (the rough base disabled
        # it because absolute world-z misfires as the generator ground rises). Rebuild the
        # term — it was set to None upstream, so we can't just mutate its params.
        self.terminations.base_height = DoneTerm(
            func=mdp.root_height_below_minimum, params={"minimum_height": 0.2}
        )
        # TODO(milestone B): when the rough base adds height_scanner + height_scan obs and
        #   the terrain_levels curriculum, null them here:
        #     self.scene.height_scanner = None
        #     self.observations.policy.height_scan = None
        #     self.observations.critic.height_scan = None
        #     self.curriculum.terrain_levels = None


@configclass
class G1AmpFlatEnvCfg_PLAY(G1AmpRoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
