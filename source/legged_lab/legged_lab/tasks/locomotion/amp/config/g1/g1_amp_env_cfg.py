"""Backward-compatibility shim.

The G1 AMP task was split into rough (base) and flat (derived) configs to mirror the
official IsaacLab velocity example. The original flat behavior now lives in
``g1_amp_flat_env_cfg``. These aliases keep the old ``G1AmpEnvCfg`` /
``G1AmpEnvCfg_PLAY`` names — and the ``LeggedLab-Isaac-AMP-G1-v0`` task id that
references them — pointing at the flat config, so existing scripts keep working.

Prefer importing ``G1AmpFlatEnvCfg`` / ``G1AmpRoughEnvCfg`` directly in new code.
"""

from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_flat_env_cfg import (
    G1AmpFlatEnvCfg as G1AmpEnvCfg,
)
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_flat_env_cfg import (
    G1AmpFlatEnvCfg_PLAY as G1AmpEnvCfg_PLAY,
)

__all__ = ["G1AmpEnvCfg", "G1AmpEnvCfg_PLAY"]
