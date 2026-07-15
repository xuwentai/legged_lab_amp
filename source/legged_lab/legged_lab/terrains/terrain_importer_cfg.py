from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.terrains import TerrainImporterCfg as TerrainImporterCfgBase
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .virtual_obstacle import VirtualObstacleCfg

# {DIR} below resolves to ``legged_lab.terrains`` at __post_init__ time.
# The resulting ResolvableString is only resolved (→ importlib.import_module
# → isaaclab.terrains.utils → pxr) when the cfg is first used inside
# launch_simulation.
_TERRAIN_DIR = "{DIR}.terrain_importer"


@configclass
class TerrainImporterCfg(TerrainImporterCfgBase):
    class_type: type = _TERRAIN_DIR + ":TerrainImporter"  # type: ignore[assignment]
    virtual_obstacles: dict[str, "VirtualObstacleCfg"] = {}
