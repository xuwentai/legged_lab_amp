from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.terrains import TerrainImporterCfg as TerrainImporterCfgBase
from isaaclab.utils import configclass

from .terrain_importer import TerrainImporter

if TYPE_CHECKING:
    from .virtual_obstacle import VirtualObstacleCfg


@configclass
class TerrainImporterCfg(TerrainImporterCfgBase):
    class_type: type = TerrainImporter
    virtual_obstacles: dict[str, "VirtualObstacleCfg"] = {}

