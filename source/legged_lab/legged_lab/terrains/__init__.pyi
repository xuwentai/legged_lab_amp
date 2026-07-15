# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

__all__ = [
    # -- height-field family (light: numpy/scipy, no pxr/torch) --
    "PerlinCrossStoneTerrainCfg",
    "PerlinDiscreteObstaclesTerrainCfg",
    "PerlinGutterTerrainCfg",
    "PerlinInvertedPyramidSlopedTerrainCfg",
    "PerlinInvertedPyramidStairsTerrainCfg",
    "PerlinParapetTerrainCfg",
    "PerlinPlaneTerrainCfg",
    "PerlinPyramidSlopedTerrainCfg",
    "PerlinPyramidStairsTerrainCfg",
    "PerlinSlopeTerrainCfg",
    "PerlinSquareGapTerrainCfg",
    "PerlinStairsDownUpTerrainCfg",
    "PerlinStairsUpDownTerrainCfg",
    "PerlinSteppingStonesTerrainCfg",
    "PerlinTiltedRampTerrainCfg",
    "PerlinTiltTerrainCfg",
    "PerlinWaveTerrainCfg",
    # -- trimesh family (heavy: imports torch) --
    "MotionMatchedTerrainCfg",
    "PerlinMeshFloatingBoxTerrainCfg",
    "PerlinMeshRandomMultiBoxTerrainCfg",
    # -- importer + virtual obstacle support --
    "TerrainImporter",
    "TerrainImporterCfg",
    "EdgeCylinder",
    "EdgeCylinderCfg",
    "GreedyconcatEdgeCylinderCfg",
    "VirtualObstacleBase",
    "VirtualObstacleCfg",
    # -- custom generator (heavy: pulls isaaclab TerrainGenerator -> pxr/USD) --
    "FiledTerrainGenerator",
    "FiledTerrainGeneratorCfg",
]

from .height_field import (
    PerlinCrossStoneTerrainCfg,
    PerlinDiscreteObstaclesTerrainCfg,
    PerlinGutterTerrainCfg,
    PerlinInvertedPyramidSlopedTerrainCfg,
    PerlinInvertedPyramidStairsTerrainCfg,
    PerlinParapetTerrainCfg,
    PerlinPlaneTerrainCfg,
    PerlinPyramidSlopedTerrainCfg,
    PerlinPyramidStairsTerrainCfg,
    PerlinSlopeTerrainCfg,
    PerlinSquareGapTerrainCfg,
    PerlinStairsDownUpTerrainCfg,
    PerlinStairsUpDownTerrainCfg,
    PerlinSteppingStonesTerrainCfg,
    PerlinTiltedRampTerrainCfg,
    PerlinTiltTerrainCfg,
    PerlinWaveTerrainCfg,
)
from .trimesh import (
    MotionMatchedTerrainCfg,
    PerlinMeshFloatingBoxTerrainCfg,
    PerlinMeshRandomMultiBoxTerrainCfg,
)
from .terrain_generator import FiledTerrainGenerator
from .terrain_generator_cfg import FiledTerrainGeneratorCfg
from .terrain_importer import TerrainImporter
from .terrain_importer_cfg import TerrainImporterCfg
from .virtual_obstacle import (
    EdgeCylinder,
    EdgeCylinderCfg,
    GreedyconcatEdgeCylinderCfg,
    VirtualObstacleBase,
    VirtualObstacleCfg,
)
