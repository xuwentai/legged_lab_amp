from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sensors import SensorBaseCfg
from isaaclab.utils import configclass

from .points_generator_cfg import PointsGeneratorCfg
from .volume_points import VolumePoints

VOLUME_POINTS_VISUALIZER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/volumePoints",
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        "sphere_penetrated": sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    },
)


@configclass
class VolumePointsCfg(SensorBaseCfg):
    """Configuration for the volume-points sensor."""

    class_type: type = VolumePoints
    points_generator: PointsGeneratorCfg = MISSING
    visualizer_cfg: VisualizationMarkersCfg = VOLUME_POINTS_VISUALIZER_CFG

