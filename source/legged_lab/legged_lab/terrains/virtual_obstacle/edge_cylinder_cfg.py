from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

from .edge_cylinder import EdgeCylinder
from .virtual_obstacle_base import VirtualObstacleCfg


@configclass
class EdgeCylinderCfg(VirtualObstacleCfg):
    """Inflated-cylinder virtual obstacle generated from stair mesh edges."""

    class_type: type = EdgeCylinder
    angle_threshold: float = 70.0
    min_face_height_diff: float = 0.01
    strict_step_edges: bool = True
    horizontal_normal_z: float = 0.9
    vertical_normal_z: float = 0.1
    max_edge_z_span: float = 0.005
    exclude_tile_border_edges: bool = True
    terrain_tile_size: tuple[float, float] | None = None
    terrain_grid_shape: tuple[int, int] | None = None
    sub_terrain_border_width: float = 0.0
    tile_border_margin: float = 0.03
    cylinder_radius: float = 0.2
    transition_edge_radius: float = 0.05
    transition_edge_margin: float = 0.04
    num_grid_cells: int = 64**3
    visualizer: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/edgeMarkers",
        markers={
            "cylinder": sim_utils.CylinderCfg(
                radius=1.0,
                height=1.0,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.9), opacity=0.2),
            )
        },
    )


@configclass
class GreedyconcatEdgeCylinderCfg(EdgeCylinderCfg):
    """Compatibility alias for configs copied from the reference repository."""

    class_type: type = EdgeCylinder
    adjacent_angle_threshold: float = 30.0
    point_distance_threshold: float = 0.06
    min_points: int = 5

