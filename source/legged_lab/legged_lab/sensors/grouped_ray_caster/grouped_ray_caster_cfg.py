from isaaclab.sensors.ray_caster import RayCasterCfg
from isaaclab.utils.configclass import configclass

from .grouped_ray_caster import GroupedRayCaster


@configclass
class GroupedRayCasterCfg(RayCasterCfg):
    """Ray caster that can hit global meshes and same-env dynamic robot meshes."""

    class_type: type = GroupedRayCaster

    min_distance: float = 0.0
    """Ignore hits closer than this distance from the ray origin."""

    aux_mesh_and_link_names: dict[str, str | None] = {}
    """Extra mesh names to search under a link Xform when mesh and link names differ."""
