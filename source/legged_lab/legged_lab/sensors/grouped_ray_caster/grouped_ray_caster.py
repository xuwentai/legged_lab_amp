from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
import re
import sys

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import numpy as np
import torch
import warp as wp
from isaaclab.sensors.ray_caster import RayCaster
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils.warp import convert_to_warp_mesh
from legged_lab.utils.warp.raycast import raycast_mesh_grouped

if TYPE_CHECKING:
    from .grouped_ray_caster_cfg import GroupedRayCasterCfg

sys.modules.setdefault("isaaclab.sensors.ray_caster.grouped_ray_caster", sys.modules[__name__])


def _isaacsim_modules():
    import isaacsim.core.utils.prims as prim_utils
    import omni.log
    import omni.physics.tensors.impl.api as physx
    from isaacsim.core.prims import XFormPrim
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    return prim_utils, omni.log, physx, XFormPrim, Gf, Usd, UsdGeom, UsdPhysics


class GroupedRayCaster(RayCaster):
    """Ray caster that updates dynamic meshes and filters hits by environment.

    This lets a height scanner hit global terrain plus robot body meshes from
    its own environment, without seeing bodies from neighboring envs.
    """

    # IsaacLab 3.x factory classes require sensor subclasses to be declared as
    # part of the isaaclab package. The implementation still lives in legged_lab.
    __module__ = "isaaclab.sensors.ray_caster.grouped_ray_caster"

    cfg: GroupedRayCasterCfg

    def __init__(self, cfg: GroupedRayCasterCfg):
        self.meshes: dict[str, list[wp.Mesh]] = {}
        self.mesh_prototype_ids: list[int] | torch.Tensor = []
        self.mesh_collision_groups: list[int] | torch.Tensor = []
        self.rigid_body_mesh_transform_segments: dict[str, slice] = {}
        self.rigid_body_views: dict[str, object] = {}
        super().__init__(cfg)

    def reset(self, env_ids: Sequence[int] | None = None):
        if not hasattr(self, "drift"):
            self.drift = torch.zeros(self._view.count, 3, device=self.device)
        super().reset(env_ids)

    def _get_rigid_body_view(
        self, env_prim_path_expr: str, matched_leaf_names: list[str]
    ):
        prim_utils, _, _, _, _, _, _, UsdPhysics = _isaacsim_modules()
        parent_prim_paths = sim_utils.find_matching_prim_paths(env_prim_path_expr)
        body_names = []
        for potential_body_name in matched_leaf_names:
            prim = prim_utils.get_prim_at_path(parent_prim_paths[0] + "/" + potential_body_name)
            if prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                body_names.append(prim.GetPath().pathString.rsplit("/", 1)[-1])

        if body_names:
            body_names_regex = r"(" + "|".join(body_names) + r")"
            body_names_regex = f"{env_prim_path_expr}/{body_names_regex}"
            body_names_glob = body_names_regex.replace(".*", "*")
            return self._physics_sim_view.create_rigid_body_view(body_names_glob)

        prim = prim_utils.get_prim_at_path(parent_prim_paths[0])
        if prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return self._physics_sim_view.create_rigid_body_view(env_prim_path_expr.replace(".*", "*"))
        return None

    def _get_collision_groups_by_env_str(
        self,
        rigid_body_view,
        env_name_pattern: str = r"/World/envs/env_\d+",
    ) -> list[int]:
        collision_groups = []
        for prim_path in rigid_body_view.prim_paths:
            match = re.search(env_name_pattern, prim_path)
            if match:
                collision_groups.append(int(match.group(0).split("_")[-1]))
            else:
                _, omni_log, _, _, _, _, _, _ = _isaacsim_modules()
                omni_log.warn(f"Could not match environment name pattern in {prim_path}.")
                collision_groups.append(-1)
        return collision_groups

    def _get_merged_mesh_from_xform_prim(self, xform_prim_path: str) -> tuple[np.ndarray | None, np.ndarray | None]:
        prim_utils, _, _, _, Gf, Usd, UsdGeom, _ = _isaacsim_modules()
        points, indices = [], []
        indices_offset = 0
        xform_leaf = xform_prim_path.split("/")[-1]
        aux_mesh_name_search_list = [xform_leaf] + list(self.cfg.aux_mesh_and_link_names.keys())

        for prim_leaf_name in aux_mesh_name_search_list:
            mesh_prim_path = "/".join([xform_prim_path, "visuals", prim_leaf_name, "mesh"])
            mesh_prim = prim_utils.get_prim_at_path(mesh_prim_path)
            if not mesh_prim.IsValid():
                continue
            mesh = UsdGeom.Mesh(mesh_prim)
            local_points = np.asarray(mesh.GetPointsAttr().Get())
            if len(local_points) == 0:
                continue
            local_indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())

            local_transform = Gf.Matrix4d(1.0)
            fixed_link_name = self.cfg.aux_mesh_and_link_names.get(prim_leaf_name, None)
            if fixed_link_name is not None:
                fixed_link_prim = prim_utils.get_prim_at_path("/".join([xform_prim_path, fixed_link_name]))
                if fixed_link_prim.IsValid():
                    xformable = UsdGeom.Xformable(fixed_link_prim)
                    for op in xformable.GetOrderedXformOps():
                        local_transform = local_transform * op.GetOpTransform(Usd.TimeCode.Default())

            transformed_points = np.array(
                [local_transform.Transform(Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])))[:3] for p in local_points]
            )
            points.append(transformed_points)
            indices.append(local_indices + indices_offset)
            indices_offset += len(transformed_points)

        if not points:
            return None, None
        return np.concatenate(points, axis=0), np.concatenate(indices, axis=0)

    def _initialize_warp_meshes(self):
        _, omni_log, _, _, _, _, UsdGeom, _ = _isaacsim_modules()
        for mesh_prim_path_regex in self.cfg.mesh_prim_paths:
            env_prim_path_expr, leaf_pattern = mesh_prim_path_regex.rsplit("/", 1)
            matched_mesh_prim_paths = sim_utils.find_matching_prim_paths(env_prim_path_expr)
            assert matched_mesh_prim_paths, f"No matching mesh prim paths found for {env_prim_path_expr}."
            mesh_prims = sim_utils.get_all_matching_child_prims(
                matched_mesh_prim_paths[0],
                lambda prim: (
                    (prim.GetTypeName() in ("Plane", "Mesh", "Xform"))
                    and re.search("/".join([matched_mesh_prim_paths[0], leaf_pattern]), prim.GetPath().pathString)
                    is not None
                ),
            )

            wp_meshes = []
            wp_mesh_names = []
            wp_mesh_ids = []
            for mesh_prim in mesh_prims:
                mesh_name = mesh_prim.GetPath().pathString.rsplit("/", 1)[-1]
                if mesh_prim.GetTypeName() == "Xform":
                    points, indices = self._get_merged_mesh_from_xform_prim(mesh_prim.GetPath().pathString)
                    if points is None:
                        continue
                    wp_mesh = convert_to_warp_mesh(points, indices, device=self.device)
                elif mesh_prim.GetTypeName() == "Mesh":
                    mesh = UsdGeom.Mesh(mesh_prim)
                    points = np.asarray(mesh.GetPointsAttr().Get())
                    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
                    wp_mesh = convert_to_warp_mesh(points, indices, device=self.device)
                elif mesh_prim.GetTypeName() == "Plane":
                    mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
                    wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=self.device)
                else:
                    omni_log.warn(f"Unsupported mesh type: {mesh_prim.GetTypeName()} in {mesh_prim.GetPath()}.")
                    continue

                wp_mesh_names.append(mesh_name)
                wp_meshes.append(wp_mesh)
                wp_mesh_ids.append(wp_mesh.id)

            self.meshes[mesh_prim_path_regex] = wp_meshes
            rigid_body_view = self._get_rigid_body_view(env_prim_path_expr, wp_mesh_names)
            if rigid_body_view is None:
                self.mesh_prototype_ids.extend(wp_mesh_ids)
                self.mesh_collision_groups.extend([-1] * len(wp_mesh_ids))
                continue

            assert rigid_body_view.count % len(wp_mesh_ids) == 0, (
                f"Rigid body view count {rigid_body_view.count} is not divisible by "
                f"the number of meshes {len(wp_mesh_ids)} in {mesh_prim_path_regex}."
            )
            self.mesh_prototype_ids.extend(wp_mesh_ids * (rigid_body_view.count // len(wp_mesh_ids)))
            self.rigid_body_mesh_transform_segments[mesh_prim_path_regex] = slice(
                len(self.mesh_prototype_ids) - rigid_body_view.count, len(self.mesh_prototype_ids)
            )
            self.rigid_body_views[mesh_prim_path_regex] = rigid_body_view
            self.mesh_collision_groups.extend(self._get_collision_groups_by_env_str(rigid_body_view))

        self.mesh_transforms_pyt = torch.zeros(len(self.mesh_prototype_ids), 7, dtype=torch.float32, device=self.device)
        self.mesh_transforms_pyt[:, -1] = 1.0
        self.mesh_inv_transforms_pyt = torch.zeros_like(self.mesh_transforms_pyt)
        self.mesh_inv_transforms_pyt[:, -1] = 1.0
        self.mesh_collision_groups_pyt = torch.tensor(self.mesh_collision_groups, dtype=torch.int32, device=self.device)
        self.mesh_prototype_ids_pyt = torch.tensor(self.mesh_prototype_ids, dtype=torch.int64, device=self.device)
        self.all_mesh_indices = torch.arange(len(self.mesh_prototype_ids), dtype=torch.int32, device=self.device)

    def _initialize_rays_impl(self):
        super()._initialize_rays_impl()
        self._create_ray_collision_groups()

    def _create_ray_collision_groups(self):
        self._ray_collision_groups = (
            torch.arange(self._view.count, dtype=torch.int32, device=self.device).unsqueeze(1).repeat(1, self.num_rays)
        )
        unique_groups = torch.unique(self._ray_collision_groups)
        mesh_ids_for_group = []
        mesh_ids_slice_for_group = []
        for group_id in unique_groups:
            global_indices = torch.where(self.mesh_collision_groups_pyt == -1)[0]
            group_indices = torch.where(self.mesh_collision_groups_pyt == group_id)[0]
            ray_group = torch.cat([global_indices, group_indices]).tolist()
            mesh_ids_for_group.append(ray_group)
            mesh_ids_slice_for_group.append(len(ray_group))
        self._mesh_ids_for_group = torch.tensor(mesh_ids_for_group, dtype=torch.int32, device=self.device).view(-1)
        self._mesh_ids_slice_for_group = [0] + [
            sum(mesh_ids_slice_for_group[: i + 1]) for i in range(len(mesh_ids_slice_for_group))
        ]
        self._mesh_ids_slice_for_group = torch.tensor(
            self._mesh_ids_slice_for_group, dtype=torch.int32, device=self.device
        )

    def _update_mesh_transforms(self, env_ids: torch.Tensor | None = None):
        for mesh_prim_path_regex, rigid_body_view in self.rigid_body_views.items():
            segment = self.rigid_body_mesh_transform_segments[mesh_prim_path_regex]
            rigid_body_transforms = rigid_body_view.get_transforms().view(-1, 7)
            segment_indices = self.all_mesh_indices[segment]

            if env_ids is None:
                selected_transforms = rigid_body_transforms
                mesh_tf_indices = segment_indices
            else:
                rigid_body_env_ids = self.mesh_collision_groups_pyt[segment]
                rigid_body_view_mask = torch.isin(rigid_body_env_ids, env_ids)
                selected_transforms = rigid_body_transforms[rigid_body_view_mask]
                mesh_tf_indices = segment_indices[rigid_body_view_mask]

            if selected_transforms.numel() == 0:
                continue
            self.mesh_transforms_pyt[mesh_tf_indices] = selected_transforms
            pos_inv, quat_wxyz_inv = math_utils.subtract_frame_transforms(
                selected_transforms[:, :3],
                math_utils.convert_quat(selected_transforms[:, 3:], to="wxyz"),
            )
            self.mesh_inv_transforms_pyt[mesh_tf_indices, :3] = pos_inv
            self.mesh_inv_transforms_pyt[mesh_tf_indices, 3:] = math_utils.convert_quat(quat_wxyz_inv, to="xyzw")

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        _, _, physx, XFormPrim, _, _, _, _ = _isaacsim_modules()
        self._update_mesh_transforms(env_ids)

        if isinstance(self._view, XFormPrim):
            pos_w, quat_w = self._view.get_world_poses(env_ids)
        elif isinstance(self._view, physx.ArticulationView):
            pos_w, quat_w = self._view.get_root_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = math_utils.convert_quat(quat_w, to="wxyz")
        elif isinstance(self._view, physx.RigidBodyView):
            pos_w, quat_w = self._view.get_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = math_utils.convert_quat(quat_w, to="wxyz")
        else:
            raise RuntimeError(f"Unsupported view type: {type(self._view)}")

        pos_w = pos_w.clone()
        quat_w = quat_w.clone()
        pos_w += self.drift[env_ids]
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w[env_ids] = quat_w

        if self.cfg.attach_yaw_only:
            ray_starts_w = math_utils.quat_apply_yaw(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = self.ray_directions[env_ids]
        else:
            ray_starts_w = math_utils.quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = math_utils.quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])

        self._data.ray_hits_w[env_ids] = raycast_mesh_grouped(
            mesh_prototypes=self.meshes,
            mesh_prototype_ids=self.mesh_prototype_ids_pyt,
            mesh_transforms=self.mesh_transforms_pyt,
            mesh_inv_transforms=self.mesh_inv_transforms_pyt,
            ray_group_ids=self._ray_collision_groups[env_ids],
            mesh_ids_for_group=self._mesh_ids_for_group,
            mesh_ids_slice_for_group=self._mesh_ids_slice_for_group,
            ray_starts=ray_starts_w,
            ray_directions=ray_directions_w,
            max_dist=self.cfg.max_distance,
            min_dist=self.cfg.min_distance,
        )[0]
