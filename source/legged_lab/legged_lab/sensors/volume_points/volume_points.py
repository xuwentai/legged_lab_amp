from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import omni.physics.tensors.impl.api as physx
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors.sensor_base import SensorBase

from .volume_points_data import VolumePointsData

if TYPE_CHECKING:
    from .volume_points_cfg import VolumePointsCfg


class VolumePoints(SensorBase):
    """Samples local body volume points and reports penetration into registered virtual obstacles."""

    cfg: VolumePointsCfg

    def __init__(self, cfg: VolumePointsCfg):
        super().__init__(cfg)
        self._body_physx_view = None

    @property
    def data(self) -> VolumePointsData:
        self._update_outdated_buffers()
        return self._data

    @property
    def num_bodies(self) -> int:
        return self._num_bodies

    @property
    def body_names(self) -> list[str]:
        prim_paths = self.body_physx_view.prim_paths[: self.num_bodies]
        return [path.split("/")[-1] for path in prim_paths]

    @property
    def body_physx_view(self) -> physx.RigidBodyView:
        return self._body_physx_view

    def register_virtual_obstacles(self, virtual_obstacles: dict[str, Any]) -> None:
        self._virtual_obstacles.update(virtual_obstacles)

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)

    def _initialize_impl(self):
        super()._initialize_impl()
        self._physics_sim_view = physx.create_simulation_view(self._backend)
        self._physics_sim_view.set_subspace_roots("/")

        leaf_pattern = self.cfg.prim_path.rsplit("/", 1)[-1]
        template_prim_path = self._parent_prims[0].GetPath().pathString
        body_names = []
        for prim in sim_utils.find_matching_prims(template_prim_path + "/" + leaf_pattern):
            body_names.append(prim.GetPath().pathString.rsplit("/", 1)[-1])
        if not body_names:
            raise RuntimeError(f"VolumePoints sensor at '{self.cfg.prim_path}' could not find any bodies.")

        body_regex = r"(" + "|".join(body_names) + r")"
        body_regex = f"{self.cfg.prim_path.rsplit('/', 1)[0]}/{body_regex}"
        self._body_physx_view = self._physics_sim_view.create_rigid_body_view(body_regex.replace(".*", "*"))
        self._num_bodies = self.body_physx_view.count // self._num_envs
        if self._num_bodies != len(body_names):
            raise RuntimeError(
                "Failed to initialize VolumePoints sensor."
                f"\n\tInput prim path    : {self.cfg.prim_path}"
                f"\n\tResolved prim paths: {body_regex}"
            )

        self._volume_points_pattern = self.cfg.points_generator.func(self.cfg.points_generator).to(self.device)
        self._data = VolumePointsData.make_zero(
            num_envs=self._num_envs,
            num_bodies=self._num_bodies,
            point_num_each_body=self._volume_points_pattern.shape[0],
            device=self.device,
        )
        self._virtual_obstacles: dict[str, Any] = {}

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        if len(env_ids) == self._num_envs:
            env_ids = slice(None)
        self._refresh_volume_points(env_ids)
        self._refresh_penetration_offset(env_ids)

    def _refresh_volume_points(self, env_ids: Sequence[int] | slice) -> None:
        body_poses = self.body_physx_view.get_transforms().view(-1, self.num_bodies, 7)[env_ids]
        body_vels = self.body_physx_view.get_velocities().view(-1, self.num_bodies, 6)[env_ids]

        self._data.pos_w[env_ids] = body_poses[..., :3]
        self._data.quat_w[env_ids] = math_utils.convert_quat(body_poses[..., 3:], to="wxyz")
        self._data.vel_w[env_ids] = body_vels[..., :3]
        self._data.ang_vel_w[env_ids] = body_vels[..., 3:]

        num_body_instances = self._data.pos_w[env_ids].shape[0] * self._data.pos_w[env_ids].shape[1]
        points_pos_w = math_utils.transform_points(
            self._volume_points_pattern.unsqueeze(0).expand(num_body_instances, -1, -1),
            self._data.pos_w[env_ids].flatten(0, 1),
            self._data.quat_w[env_ids].flatten(0, 1),
        ).reshape(*self._data.pos_w[env_ids].shape[:2], self._data.point_num_each_body, 3)
        self._data.points_pos_w[env_ids] = points_pos_w

        points_vel_w = self._data.vel_w[env_ids].unsqueeze(-2).expand_as(points_pos_w).clone()
        points_vel_w += torch.linalg.cross(
            self._data.ang_vel_w[env_ids].unsqueeze(-2),
            points_pos_w - self._data.pos_w[env_ids].unsqueeze(-2),
            dim=-1,
        )
        self._data.points_vel_w[env_ids] = points_vel_w

    def _refresh_penetration_offset(self, env_ids: Sequence[int] | slice) -> None:
        penetration = self._data.penetration_offset[env_ids]
        transition_penetration = self._data.transition_penetration_offset[env_ids]
        penetration.zero_()
        transition_penetration.zero_()
        penetration_depth = torch.zeros_like(penetration[..., 0])
        transition_depth = torch.zeros_like(transition_penetration[..., 0])

        if not self._virtual_obstacles:
            return

        points = self._data.points_pos_w[env_ids].flatten(0, 2)
        for virtual_obstacle in self._virtual_obstacles.values():
            offset = virtual_obstacle.get_points_penetration_offset(points).reshape(
                self._data.points_pos_w[env_ids].shape
            )
            depth = torch.norm(offset, dim=-1)
            mask = depth > penetration_depth
            penetration_depth[mask] = depth[mask]
            penetration[mask] = offset[mask]

            if hasattr(virtual_obstacle, "get_transition_points_penetration_offset"):
                transition_offset = virtual_obstacle.get_transition_points_penetration_offset(points).reshape(
                    self._data.points_pos_w[env_ids].shape
                )
                transition_offset_depth = torch.norm(transition_offset, dim=-1)
                transition_mask = transition_offset_depth > transition_depth
                transition_depth[transition_mask] = transition_offset_depth[transition_mask]
                transition_penetration[transition_mask] = transition_offset[transition_mask]

        self._data.penetration_offset[env_ids] = penetration
        self._data.transition_penetration_offset[env_ids] = transition_penetration

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "points_visualizer"):
                self.points_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            self.points_visualizer.set_visibility(True)
        elif hasattr(self, "points_visualizer"):
            self.points_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if self.body_physx_view is None:
            return
        points = self._data.points_pos_w.view(-1, 3)
        penetrated = torch.norm(self._data.penetration_offset.view(-1, 3), dim=-1) > 0.0
        if not torch.any(penetrated):
            points = torch.cat([points, torch.zeros_like(points[:1])], dim=0)
            penetrated = torch.cat([penetrated, torch.tensor([True], device=self.device)], dim=0)
        self.points_visualizer.visualize(translations=points, marker_indices=penetrated.long())

    def _invalidate_initialize_callback(self, event):
        super()._invalidate_initialize_callback(event)
        if hasattr(self, "points_visualizer"):
            delattr(self, "points_visualizer")
        self._physics_sim_view = None
        self._body_physx_view = None

