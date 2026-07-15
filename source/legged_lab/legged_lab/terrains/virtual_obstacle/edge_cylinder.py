from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh

import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.timer import Timer

from legged_lab.utils.warp.cylinder import CylinderSpatialGrid

from .virtual_obstacle_base import VirtualObstacleBase

if TYPE_CHECKING:
    from .edge_cylinder_cfg import EdgeCylinderCfg


class EdgeCylinder(VirtualObstacleBase):
    """Detect sharp stair edges and inflate them as virtual cylinders."""

    cfg: EdgeCylinderCfg

    def __init__(self, cfg: EdgeCylinderCfg):
        super().__init__(cfg)
        self.device = torch.device("cpu")
        self.cylinders = None
        self.transition_cylinders = None
        self.edges_pyt = torch.empty((0, 6), dtype=torch.float32)
        self.edge_radii_pyt = torch.empty((0,), dtype=torch.float32)

    def generate(self, mesh: trimesh.Trimesh, device: torch.device | str = "cpu") -> None:
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        angles = mesh.face_adjacency_angles
        sharp_mask = angles > np.deg2rad(self.cfg.angle_threshold)

        if np.any(sharp_mask) and self.cfg.strict_step_edges:
            with Timer("Filter stair-lip edges"):
                sharp_mask &= self._get_step_edge_mask(mesh, sharp_mask)

        if not np.any(sharp_mask):
            print("[EdgeCylinder] No sharp stair edges detected.")
            self.edges_pyt = torch.empty((0, 6), dtype=torch.float32, device=self.device)
            self.edge_radii_pyt = torch.empty((0,), dtype=torch.float32, device=self.device)
            self.cylinders = None
            self.transition_cylinders = None
            return

        vertices = mesh.vertices
        sharp_edges = mesh.face_adjacency_edges[sharp_mask]
        edge_end_points = np.hstack([vertices[sharp_edges[:, 0]], vertices[sharp_edges[:, 1]]]).astype(np.float32)
        transition_mask = self._get_transition_edge_mask(edge_end_points)
        edge_radii = self._get_edge_radii(edge_end_points, transition_mask)

        self.edges_pyt = torch.tensor(edge_end_points, dtype=torch.float32, device=self.device)
        self.edge_radii_pyt = torch.tensor(edge_radii, dtype=torch.float32, device=self.device)
        self.cylinders = CylinderSpatialGrid(
            cylinders=np.concatenate([edge_end_points, edge_radii[:, None]], axis=1),
            num_grid_cells=self.cfg.num_grid_cells,
            device=self.device,
        )
        if np.any(transition_mask):
            self.transition_cylinders = CylinderSpatialGrid(
                cylinders=np.concatenate([edge_end_points[transition_mask], edge_radii[transition_mask, None]], axis=1),
                num_grid_cells=self.cfg.num_grid_cells,
                device=self.device,
            )
        else:
            self.transition_cylinders = None
        print(f"[EdgeCylinder] generated {edge_end_points.shape[0]} inflated stair edges.")

    def _get_step_edge_mask(self, mesh: trimesh.Trimesh, sharp_mask: np.ndarray) -> np.ndarray:
        adj_faces = mesh.face_adjacency
        adj_edges = mesh.face_adjacency_edges
        vertices = mesh.vertices
        face_centers = vertices[mesh.faces].mean(axis=1)
        normals = mesh.face_normals

        face_a = adj_faces[:, 0]
        face_b = adj_faces[:, 1]
        normal_a = normals[face_a]
        normal_b = normals[face_b]
        a_horizontal = normal_a[:, 2] >= self.cfg.horizontal_normal_z
        b_horizontal = normal_b[:, 2] >= self.cfg.horizontal_normal_z
        a_vertical = np.abs(normal_a[:, 2]) <= self.cfg.vertical_normal_z
        b_vertical = np.abs(normal_b[:, 2]) <= self.cfg.vertical_normal_z

        a_tread_b_riser = a_horizontal & b_vertical
        b_tread_a_riser = b_horizontal & a_vertical
        tread_faces = np.where(a_tread_b_riser, face_a, face_b)
        riser_faces = np.where(a_tread_b_riser, face_b, face_a)
        face_z_diff = np.abs(face_centers[tread_faces, 2] - face_centers[riser_faces, 2])

        edge_vertices = vertices[adj_edges]
        edge_z_span = np.abs(edge_vertices[:, 0, 2] - edge_vertices[:, 1, 2])
        edge_z = edge_vertices[:, :, 2].mean(axis=1)
        tread_z = face_centers[tread_faces, 2]

        mask = (a_tread_b_riser | b_tread_a_riser) & (face_z_diff >= self.cfg.min_face_height_diff)
        mask &= edge_z_span <= self.cfg.max_edge_z_span
        mask &= np.abs(edge_z - tread_z) <= self.cfg.max_edge_z_span
        mask &= self._get_tile_border_mask(edge_vertices)
        return mask

    def _get_tile_border_mask(self, edge_vertices: np.ndarray) -> np.ndarray:
        keep = np.ones(edge_vertices.shape[0], dtype=bool)
        if not self.cfg.exclude_tile_border_edges:
            return keep
        if self.cfg.terrain_tile_size is None or self.cfg.terrain_grid_shape is None:
            return keep

        tile_size = np.asarray(self.cfg.terrain_tile_size, dtype=np.float64)
        grid_shape = np.asarray(self.cfg.terrain_grid_shape, dtype=np.float64)
        total_size = tile_size * grid_shape
        mid_xy = edge_vertices.mean(axis=1)[:, :2]
        xy_in_grid = mid_xy + 0.5 * total_size

        eps = 1.0e-6
        keep &= np.all((xy_in_grid >= -eps) & (xy_in_grid <= total_size + eps), axis=1)
        local_xy = np.mod(np.clip(xy_in_grid, 0.0, total_size - eps), tile_size)
        dist_to_tile_edge = np.minimum(local_xy, tile_size - local_xy).min(axis=1)
        keep &= dist_to_tile_edge > self.cfg.tile_border_margin

        if self.cfg.sub_terrain_border_width > 0.0:
            inner_min = self.cfg.sub_terrain_border_width - self.cfg.tile_border_margin
            inner_max = tile_size - self.cfg.sub_terrain_border_width + self.cfg.tile_border_margin
            keep &= np.all((local_xy >= inner_min) & (local_xy <= inner_max), axis=1)
        return keep

    def _get_transition_edge_mask(self, edge_end_points: np.ndarray) -> np.ndarray:
        transition_mask = np.zeros(edge_end_points.shape[0], dtype=bool)
        if edge_end_points.size == 0 or self.cfg.terrain_tile_size is None or self.cfg.terrain_grid_shape is None:
            return transition_mask
        if self.cfg.sub_terrain_border_width <= 0.0:
            return transition_mask

        tile_size = np.asarray(self.cfg.terrain_tile_size, dtype=np.float64)
        grid_shape = np.asarray(self.cfg.terrain_grid_shape, dtype=np.float64)
        edge_vertices = edge_end_points.reshape(-1, 2, 3)
        total_size = tile_size * grid_shape
        mid_xy_in_grid = edge_vertices.mean(axis=1)[:, :2] + 0.5 * total_size
        eps = 1.0e-6
        inside_grid = np.all((mid_xy_in_grid >= -eps) & (mid_xy_in_grid <= total_size + eps), axis=1)
        if not np.any(inside_grid):
            return transition_mask

        tile_ids = np.floor(np.clip(mid_xy_in_grid, 0.0, total_size - eps) / tile_size).astype(np.int64)
        local_xy = mid_xy_in_grid - tile_ids * tile_size
        endpoint_xy_in_grid = edge_vertices[:, :, :2] + 0.5 * total_size
        endpoint_local_xy = endpoint_xy_in_grid - tile_ids[:, None, :] * tile_size
        edge_xy_span = np.abs(endpoint_local_xy[:, 1] - endpoint_local_xy[:, 0])
        is_x_aligned = edge_xy_span[:, 0] >= edge_xy_span[:, 1]

        coord = np.where(is_x_aligned, local_xy[:, 1], local_xy[:, 0])
        coord_dim = np.where(is_x_aligned, tile_size[1], tile_size[0])
        transition_mask |= inside_grid & (
            (np.abs(coord - self.cfg.sub_terrain_border_width) <= self.cfg.transition_edge_margin)
            | (np.abs(coord - (coord_dim - self.cfg.sub_terrain_border_width)) <= self.cfg.transition_edge_margin)
        )
        return transition_mask

    def _get_edge_radii(self, edge_end_points: np.ndarray, transition_mask: np.ndarray) -> np.ndarray:
        radii = np.full((edge_end_points.shape[0],), self.cfg.cylinder_radius, dtype=np.float32)
        if self.cfg.transition_edge_radius != self.cfg.cylinder_radius:
            radii[transition_mask] = self.cfg.transition_edge_radius
        return radii

    def disable_visualizer(self) -> None:
        if hasattr(self, "_cylinder_visualizer"):
            self._cylinder_visualizer.set_visibility(False)

    def visualize(self) -> None:
        if self.edges_pyt.numel() == 0:
            return
        if not hasattr(self, "_cylinder_visualizer"):
            self._cylinder_visualizer = VisualizationMarkers(self.cfg.visualizer)
            self._cylinder_rotate_y_90 = math_utils.quat_from_angle_axis(
                angle=torch.tensor([np.pi / 2], device=self.device),
                axis=torch.tensor([[0.0, 1.0, 0.0]], device=self.device),
            )

        trans = (self.edges_pyt[:, :3] + self.edges_pyt[:, 3:6]) / 2.0
        direction = self.edges_pyt[:, 3:6] - self.edges_pyt[:, :3]
        default_direction = torch.zeros_like(direction)
        default_direction[:, 0] = 1.0
        normalized_direction = direction / torch.norm(direction, dim=-1, keepdim=True).clamp_min(1.0e-6)
        axis = torch.cross(default_direction, normalized_direction, dim=-1)
        dot_prod = torch.sum(default_direction * normalized_direction, dim=-1)
        angle = torch.acos(torch.clamp(dot_prod, -1.0, 1.0))
        quat = math_utils.quat_mul(
            math_utils.quat_from_angle_axis(angle, axis),
            self._cylinder_rotate_y_90.expand(direction.shape[0], -1),
        )
        scales = torch.ones(len(self.edges_pyt), 3, device=self.device)
        scales[:, 0] = self.edge_radii_pyt
        scales[:, 1] = self.edge_radii_pyt
        scales[:, 2] = torch.norm(direction, dim=-1)
        self._cylinder_visualizer.visualize(translations=trans, orientations=quat, scales=scales)
        self._cylinder_visualizer.set_visibility(True)

    def get_points_penetration_offset(self, points: torch.Tensor) -> torch.Tensor:
        if self.cylinders is None:
            return torch.zeros_like(points, device=points.device)
        return self.cylinders.get_points_penetration_offset(points)

    def get_transition_points_penetration_offset(self, points: torch.Tensor) -> torch.Tensor:
        if self.transition_cylinders is None:
            return torch.zeros_like(points, device=points.device)
        return self.transition_cylinders.get_points_penetration_offset(points)

