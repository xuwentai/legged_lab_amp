from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import warp as wp

from .kernels import points_penetrate_cylinder_kernel


class CylinderSpatialGrid:
    """Spatial grid for querying point penetration against many cylinders."""

    def __init__(
        self,
        cylinders: torch.Tensor | np.ndarray,
        num_grid_cells: int = 64**3,
        device: str | torch.device = "cuda",
    ):
        self.cylinders_np = cylinders if isinstance(cylinders, np.ndarray) else cylinders.cpu().numpy()
        if self.cylinders_np.shape[1] != 7:
            raise ValueError("Cylinders must have shape (M, 7): start xyz, end xyz, radius.")
        self.num_grid_cells = num_grid_cells
        self.device = device
        self._compute_bounding_box()
        self._create_grid()

    def _compute_bounding_box(self):
        xyz = np.concatenate([self.cylinders_np[:, :3], self.cylinders_np[:, 3:6]], axis=0)
        r_max = self.cylinders_np[:, 6].max()
        bbox_min = xyz.min(axis=0) - r_max
        bbox_max = xyz.max(axis=0) + r_max
        extent = np.maximum(bbox_max - bbox_min, 1.0e-6)

        scale = extent / extent.max()
        grid_res = np.maximum(np.round(scale * (self.num_grid_cells ** (1.0 / 3.0))), 1).astype(int)
        self.grid_res = grid_res
        self.total_num_cells = int(np.prod(grid_res))
        self.cell_size = extent / grid_res
        self.bbox_min = bbox_min

    def _flat_grid_idx(self, ix: int, iy: int, iz: int) -> int:
        if ix < 0 or ix >= self.grid_res[0] or iy < 0 or iy >= self.grid_res[1] or iz < 0 or iz >= self.grid_res[2]:
            return -1
        return ix * self.grid_res[1] * self.grid_res[2] + iy * self.grid_res[2] + iz

    def _create_grid(self):
        grid = defaultdict(list)
        for idx, cyl in enumerate(self.cylinders_np):
            radius = cyl[6]
            bbox_min = np.minimum(cyl[:3], cyl[3:6]) - radius
            bbox_max = np.maximum(cyl[:3], cyl[3:6]) + radius
            min_cell = np.floor((bbox_min - self.bbox_min) / self.cell_size).astype(int)
            max_cell = np.floor((bbox_max - self.bbox_min) / self.cell_size).astype(int)

            for ix in range(min_cell[0], max_cell[0] + 1):
                for iy in range(min_cell[1], max_cell[1] + 1):
                    for iz in range(min_cell[2], max_cell[2] + 1):
                        flat = self._flat_grid_idx(ix, iy, iz)
                        if 0 <= flat < self.total_num_cells:
                            grid[flat].append(idx)

        self.cell_offsets = np.zeros(self.total_num_cells + 1, dtype=np.int32)
        cell_indices: list[int] = []
        for idx in range(self.total_num_cells):
            self.cell_offsets[idx] = len(cell_indices)
            cell_indices.extend(grid[idx])
        self.cell_offsets[-1] = len(cell_indices)
        self.cell_indices = np.asarray(cell_indices, dtype=np.int32)

        device = str(self.device)
        self.cell_offsets_wp = wp.array(self.cell_offsets, dtype=wp.int32, device=device)
        self.cell_indices_wp = wp.array(self.cell_indices, dtype=wp.int32, device=device)
        self.cell_size_wp = wp.vec3(*self.cell_size)
        self.bbox_min_wp = wp.vec3(*self.bbox_min)
        self.grid_res_wp = wp.vec3i(*self.grid_res)
        self.cylinder_start_wp = wp.array(self.cylinders_np[:, :3], dtype=wp.vec3, device=device)
        self.cylinder_end_wp = wp.array(self.cylinders_np[:, 3:6], dtype=wp.vec3, device=device)
        self.cylinder_radius_wp = wp.array(self.cylinders_np[:, 6], dtype=wp.float32, device=device)

    def get_points_penetration_offset(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points must have shape (N, 3).")
        points_wp = wp.from_torch(points.contiguous(), dtype=wp.vec3)
        penetration_offset = torch.zeros(points.shape[0], 3, device=points.device, dtype=points.dtype)
        penetration_offset_wp = wp.from_torch(penetration_offset, dtype=wp.vec3)

        wp.launch(
            points_penetrate_cylinder_kernel,
            dim=points.shape[0],
            inputs=[
                points_wp,
                self.cylinder_start_wp,
                self.cylinder_end_wp,
                self.cylinder_radius_wp,
                self.cell_offsets_wp,
                self.cell_indices_wp,
                self.grid_res_wp,
                self.bbox_min_wp,
                self.cell_size_wp,
                penetration_offset_wp,
            ],
            device=str(points.device),
        )
        return penetration_offset

