from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .points_generator_cfg import Grid3dPointsGeneratorCfg


def grid3d_points_generator(points_generator_cfg: Grid3dPointsGeneratorCfg) -> torch.Tensor:
    """Create a local 3D grid of sample points."""
    x = torch.linspace(points_generator_cfg.x_min, points_generator_cfg.x_max, points_generator_cfg.x_num)
    y = torch.linspace(points_generator_cfg.y_min, points_generator_cfg.y_max, points_generator_cfg.y_num)
    z = torch.linspace(points_generator_cfg.z_min, points_generator_cfg.z_max, points_generator_cfg.z_num)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
    return torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

