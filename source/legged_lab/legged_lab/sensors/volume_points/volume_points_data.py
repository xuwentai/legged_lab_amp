from __future__ import annotations

from dataclasses import MISSING, dataclass

import torch


@dataclass
class VolumePointsData:
    """Data container for volume-point sensors."""

    pos_w: torch.Tensor = MISSING
    quat_w: torch.Tensor = MISSING
    vel_w: torch.Tensor = MISSING
    ang_vel_w: torch.Tensor = MISSING
    point_num_each_body: int = MISSING
    points_pos_w: torch.Tensor = MISSING
    points_vel_w: torch.Tensor = MISSING
    penetration_offset: torch.Tensor = MISSING
    transition_penetration_offset: torch.Tensor = MISSING

    @staticmethod
    def make_zero(
        num_envs: int,
        num_bodies: int,
        point_num_each_body: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "VolumePointsData":
        return VolumePointsData(
            pos_w=torch.zeros((num_envs, num_bodies, 3), device=device, dtype=dtype),
            quat_w=torch.zeros((num_envs, num_bodies, 4), device=device, dtype=dtype),
            vel_w=torch.zeros((num_envs, num_bodies, 3), device=device, dtype=dtype),
            ang_vel_w=torch.zeros((num_envs, num_bodies, 3), device=device, dtype=dtype),
            point_num_each_body=point_num_each_body,
            points_pos_w=torch.zeros((num_envs, num_bodies, point_num_each_body, 3), device=device, dtype=dtype),
            points_vel_w=torch.zeros((num_envs, num_bodies, point_num_each_body, 3), device=device, dtype=dtype),
            penetration_offset=torch.zeros(
                (num_envs, num_bodies, point_num_each_body, 3), device=device, dtype=dtype
            ),
            transition_penetration_offset=torch.zeros(
                (num_envs, num_bodies, point_num_each_body, 3), device=device, dtype=dtype
            ),
        )

