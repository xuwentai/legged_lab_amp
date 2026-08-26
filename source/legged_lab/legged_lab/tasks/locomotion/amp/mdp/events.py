from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg


def register_virtual_obstacle_to_sensor(
    env,
    env_ids: torch.Tensor | None,
    sensor_cfgs: list[SceneEntityCfg] | SceneEntityCfg,
):
    """Register terrain virtual obstacles on volume-point sensors."""
    if isinstance(sensor_cfgs, SceneEntityCfg):
        sensor_cfgs = [sensor_cfgs]

    virtual_obstacles = getattr(env.scene.terrain, "virtual_obstacles", {})
    for sensor_cfg in sensor_cfgs:
        sensor = env.scene[sensor_cfg.name]
        if not hasattr(sensor, "register_virtual_obstacles"):
            raise ValueError(f"Sensor '{sensor_cfg.name}' does not support virtual obstacles.")
        sensor.register_virtual_obstacles(virtual_obstacles)


def randomize_camera_offsets(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    offset_pose_ranges: dict[str, tuple[float, float]],
):
    """Randomize per-environment camera mounting offsets from its configured nominal pose."""
    sensor = env.scene.sensors[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    base_pos = torch.tensor(list(sensor.cfg.offset.pos), device=env.device).repeat(len(env_ids), 1)
    pos_noise = torch.empty_like(base_pos)
    for axis, key in enumerate(("x", "y", "z")):
        low, high = offset_pose_ranges.get(key, (0.0, 0.0))
        pos_noise[:, axis].uniform_(low, high)

    base_quat = math_utils.convert_camera_frame_orientation_convention(
        torch.tensor([sensor.cfg.offset.rot], device=env.device),
        origin=sensor.cfg.offset.convention,
        target="world",
    ).repeat(len(env_ids), 1)
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(base_quat)
    for values, key in ((roll, "roll"), (pitch, "pitch"), (yaw, "yaw")):
        low, high = offset_pose_ranges.get(key, (0.0, 0.0))
        values += torch.empty_like(values).uniform_(low, high)
    offset_quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

    if hasattr(sensor, "_offset_pos") and hasattr(sensor, "_offset_quat"):
        sensor._offset_pos[env_ids] = base_pos + pos_noise
        sensor._offset_quat[env_ids] = offset_quat
        return

    raise AttributeError(f"Sensor '{asset_cfg.name}' does not expose mutable camera offsets.")
