from __future__ import annotations

import torch

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
