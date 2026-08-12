from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.assets import RigidObject  # runtime class, guarded per v3 pattern
    from isaaclab.sensors import ContactSensor  # runtime class, guarded per v3 pattern
    from isaaclab.envs import ManagerBasedRLEnv


def feet_orientation_l2(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet orientation not parallel to the ground when in contact.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]

    in_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    # shape: (N, M)

    num_feet = len(sensor_cfg.body_ids)

    feet_quat = asset.data.body_quat_w[:, sensor_cfg.body_ids, :]  # shape: (N, M, 4)
    feet_proj_g = math_utils.quat_apply_inverse(
        feet_quat, asset.data.GRAVITY_VEC_W.unsqueeze(1).expand(-1, num_feet, -1)  # shape: (N, M, 3)
    )
    feet_proj_g_xy_square = torch.sum(torch.square(feet_proj_g[:, :, :2]), dim=-1)  # shape: (N, M)

    return torch.sum(feet_proj_g_xy_square * in_contact, dim=-1)  # shape: (N, )


def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


def feet_contact_without_command(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward both feet being in contact on every zero-command step."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return torch.sum(is_contact, dim=1).float() * (command_norm < command_threshold)


def link_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat link orientation using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    link_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    link_projected_gravity = math_utils.quat_apply_inverse(link_quat, asset.data.GRAVITY_VEC_W)

    return torch.sum(torch.square(link_projected_gravity[:, :2]), dim=1)


def _resolve_stair_terrain_masks(
    env: ManagerBasedRLEnv,
) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor | None]:
    """Resolve per-env up/down-stair masks from IsaacLab terrain metadata when available."""
    device = env.device
    num_envs = env.scene.num_envs
    mask_up = torch.zeros(num_envs, dtype=torch.bool, device=device)
    mask_down = torch.zeros(num_envs, dtype=torch.bool, device=device)
    terrain = getattr(env.scene, "terrain", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    terrain_origins = getattr(terrain, "terrain_origins", None)
    if terrain_types is None or terrain_origins is None:
        return mask_up, mask_down, [], None

    terrain_levels = getattr(terrain, "terrain_levels", None)
    terrain_types_idx = getattr(terrain, "terrain_types_idx", None)
    if terrain_levels is None or terrain_types_idx is None:
        return mask_up, mask_down, list(terrain_types), None

    terrain_types_idx = terrain_types_idx.to(device=device, dtype=torch.long)
    sub_idx_per_env = terrain_types_idx
    names = list(terrain_types)
    lower_names = [name.lower() for name in names]
    up_ids = torch.tensor(
        [idx for idx, name in enumerate(lower_names) if "stair" in name and "inv" not in name and "down" not in name],
        dtype=torch.long,
        device=device,
    )
    down_ids = torch.tensor(
        [idx for idx, name in enumerate(lower_names) if "stair" in name and ("inv" in name or "down" in name)],
        dtype=torch.long,
        device=device,
    )
    if len(up_ids) > 0:
        mask_up = (terrain_types_idx.unsqueeze(-1) == up_ids).any(dim=-1)
    if len(down_ids) > 0:
        mask_down = (terrain_types_idx.unsqueeze(-1) == down_ids).any(dim=-1)
    return mask_up, mask_down, names, sub_idx_per_env


def volume_points_penetration_feet(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    tolerance: float = 0.0,
    normal_penalty_scale: float = 1.0,
    transition_penalty_scale: float = 2.0,
    enable_terrain_foot_weights: bool = False,
    stairs_weight_min: float = 0.2,
    stairs_weight_max: float = 1.0,
    heel_weight_scale: float = 1.0,
) -> torch.Tensor:
    """Penalize foot volume-point penetration into inflated stair-edge cylinders."""
    from legged_lab.sensors.volume_points.volume_points import VolumePoints
    from legged_lab.sensors.volume_points.points_generator import grid3d_points_generator

    volume_sensor: VolumePoints = env.scene.sensors[sensor_cfg.name]
    penetration = volume_sensor.data.penetration_offset
    num_envs, num_bodies, num_points, _ = penetration.shape

    penetration_depth = torch.norm(penetration.flatten(1, 2), dim=-1)
    in_obstacle = (penetration_depth > tolerance).float()
    points_vel_norm = torch.norm(volume_sensor.data.points_vel_w.flatten(1, 2), dim=-1)
    weighted_penetration = normal_penalty_scale * in_obstacle * (points_vel_norm + 1.0e-6) * penetration_depth

    transition_penetration = volume_sensor.data.transition_penetration_offset.flatten(1, 2)
    transition_penetration_depth = torch.norm(transition_penetration, dim=-1)
    in_transition = (transition_penetration_depth > tolerance).float()
    weighted_penetration += (
        (transition_penalty_scale - normal_penalty_scale)
        * in_transition
        * (points_vel_norm + 1.0e-6)
        * transition_penetration_depth
    )

    if enable_terrain_foot_weights or heel_weight_scale != 1.0:
        points_cfg = volume_sensor.cfg.points_generator
        local_points = grid3d_points_generator(points_cfg).to(env.device)
        x_frac = (local_points[:, 0] - points_cfg.x_min) / (points_cfg.x_max - points_cfg.x_min + 1.0e-8)
        x_frac = x_frac.clamp(0.0, 1.0)

        env_w = torch.ones((num_envs, num_points), device=env.device)
        if enable_terrain_foot_weights:
            mask_up, mask_down, _, sub_idx_per_env = _resolve_stair_terrain_masks(env)
            weight_span = stairs_weight_max - stairs_weight_min
            w_toe_heavy = stairs_weight_min + weight_span * x_frac
            w_heel_heavy = stairs_weight_max - weight_span * x_frac
            w_mid_heavy = stairs_weight_min + weight_span * (1.0 - torch.abs(2.0 * x_frac - 1.0))
            env_w = w_mid_heavy.unsqueeze(0).repeat(num_envs, 1)
            if sub_idx_per_env is not None:
                env_w[mask_up] = w_toe_heavy.unsqueeze(0)
                env_w[mask_down] = w_heel_heavy.unsqueeze(0)
            else:
                env_w = w_heel_heavy.unsqueeze(0).repeat(num_envs, 1)

        if heel_weight_scale != 1.0:
            env_w *= (1.0 + (heel_weight_scale - 1.0) * (1.0 - x_frac)).unsqueeze(0)

        weighted_penetration = weighted_penetration.view(num_envs, num_bodies, num_points)
        weighted_penetration = (weighted_penetration * env_w.unsqueeze(1)).flatten(1, 2)

    return torch.sum(weighted_penetration, dim=-1)
