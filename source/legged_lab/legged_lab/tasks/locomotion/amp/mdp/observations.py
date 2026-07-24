from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation  # runtime class, guarded per v3 pattern
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors import RayCaster

    from legged_lab.envs import ManagerBasedAnimationEnv
    from legged_lab.managers import AnimationTerm


def height_scan_xyz(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.5,
) -> torch.Tensor:
    """Return yaw-frame ``(x, y, z)`` terrain points with a height bias.

    The horizontal coordinates retain the scanner's forward offset. The vertical
    coordinate matches Isaac Lab's scalar height scan convention:
    ``sensor_z - hit_z - offset``.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    sensor_pos_w = getattr(sensor.data.pos_w, "torch", sensor.data.pos_w)
    sensor_quat_w = getattr(sensor.data.quat_w, "torch", sensor.data.quat_w)
    ray_hits_w = getattr(sensor.data.ray_hits_w, "torch", sensor.data.ray_hits_w)

    num_envs, num_rays = ray_hits_w.shape[:2]
    relative_pos_w = ray_hits_w - sensor_pos_w.unsqueeze(1)
    yaw_quat_inv = math_utils.quat_conjugate(math_utils.yaw_quat(sensor_quat_w))
    coords = math_utils.quat_apply(
        yaw_quat_inv.unsqueeze(1).expand(num_envs, num_rays, 4).reshape(-1, 4),
        relative_pos_w.reshape(-1, 3),
    ).reshape(num_envs, num_rays, 3)
    coords[..., 2] = -coords[..., 2] - offset
    result = coords.reshape(num_envs, -1)
    # ===== 新增：NaN 保护 =====
    result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
    # ==========================
    return result


def root_local_rot_tan_norm(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]

    root_quat = robot.data.root_quat_w.torch  # v3: ProxyArray -> Tensor for @jit.script fns
    yaw_quat = math_utils.yaw_quat(root_quat)

    root_quat_local = math_utils.quat_mul(math_utils.quat_conjugate(yaw_quat), root_quat)

    root_rotm_local = math_utils.matrix_from_quat(root_quat_local)
    # use the first and last column of the rotation matrix as the tangent and normal vectors
    tan_vec = root_rotm_local[:, :, 0]  # (N, 3)
    norm_vec = root_rotm_local[:, :, 2]  # (N, 3)
    obs = torch.cat([tan_vec, norm_vec], dim=-1)  # (N, 6)

    return obs


def ref_root_local_rot_tan_norm(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
) -> torch.Tensor:
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    num_envs = env.num_envs

    ref_root_quat = animation_term.get_root_quat()  # shape: (num_envs, num_steps, 4)
    ref_yaw_quat = math_utils.yaw_quat(ref_root_quat)
    ref_root_quat_local = math_utils.quat_mul(
        math_utils.quat_conjugate(ref_yaw_quat), ref_root_quat
    )  # shape: (num_envs, num_steps, 4)
    ref_root_rotm_local = math_utils.matrix_from_quat(ref_root_quat_local)  # shape: (num_envs, num_steps, 3, 3)

    tan_vec = ref_root_rotm_local[:, :, :, 0]  # (num_envs, num_steps, 3)
    norm_vec = ref_root_rotm_local[:, :, :, 2]  # (num_envs, num_steps, 3)
    obs = torch.cat([tan_vec, norm_vec], dim=-1)  # (num_envs, num_steps, 6)

    if flatten_steps_dim:
        return obs.reshape(num_envs, -1)
    else:
        return obs
