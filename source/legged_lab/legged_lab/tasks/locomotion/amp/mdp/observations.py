from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation  # runtime class, guarded per v3 pattern
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors import RayCaster

    from legged_lab.envs import ManagerBasedAnimationEnv
    from legged_lab.managers import AnimationTerm


def _crop_and_resize_depth(
    data: torch.Tensor,
    crop_region: tuple[int, int, int, int] = (0, 0, 0, 0),
    resize_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    top, bottom, left, right = crop_region
    h_end = data.shape[1] - bottom if bottom > 0 else data.shape[1]
    w_end = data.shape[2] - right if right > 0 else data.shape[2]
    data = data[:, top:h_end, left:w_end, :]
    if resize_shape is None:
        return data
    data_chw = data.permute(0, 3, 1, 2)
    return F.interpolate(data_chw, size=resize_shape, mode="bilinear", align_corners=False).permute(0, 2, 3, 1)


def _gaussian_blur_depth(data: torch.Tensor, kernel_size: int = 3, sigma: float = 1.0) -> torch.Tensor:
    if kernel_size <= 1:
        return data
    coords = torch.arange(kernel_size, device=data.device, dtype=data.dtype) - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, kernel_size, kernel_size)
    channels = data.shape[-1]
    kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)
    data_chw = data.permute(0, 3, 1, 2)
    blurred = F.conv2d(data_chw, kernel_2d, padding=kernel_size // 2, groups=channels)
    return blurred.permute(0, 2, 3, 1)


def _scale_randomization_depth(data: torch.Tensor, apply_probability: float = 0.5) -> torch.Tensor:
    if torch.rand((), device=data.device) > apply_probability:
        return data
    scales = torch.empty((data.shape[0], 1, 1, 1), device=data.device).uniform_(0.97, 1.03)
    return data * scales


def _stereo_fusion_depth(data: torch.Tensor, apply_probability: float = 0.4) -> torch.Tensor:
    if torch.rand((), device=data.device) > apply_probability:
        return data
    depth_chw = data.mean(dim=-1, keepdim=True).permute(0, 3, 1, 2)
    grad_x = torch.abs(depth_chw[:, :, :, 1:] - depth_chw[:, :, :, :-1])
    grad_y = torch.abs(depth_chw[:, :, 1:, :] - depth_chw[:, :, :-1, :])
    grad = torch.zeros_like(depth_chw)
    grad[:, :, :, 1:] += grad_x
    grad[:, :, :, :-1] += grad_x
    grad[:, :, 1:, :] += grad_y
    grad[:, :, :-1, :] += grad_y
    local_mean = F.avg_pool2d(depth_chw, kernel_size=3, stride=1, padding=1)
    local_var = F.avg_pool2d((depth_chw - local_mean) ** 2, kernel_size=3, stride=1, padding=1)
    hole_mask = ((grad > 0.10) | (local_var < 3e-4)) & (torch.rand_like(depth_chw) < 0.02)
    noisy = data.clone()
    noisy[hole_mask.permute(0, 2, 3, 1).expand_as(noisy)] = 2.5
    return noisy


def _random_conv_depth(data: torch.Tensor, apply_probability: float = 0.3) -> torch.Tensor:
    if torch.rand((), device=data.device) > apply_probability:
        return data
    data_chw = data.permute(0, 3, 1, 2)
    channels = data_chw.shape[1]
    kernel = torch.randn((1, 1, 3, 3), device=data.device, dtype=data.dtype) * 0.05
    kernel[:, :, 1, 1] += 1.0
    kernel = kernel / kernel.abs().sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    kernel = kernel.repeat(channels, 1, 1, 1)
    return F.conv2d(data_chw, kernel, padding=1, groups=channels).permute(0, 2, 3, 1)


def _perlin_depth(data: torch.Tensor, apply_probability: float = 0.5) -> torch.Tensor:
    if torch.rand((), device=data.device) > apply_probability:
        return data
    num_envs, height, width, _ = data.shape
    noise = torch.zeros((num_envs, 1, height, width), device=data.device, dtype=data.dtype)
    frequency = 8.0
    amplitude = 1.0
    for _ in range(3):
        grid_h = max(2, int(round(height / frequency)))
        grid_w = max(2, int(round(width / frequency)))
        octave = torch.randn((num_envs, 1, grid_h, grid_w), device=data.device, dtype=data.dtype)
        octave = F.interpolate(octave, size=(height, width), mode="bilinear", align_corners=False)
        noise += amplitude * octave
        frequency *= 2.0
        amplitude *= 0.5
    noise = noise / noise.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return data + 0.01 * noise.permute(0, 2, 3, 1)


def _pixel_failures_depth(data: torch.Tensor, apply_probability: float = 0.5) -> torch.Tensor:
    if torch.rand((), device=data.device) > apply_probability:
        return data
    noisy = data.clone()
    rand_map = torch.rand_like(noisy)
    noisy[rand_map < 5e-4] = 0.0
    noisy[(rand_map >= 5e-4) & (rand_map < 1e-3)] = 2.5
    return noisy


def depth_image(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return noised, cropped, normalized depth image for AME encoder input."""
    sensor = env.scene.sensors[sensor_cfg.name]
    data = sensor.data.output["distance_to_image_plane"]
    data = torch.nan_to_num(data, nan=2.5, posinf=2.5, neginf=0.0).clamp(0.0, 2.5)
    data = _scale_randomization_depth(data)
    data = _stereo_fusion_depth(data)
    data = _random_conv_depth(data)
    data = _perlin_depth(data)
    data = _pixel_failures_depth(data)
    data = _crop_and_resize_depth(data, crop_region=(18, 0, 16, 16))
    data = _gaussian_blur_depth(data, kernel_size=3, sigma=1.0)
    data = data.clamp(0.0, 2.5) / 2.5
    return data.reshape(data.shape[0], -1)


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
