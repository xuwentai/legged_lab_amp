#!/usr/bin/env python3
"""
MuJoCo sim2sim deployment for LeggedLab-trained G1 policies.

Loads an ActorCriticRecurrent (LSTM) or ActorCritic (MLP) checkpoint and
deploys it in MuJoCo with observation construction aligned to legged_lab BaseEnv.

Usage:
  cd AME_mujoco_sim2sim
  python deploy_mujoco.py
  python deploy_mujoco.py --policy_path ../logs/rsl_rl/g1_amp_flat/2026-07-31_18-29-09/model_49999.pt
"""
import time
import os
import sys
import argparse
import threading
import re
import numpy as np
import mujoco
import mujoco.viewer
import torch
import yaml
import pygame

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_POLICY_PATH = "../logs/rsl_rl/g1_amp_flat/2026-07-31_18-29-09/model_49999.pt"

# Add project root so rsl_rl is importable
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))

# =============================================================================
# Joint order constants
# =============================================================================

# Isaac Lab policy joint order (matches USD asset joint order from training).
POLICY_JOINT_ORDER = [
    "left_hip_pitch_joint",       # 0
    "right_hip_pitch_joint",      # 1
    "waist_yaw_joint",            # 2
    "left_hip_roll_joint",        # 3
    "right_hip_roll_joint",       # 4
    "waist_roll_joint",           # 5
    "left_hip_yaw_joint",         # 6
    "right_hip_yaw_joint",        # 7
    "waist_pitch_joint",          # 8
    "left_knee_joint",            # 9
    "right_knee_joint",           # 10
    "left_shoulder_pitch_joint",  # 11
    "right_shoulder_pitch_joint", # 12
    "left_ankle_pitch_joint",     # 13
    "right_ankle_pitch_joint",    # 14
    "left_shoulder_roll_joint",   # 15
    "right_shoulder_roll_joint",  # 16
    "left_ankle_roll_joint",      # 17
    "right_ankle_roll_joint",     # 18
    "left_shoulder_yaw_joint",    # 19
    "right_shoulder_yaw_joint",   # 20
    "left_elbow_joint",           # 21
    "right_elbow_joint",          # 22
    "left_wrist_roll_joint",      # 23
    "right_wrist_roll_joint",     # 24
    "left_wrist_pitch_joint",     # 25
    "right_wrist_pitch_joint",    # 26
    "left_wrist_yaw_joint",       # 27
    "right_wrist_yaw_joint",      # 28
]

# Default joint angles in POLICY_JOINT_ORDER, from G1_CFG.init_state.joint_pos.
# These are the regex patterns used in unitree.py line 155-167.
_DEFAULT_JOINT_PATTERNS = {
    ".*_hip_pitch_joint": -0.20,
    ".*_knee_joint": 0.42,
    ".*_ankle_pitch_joint": -0.23,
    ".*_elbow_joint": 0.87,
    "left_shoulder_roll_joint": 0.18,
    "left_shoulder_pitch_joint": 0.35,
    "right_shoulder_roll_joint": -0.18,
    "right_shoulder_pitch_joint": 0.35,
}

def _resolve_default(name):
    for pat, val in _DEFAULT_JOINT_PATTERNS.items():
        if re.match(pat, name):
            return val
    return 0.0

DEFAULT_JOINT_ANGLES_POLICY = np.array(
    [_resolve_default(n) for n in POLICY_JOINT_ORDER], dtype=np.float32
)

# =============================================================================
# PD gains from G1_CFG actuators (unitree.py)
# =============================================================================

_G1_PD_GROUPS = {
    "legs": {
        "stiffness": {".*_hip_yaw_joint": 150, ".*_hip_roll_joint": 150,
                       ".*_hip_pitch_joint": 200, ".*_knee_joint": 200,
                       ".*waist.*": 200},
        "damping": {".*_hip_yaw_joint": 5, ".*_hip_roll_joint": 5,
                     ".*_hip_pitch_joint": 5, ".*_knee_joint": 5,
                     ".*waist.*": 5},
        "effort": {".*_hip_yaw_joint": 88, ".*_hip_roll_joint": 139,
                    ".*_hip_pitch_joint": 88, ".*_knee_joint": 139,
                    "waist_yaw_joint": 88, "waist_roll_joint": 35,
                    "waist_pitch_joint": 35},
    },
    "feet": {
        "stiffness": {".*_ankle_pitch_joint": 20, ".*_ankle_roll_joint": 20},
        "damping": {".*_ankle_pitch_joint": 2, ".*_ankle_roll_joint": 2},
        "effort": {".*_ankle_pitch_joint": 35, ".*_ankle_roll_joint": 35},
    },
    "shoulders": {
        "stiffness": {".*_shoulder_pitch_joint": 100, ".*_shoulder_roll_joint": 100},
        "damping": {".*_shoulder_pitch_joint": 2, ".*_shoulder_roll_joint": 2},
        "effort": {".*_shoulder_pitch_joint": 25, ".*_shoulder_roll_joint": 25},
    },
    "arms": {
        "stiffness": {".*_shoulder_yaw_joint": 50, ".*_elbow_joint": 50},
        "damping": {".*_shoulder_yaw_joint": 2, ".*_elbow_joint": 2},
        "effort": {".*_shoulder_yaw_joint": 25, ".*_elbow_joint": 25},
    },
    "wrist": {
        "stiffness": {".*_wrist_yaw_joint": 40, ".*_wrist_roll_joint": 40, ".*_wrist_pitch_joint": 40},
        "damping": {".*_wrist_yaw_joint": 2, ".*_wrist_roll_joint": 2, ".*_wrist_pitch_joint": 2},
        "effort": {".*_wrist_yaw_joint": 5, ".*_wrist_roll_joint": 25, ".*_wrist_pitch_joint": 5},
    },
}

def _resolve_pd(name, key):
    for grp in _G1_PD_GROUPS.values():
        for pat, val in grp[key].items():
            if re.match(pat, name):
                return val
    return 0.0


def _cfg_get_compat(cfg, key, legacy_keys, default=None, required=False):
    """Read a config value while supporting legacy key names."""
    if key in cfg:
        return cfg[key]
    for legacy_key in legacy_keys:
        if legacy_key in cfg:
            return cfg[legacy_key]
    if required:
        searched = ", ".join([key, *legacy_keys])
        raise KeyError(f"Missing required config key. Tried: {searched}")
    return default

def get_training_pd_gains(joint_names):
    kp = np.array([_resolve_pd(n, "stiffness") for n in joint_names], dtype=np.float32)
    kd = np.array([_resolve_pd(n, "damping") for n in joint_names], dtype=np.float32)
    return kp, kd

def get_training_effort_limits(joint_names):
    return np.array([_resolve_pd(n, "effort") for n in joint_names], dtype=np.float32)

# =============================================================================
# Projected gravity (matches Isaac Lab projected_gravity_b)
# =============================================================================

def get_projected_gravity(mj_model, mj_data, body_name="pelvis"):
    """Gravity unit vector [gx, gy, gz] in body frame."""
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found")
    rot = mj_data.xmat[body_id].reshape(3, 3)
    return (rot.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)).astype(np.float32)


def _quat_conjugate(quat_wxyz):
    qw, qx, qy, qz = quat_wxyz
    return np.array([qw, -qx, -qy, -qz], dtype=np.float64)


def _quat_mul(q1_wxyz, q2_wxyz):
    w1, x1, y1, z1 = q1_wxyz
    w2, x2, y2, z2 = q2_wxyz
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _rotmat_from_quat(quat_wxyz):
    qw, qx, qy, qz = quat_wxyz
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def get_root_local_rot_tan_norm(mj_data):
    """Match mdp.root_local_rot_tan_norm for policies that use the 6D root orientation term."""
    root_quat = mj_data.qpos[3:7].astype(np.float64)
    qw, qx, qy, qz = root_quat
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    yaw_quat = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float64)
    root_quat_local = _quat_mul(_quat_conjugate(yaw_quat), root_quat)
    root_rotm_local = _rotmat_from_quat(root_quat_local)
    tan_vec = root_rotm_local[:, 0]
    norm_vec = root_rotm_local[:, 2]
    return np.concatenate([tan_vec, norm_vec]).astype(np.float32)

# =============================================================================
# Height scanner (aligned with training RayCaster config)
# =============================================================================

class HeightScanner:
    """
    Ray-based height scanner mimicking Isaac Lab RayCaster.

    Config from training:
      prim_body_name = "torso_link"
      resolution = 0.04
      size = (0.4, 0.4)  → 11×11 grid
      offset = (0.20, 0.0, 20.0)
    """

    def __init__(self, resolution=0.04, scan_size=(0.4, 0.4),
                 offset=(0.20, 0.0, 20.0), body_name="torso_link"):
        self.resolution = resolution
        self.scan_size = scan_size
        self.offset = offset
        self.body_name = body_name

        self.nx = int(scan_size[0] / resolution) + 1  # 11
        self.ny = int(scan_size[1] / resolution) + 1  # 11

        # Grid in prim frame, centered at offset
        ox, oy, oz = offset
        hx, hy = scan_size[0] / 2, scan_size[1] / 2
        self.grid_x = np.linspace(ox - hx, ox + hx, self.nx)  # [0.0, 0.04, ..., 0.4]
        self.grid_y = np.linspace(oy - hy, oy + hy, self.ny)  # [-0.2, -0.16, ..., 0.2]

        self._body_id = -1
        self._robot_geom_ids = set()
        self.ray_hits_w = np.full((self.nx * self.ny, 3), np.nan, dtype=np.float32)

    def bind(self, mj_model):
        self._body_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_BODY, self.body_name)
        if self._body_id < 0:
            raise ValueError(f"Body '{self.body_name}' not found")

        # Collect all robot geom ids for self-exclusion
        root_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        robot_bodies = {root_id}
        for i in range(mj_model.nbody):
            if int(mj_model.body_parentid[i]) in robot_bodies:
                robot_bodies.add(i)
        self._robot_geom_ids = set()
        for i in range(mj_model.ngeom):
            if int(mj_model.geom_bodyid[i]) in robot_bodies:
                self._robot_geom_ids.add(i)

    def _yaw_rot(self, quat_wxyz):
        qw, qx, qy, qz = quat_wxyz
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

    def scan(self, mj_model, mj_data):
        """
        Returns height_scan: (nx*ny,) flat array.
        height_scan = (body_z - hit_z - norm_offset) * scale
        """
        body_pos = mj_data.xpos[self._body_id].copy()
        body_quat = mj_data.xquat[self._body_id].copy()
        R_yaw = self._yaw_rot(body_quat)

        hits_w = self.ray_hits_w
        ray_dir = np.array([[0.0], [0.0], [-1.0]], dtype=np.float64)
        geom_id_buf = np.zeros((1, 1), dtype=np.int32)
        body_z = float(body_pos[2])
        ray_z = body_z + float(self.offset[2])  # torso_z + 20

        heights = np.zeros(self.nx * self.ny, dtype=np.float32)

        for ix, lx in enumerate(self.grid_x):
            for iy, ly in enumerate(self.grid_y):
                idx = iy * self.nx + ix
                world_xy = body_pos[:2] + R_yaw[:2, :2] @ np.array([lx, ly])
                origin = np.array([[world_xy[0]], [world_xy[1]], [ray_z]], dtype=np.float64)

                hit_world = None
                search_origin = origin.copy()
                for _ in range(32):
                    geom_id_buf[0, 0] = -1
                    dist = mujoco.mj_ray(mj_model, mj_data, search_origin,
                                         ray_dir, None, 1, -1, geom_id_buf)
                    if dist < 0:
                        break
                    if int(geom_id_buf[0, 0]) not in self._robot_geom_ids:
                        hit_world = search_origin[:, 0] + ray_dir[:, 0] * dist
                        break
                    search_origin[:, 0] += ray_dir[:, 0] * (dist + 1e-3)

                if hit_world is None:
                    hit_world = np.array([world_xy[0], world_xy[1], 0.0], dtype=np.float64)
                    hits_w[idx] = np.nan
                else:
                    hits_w[idx] = hit_world.astype(np.float32)
                raw_h = body_z - float(hit_world[2])
                heights[idx] = np.clip(raw_h, 0.0, 1.5)

                # height_scan = (sensor_z - hit_z - norm_offset) * scale
                # sensor_z = torso_z + 20 (ray origin z), matching Isaac Lab pos_w
                # heights[idx] = float(ray_z) - float(hit_world[2])

        return heights, hits_w


def build_height_scan_xyz(hits_world, body_quat_wxyz, body_pos_w, norm_offset):
    """Match mdp.height_scan_xyz by expressing ray hits in the torso yaw frame."""
    if hits_world is None:
        return None

    relative_pos_w = hits_world.astype(np.float64) - body_pos_w.astype(np.float64)[None, :]
    qw, qx, qy, qz = body_quat_wxyz.astype(np.float64)
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    c, s = np.cos(yaw), np.sin(yaw)
    rot_inv = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    coords = relative_pos_w @ rot_inv.T
    coords[:, 2] = -coords[:, 2] - norm_offset
    coords = np.nan_to_num(coords, nan=0.0, posinf=1.0, neginf=-1.0)
    return coords.reshape(-1).astype(np.float32)


def infer_obs_layout(expected_obs_dim, configured_history_length, enable_height_scan):
    """Infer the observation layout from the checkpoint's expected actor input size."""
    known_layouts = {
        217: {"single_frame_dim": 96, "history_length": 1, "height_scan_mode": "scalar121"},
        495: {"single_frame_dim": 99, "history_length": 5, "height_scan_mode": None},
        601: {"single_frame_dim": 96, "history_length": 5, "height_scan_mode": "scalar121"},
        858: {"single_frame_dim": 99, "history_length": 5, "height_scan_mode": "xyz363"},
    }
    if expected_obs_dim in known_layouts:
        return known_layouts[expected_obs_dim]

    height_scan_mode = "scalar121" if enable_height_scan else None
    height_scan_dim = 121 if height_scan_mode == "scalar121" else 0
    if expected_obs_dim > height_scan_dim and (expected_obs_dim - height_scan_dim) % configured_history_length == 0:
        return {
            "single_frame_dim": (expected_obs_dim - height_scan_dim) // configured_history_length,
            "history_length": configured_history_length,
            "height_scan_mode": height_scan_mode,
        }
    return {
        "single_frame_dim": 96,
        "history_length": configured_history_length,
        "height_scan_mode": height_scan_mode,
    }


def normalize_checkpoint_state_dict(checkpoint):
    """Support actor_state_dict / model_state_dict / state_dict checkpoint formats."""
    actor_state_dict = checkpoint.get("actor_state_dict")
    if isinstance(actor_state_dict, dict):
        mapped_state_dict = {}
        for key, value in actor_state_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key == "distribution.std_param":
                mapped_state_dict["std"] = value
            elif key.startswith("proprio_embedding."):
                mapped_state_dict[f"actor_proprio_embedding.{key.split('.', 1)[1]}"] = value
            elif key.startswith("mlp."):
                mapped_state_dict[f"actor.{key.split('.', 1)[1]}"] = value
            else:
                mapped_state_dict[key] = value
        return mapped_state_dict

    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    if isinstance(state_dict, dict):
        return {key: value for key, value in state_dict.items() if isinstance(value, torch.Tensor)}

    raise RuntimeError(
        "Unsupported checkpoint format. Expected actor_state_dict, model_state_dict, or state_dict."
    )


def infer_actor_input_dim(state_dict):
    """Infer actor input dim from common checkpoint key layouts."""
    candidate_keys = [
        "actor.0.weight",
        "actor.1.weight",
        "actor.mlp.0.weight",
        "mlp.0.weight",
    ]
    for key in candidate_keys:
        tensor = state_dict.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
            return int(tensor.shape[1])

    for key, value in state_dict.items():
        if key.startswith("actor.") and key.endswith(".weight") and isinstance(value, torch.Tensor) and value.ndim == 2:
            return int(value.shape[1])

    raise KeyError("Could not infer actor input dim from checkpoint state_dict.")

# =============================================================================
# Observation builder (aligned with BaseEnv.compute_observations)
# =============================================================================

def build_actor_obs(ang_vel, orientation_obs, command, joint_pos, joint_vel,
                    last_action, default_joint_pos, num_actions,
                    ang_vel_scale, dof_pos_scale, dof_vel_scale, cmd_scale,
                    obs_history, height_scan_obs=None):
    """Build actor observation matching the inferred checkpoint layout."""
    num_single = obs_history.shape[1]

    joint_pos_rel = (joint_pos - default_joint_pos) * dof_pos_scale

    single = np.zeros(num_single, dtype=np.float32)
    single[0:3] = ang_vel * ang_vel_scale
    orient_dim = len(orientation_obs)
    single[3:3 + orient_dim] = orientation_obs
    cmd_start = 3 + orient_dim
    single[cmd_start:cmd_start + 3] = command * cmd_scale
    joint_start = cmd_start + 3
    single[joint_start:joint_start + num_actions] = joint_pos_rel
    vel_start = joint_start + num_actions
    single[vel_start:vel_start + num_actions] = joint_vel * dof_vel_scale
    action_start = vel_start + num_actions
    single[action_start:action_start + num_actions] = last_action

    obs_history[:-1] = obs_history[1:]
    obs_history[-1] = single
    stacked = obs_history.reshape(-1)

    if height_scan_obs is not None:
        return np.concatenate([stacked, height_scan_obs], dtype=np.float32)
    return stacked

# =============================================================================
# Policy wrapper
# =============================================================================

class PolicyRunner:
    """Wraps ActorCritic or ActorCriticRecurrent for sim2sim inference."""

    def __init__(self, policy, device, is_recurrent=False):
        self.policy = policy
        self.device = device
        self.is_recurrent = is_recurrent

    def reset(self):
        """Reset RNN hidden states if recurrent."""
        if self.is_recurrent:
            self.policy.reset()

    def act(self, obs):
        """Run inference, return action mean."""
        obs_t = torch.from_numpy(obs).float().reshape(1, -1).to(self.device)
        with torch.inference_mode():
            return self.policy.act_inference(obs_t).detach().cpu().numpy().reshape(-1)

# =============================================================================
# PD Controller
# =============================================================================

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

# =============================================================================
# Gamepad
# =============================================================================

def _deadzone(v, dz):
    return 0.0 if abs(v) < dz else v

def gamepad_reader(joystick, cmd, cmd_lock, running, deadzone, cmd_gain):
    while running[0]:
        try:
            pygame.event.pump()
            ly = -joystick.get_axis(1)  # left stick y → forward
            lx = -joystick.get_axis(0)  # left stick x → lateral
            rx = -joystick.get_axis(3)  # right stick x → yaw
            ly = _deadzone(ly, deadzone)
            lx = _deadzone(lx, deadzone)
            rx = _deadzone(rx, deadzone)
            with cmd_lock:
                cmd[:] = [ly * cmd_gain[0], lx * cmd_gain[1], rx * cmd_gain[2]]
        except Exception:
            pass
        time.sleep(0.01)

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MuJoCo sim2sim for LeggedLab G1 policies")
    parser.add_argument("--policy_path", type=str, default=DEFAULT_POLICY_PATH,
                        help=f"Path to checkpoint .pt file (default: {DEFAULT_POLICY_PATH})")
    parser.add_argument("--config", type=str, default="g1_29dof.yaml",
                        help="Path to YAML config")
    parser.add_argument("--no_height_scan", action="store_true",
                        help="Disable height scan (use zeros)")
    parser.add_argument("--gamepad", action="store_true",
                        help="Enable gamepad velocity commands")
    parser.add_argument("--print_elevation", action="store_true",
                        help="Print height scan debug info")
    args = parser.parse_args()

    # ---- Load config --------------------------------------------------------
    config_path = os.path.join(SCRIPT_DIR, args.config)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    xml_path = cfg["xml_path"]
    if not os.path.isabs(xml_path):
        xml_path = os.path.join(SCRIPT_DIR, xml_path)

    policy_path = args.policy_path
    if not os.path.isabs(policy_path):
        policy_path = os.path.join(SCRIPT_DIR, policy_path)

    sim_dt = float(cfg["simulation_dt"])
    control_decimation = int(cfg["control_decimation"])
    sim_duration = float(cfg["simulation_duration"])

    num_actions = int(cfg["num_actions"])
    action_scale = float(cfg["action_scale"])
    ang_vel_scale = float(cfg.get("ang_vel_scale", 1.0))
    dof_pos_scale = float(cfg.get("dof_pos_scale", 1.0))
    dof_vel_scale = float(cfg.get("dof_vel_scale", 1.0))
    cmd_scale = np.array(cfg.get("cmd_scale", [1.0, 1.0, 1.0]), dtype=np.float32)
    clip_obs = float(cfg.get("clip_observations", 100.0))
    clip_act = float(cfg.get("clip_actions", 100.0))

    actor_obs_history_length = int(
        _cfg_get_compat(cfg, "actor_obs_history_length", ["policy_history_length"], default=1)
    )

    enable_height_scan = cfg.get("enable_height_scan", True) and not args.no_height_scan
    hs_resolution = float(cfg.get("height_scan_resolution", 0.04))
    hs_size = tuple(cfg.get("height_scan_size", [0.4, 0.4]))
    hs_offset = tuple(
        _cfg_get_compat(cfg, "height_scan_offset", ["height_scan_position_offset"], default=[0.20, 0.0, 20.0])
    )
    hs_norm_offset = float(
        _cfg_get_compat(cfg, "height_scan_norm_offset", ["height_scan_height_offset"], default=0.5)
    )
    hs_scale = 1.0  # obs_scales.height_scan = 1.0 in training

    use_training_pd = bool(cfg.get("use_training_pd", False))
    pd_scale = float(cfg.get("pd_scale", 1.0))

    cmd_init = np.array(
        _cfg_get_compat(cfg, "cmd_init", ["command"], default=[0.0, 0.0, 0.0]),
        dtype=np.float32,
    )
    use_gamepad = bool(cfg.get("use_gamepad", False)) or args.gamepad
    gamepad_deadzone = float(cfg.get("gamepad_deadzone", 0.15))
    gamepad_cmd_gain = np.array(
        _cfg_get_compat(cfg, "gamepad_cmd_gain", ["gamepad_command_gains"], default=[1.0, 0.6, 1.0]),
        dtype=np.float32,
    )
    startup_blend_steps = int(cfg.get("startup_blend_steps", 30))
    startup_cmd_zero_steps = int(cfg.get("startup_cmd_zero_steps", 20))
    policy_delay_steps = int(cfg.get("policy_delay_steps", 0))
    print_elevation = bool(cfg.get("print_elevation_z", False)) or args.print_elevation

    # ---- Policy metadata / expected observation layout ---------------------
    policy_joint_order = cfg.get("policy_joint_order", POLICY_JOINT_ORDER)
    checkpoint = torch.load(policy_path, map_location="cpu", weights_only=False)
    state_dict = normalize_checkpoint_state_dict(checkpoint)
    is_recurrent = any("memory_a" in k for k in state_dict.keys())
    if is_recurrent:
        expected_obs_dim = state_dict["memory_a.rnn.weight_ih_l0"].shape[1]
    else:
        expected_obs_dim = infer_actor_input_dim(state_dict)

    obs_layout = infer_obs_layout(expected_obs_dim, actor_obs_history_length, enable_height_scan)
    num_single_frame = obs_layout["single_frame_dim"]
    actor_obs_history_length = obs_layout["history_length"]
    height_scan_mode = obs_layout["height_scan_mode"]
    height_scan_requested = height_scan_mode is not None
    hs_nx = int(hs_size[0] / hs_resolution) + 1
    hs_ny = int(hs_size[1] / hs_resolution) + 1
    hs_dim = 0
    if height_scan_mode == "scalar121":
        hs_dim = hs_nx * hs_ny
    elif height_scan_mode == "xyz363":
        hs_dim = hs_nx * hs_ny * 3
    num_actor_obs = expected_obs_dim

    # ---- Load MuJoCo model -------------------------------------------------
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = sim_dt

    # Get MuJoCo joint names (excluding free joint)
    mujoco_joint_names = []
    for i in range(mj_model.njnt):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name != "floating_base_joint":
            mujoco_joint_names.append(name)

    num_mujoco_joints = len(mujoco_joint_names)
    print(f"MuJoCo joints: {num_mujoco_joints}")

    # Build policy → MuJoCo index mapping
    policy_to_mujoco = []
    for pname in policy_joint_order:
        try:
            policy_to_mujoco.append(mujoco_joint_names.index(pname))
        except ValueError:
            print(f"WARNING: policy joint '{pname}' not in MuJoCo XML!")
            policy_to_mujoco.append(-1)

    # ---- PD gains ----------------------------------------------------------
    if use_training_pd:
        kps, kds = get_training_pd_gains(mujoco_joint_names)
        torque_limit = get_training_effort_limits(mujoco_joint_names)
    else:
        kps = np.array(_cfg_get_compat(cfg, "stiffness", [], required=True), dtype=np.float32)
        kds = np.array(_cfg_get_compat(cfg, "damping", [], required=True), dtype=np.float32)
        torque_limit = np.array(
            _cfg_get_compat(cfg, "torque_limit", ["effort_limits"], required=True),
            dtype=np.float32,
        )

    if pd_scale != 1.0:
        kps *= pd_scale
        kds *= pd_scale

    # ---- Default joint angles (policy order) --------------------------------
    default_joints_yaml = np.array(cfg.get("default_joint_angles",
                                   DEFAULT_JOINT_ANGLES_POLICY.tolist()), dtype=np.float32)

    # ---- Height scanner -----------------------------------------------------
    scanner = None
    if height_scan_requested and not args.no_height_scan:
        scanner = HeightScanner(
            resolution=hs_resolution, scan_size=hs_size,
            offset=hs_offset, body_name="torso_link")
        scanner.bind(mj_model)
        print(f"Height scanner: {scanner.nx}x{scanner.ny} grid, offset={hs_offset}")

    # ---- Initialize simulation state ----------------------------------------
    mujoco.mj_resetData(mj_model, mj_data)
    mj_data.qpos[0:3] = [0.0, 0.0, 0.8]
    mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

    # Set initial joint positions using yaml defaults mapped to MuJoCo order
    init_q_mujoco = np.zeros(num_mujoco_joints, dtype=np.float32)
    for pi, mi in enumerate(policy_to_mujoco):
        if mi >= 0:
            init_q_mujoco[mi] = default_joints_yaml[pi]
    mj_data.qpos[7:7 + num_mujoco_joints] = init_q_mujoco
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    # ---- Load policy --------------------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Detect policy type from state dict keys
    print(f"Policy type: {'ActorCriticRecurrent (LSTM)' if is_recurrent else 'ActorCritic (MLP)'}")
    print("=" * 70)
    print("LeggedLab MuJoCo Sim2Sim Deploy")
    print("=" * 70)
    print(f"  XML:          {xml_path}")
    print(f"  Policy:       {policy_path}")
    print(f"  Actions:      {num_actions}")
    print(f"  Single-frame: {num_single_frame}")
    print(f"  History:      {actor_obs_history_length}")
    print(
        f"  Height scan:  {hs_dim} dims ({hs_nx}x{hs_ny}) mode={height_scan_mode}"
        if height_scan_requested else "  Height scan:  disabled"
    )
    print(f"  Actor obs:    {num_actor_obs}")
    print(f"  Action scale: {action_scale}")
    print("=" * 70)

    if is_recurrent:
        from rsl_rl.modules import ActorCriticRecurrent

        # Infer rnn_hidden_dim from state dict shape
        rnn_hidden_dim = state_dict["memory_a.rnn.weight_hh_l0"].shape[1]  # 256

        policy = ActorCriticRecurrent(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_actor_obs + 5,  # dummy, critic not used
            num_actions=num_actions,
            actor_hidden_dims=[256, 256, 128],   # from G1RoughAgentCfg
            critic_hidden_dims=[256, 256, 128],
            activation="elu",
            rnn_type="lstm",
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=1,
        ).to(device)
    else:
        from rsl_rl.modules import ActorCritic

        policy = ActorCritic(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_actor_obs + 5,
            num_actions=num_actions,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        ).to(device)

    policy.load_state_dict(state_dict, strict=False)
    policy.eval()
    print("Model loaded.")

    runner = PolicyRunner(policy, device, is_recurrent)

    # ---- Warmup -------------------------------------------------------------
    warm_obs = np.zeros(num_actor_obs, dtype=np.float32)
    if num_single_frame == 96:
        warm_obs[5] = -1.0  # gravity z ≈ -1
    elif num_single_frame == 99:
        warm_obs[3] = 1.0   # tan x
        warm_obs[8] = 1.0   # norm z
    if is_recurrent:
        policy.reset()  # 只重置一次, 然后隐藏状态跨 warmup 步骤传递
    for _ in range(3):
        runner.act(warm_obs)
    print("Warmup done.")

    # ---- Init last_action from current posture ------------------------------
    qj = mj_data.qpos[7:]
    dqj = mj_data.qvel[6:]
    last_action = np.zeros(num_actions, dtype=np.float32)
    for pi, mi in enumerate(policy_to_mujoco):
        if mi >= 0 and mi < len(qj):
            last_action[pi] = (qj[mi] - default_joints_yaml[pi]) / action_scale

    target_dof_pos = init_q_mujoco.copy()
    cmd = cmd_init.copy()

    # ---- Gamepad ------------------------------------------------------------
    pygame.init()
    pygame.joystick.init()
    cmd_lock = threading.Lock()
    running = [True]

    if use_gamepad and pygame.joystick.get_count() > 0:
        js = pygame.joystick.Joystick(0)
        js.init()
        print(f"Gamepad: {js.get_name()}")
        t = threading.Thread(target=gamepad_reader,
                             args=(js, cmd, cmd_lock, running, gamepad_deadzone, gamepad_cmd_gain),
                             daemon=True)
        t.start()
    else:
        print(f"Command fixed: {cmd_init}")

    # ---- Observation history buffer (if history > 1) ------------------------
    obs_history = np.zeros((actor_obs_history_length, num_single_frame), dtype=np.float32)
    if num_single_frame == 96:
        obs_history[-1, 5] = -1.0
    elif num_single_frame == 99:
        obs_history[-1, 3] = 1.0
        obs_history[-1, 8] = 1.0

    # ---- Main loop ----------------------------------------------------------
    startup_step = 0
    policy_step = 0
    mj_per_step = sim_dt * control_decimation

    print("\nStarting simulation...\n")

    with mujoco.viewer.launch_passive(mj_model, mj_data,
                                      show_left_ui=True, show_right_ui=True) as viewer:
        while viewer.is_running() and mj_data.time < sim_duration:
            step_start = time.time()

            # Policy inference
            if policy_step >= policy_delay_steps:
                with cmd_lock:
                    cur_cmd = cmd.copy()
                if startup_step < startup_cmd_zero_steps:
                    cur_cmd[:] = 0.0

                qj = mj_data.qpos[7:]
                dqj = mj_data.qvel[6:]
                omega = mj_data.qvel[3:6]
                if num_single_frame == 96:
                    orientation_obs = get_projected_gravity(mj_model, mj_data)
                else:
                    orientation_obs = get_root_local_rot_tan_norm(mj_data)

                # Reorder to policy order
                qj_pol = np.zeros(num_actions, dtype=np.float32)
                dqj_pol = np.zeros(num_actions, dtype=np.float32)
                for pi, mi in enumerate(policy_to_mujoco):
                    if mi >= 0 and mi < len(qj):
                        qj_pol[pi] = qj[mi]
                        dqj_pol[pi] = dqj[mi]

                # Height scan
                hs_raw = None
                hs_hits = None
                if scanner is not None:
                    hs_raw, hs_hits = scanner.scan(mj_model, mj_data)

                height_scan_obs = None
                if height_scan_mode == "scalar121":
                    if hs_raw is not None:
                        height_scan_obs = (hs_raw - hs_norm_offset) * hs_scale
                    else:
                        height_scan_obs = np.zeros(hs_dim, dtype=np.float32)
                elif height_scan_mode == "xyz363":
                    if hs_hits is not None:
                        torso_body_id = scanner._body_id if scanner is not None else mujoco.mj_name2id(
                            mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
                        )
                        body_quat = mj_data.xquat[torso_body_id].copy()
                        body_pos = mj_data.xpos[torso_body_id].copy()
                        height_scan_obs = build_height_scan_xyz(hs_hits, body_quat, body_pos, hs_norm_offset)
                    else:
                        height_scan_obs = np.zeros(hs_dim, dtype=np.float32)

                # Build observation
                obs = build_actor_obs(
                    omega, orientation_obs, cur_cmd, qj_pol, dqj_pol, last_action,
                    default_joints_yaml, num_actions,
                    ang_vel_scale, dof_pos_scale, dof_vel_scale, cmd_scale,
                    obs_history, height_scan_obs)

                # Clip observation
                obs = np.clip(obs, -clip_obs, clip_obs)

                # Inference
                t0 = time.perf_counter()
                raw_action = runner.act(obs)
                inf_ms = (time.perf_counter() - t0) * 1000

                raw_action = np.clip(raw_action, -clip_act, clip_act)

                # Startup blend
                if startup_blend_steps > 0:
                    alpha = min(float(startup_step + 1) / startup_blend_steps, 1.0)
                else:
                    alpha = 1.0
                action = (1.0 - alpha) * last_action + alpha * raw_action
                last_action = raw_action.copy()

                # Action → target joint angles (policy → MuJoCo)
                loco_target = action * action_scale + default_joints_yaml
                for pi, mi in enumerate(policy_to_mujoco):
                    if mi >= 0 and mi < num_mujoco_joints:
                        target_dof_pos[mi] = loco_target[pi]

                startup_step += 1

                if policy_step % 50 == 0:
                    print(f"[step {policy_step}] inf={inf_ms:.1f}ms  "
                          f"cmd=[{cur_cmd[0]:.2f},{cur_cmd[1]:.2f},{cur_cmd[2]:.2f}]  "
                          f"action_range=[{raw_action.min():.2f},{raw_action.max():.2f}]")

                if print_elevation and scanner is not None and policy_step % 25 == 0:
                    hs_grid = hs_raw.reshape(scanner.ny, scanner.nx)
                    print(f"--- height_scan step={policy_step} ---")
                    print(f"  mean={hs_grid.mean():.3f} min={hs_grid.min():.3f} max={hs_grid.max():.3f}")
                    for iy in range(scanner.ny):
                        row = " ".join(f"{v:6.3f}" for v in hs_grid[iy])
                        print(f"  y[{iy:02d}] {row}")

            # PD control + physics steps
            for _ in range(control_decimation):
                q_now = mj_data.qpos[7:7 + num_mujoco_joints]
                dq_now = mj_data.qvel[6:6 + num_mujoco_joints]
                tau = pd_control(target_dof_pos, q_now, kps,
                                 np.zeros(num_mujoco_joints), dq_now, kds)
                tau = np.clip(tau, -torque_limit, torque_limit)
                mj_data.ctrl[:num_mujoco_joints] = tau
                mujoco.mj_step(mj_model, mj_data)

            # Draw hit markers
            if scanner is not None:
                valid = np.isfinite(scanner.ray_hits_w).all(axis=1)
                pts = scanner.ray_hits_w[valid]
                ngeom = min(len(pts), viewer.user_scn.maxgeom)
                viewer.user_scn.ngeom = ngeom
                for i in range(ngeom):
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[i],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=np.array([0.012, 0.0, 0.0], dtype=np.float64),
                        pos=pts[i],
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32),
                    )
            else:
                viewer.user_scn.ngeom = 0

            viewer.sync()

            elapsed = time.time() - step_start
            if elapsed < mj_per_step:
                time.sleep(mj_per_step - elapsed)

            policy_step += 1

    running[0] = False
    print("Simulation ended.")

if __name__ == "__main__":
    main()
