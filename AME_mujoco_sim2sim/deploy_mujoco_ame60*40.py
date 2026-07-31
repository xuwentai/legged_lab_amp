#!/usr/bin/env python3
"""
MuJoCo sim2sim for LeggedLab-trained G1 policies (AME Encoder).

Loads an ActorCriticEncoder checkpoint with CNN + MultiheadAttention terrain
encoder and deploys it in MuJoCo.  Observaion construction matches the AME
training pipeline: height_scan outputs xyz (yaw-aligned body frame, 16×11×3).

Usage:
  cd AME_mujoco_sim2sim
  python deploy_mujoco_ame.py --policy_path ../logs/g1_rough/.../model_XX.pt --gamepad
"""
import time
import os
import sys
import argparse
import threading
import re
import math
import numpy as np
import mujoco
import mujoco.viewer
import torch
import yaml
import pygame

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))

# =============================================================================
# Configurable history lengths (must match g1_config.py training settings)
# =============================================================================

DEPLOY_HEIGHT_SCAN_COORD_DIM = 3       # xyz
DEPLOY_HEIGHT_SCAN_RESOLUTION = 0.04
DEPLOY_HEIGHT_SCAN_SIZE = (0.6, 0.4)   # -> 16x11 grid
DEPLOY_HEIGHT_SCAN_OFFSET = (0.30, 0.0, 20.0)  # x coverage [0.0, 0.6] m
DEPLOY_HEIGHT_SCAN_HISTORY_LENGTH = 1   # height_scan stacked frames
DEPLOY_ACTOR_OBS_HISTORY_LENGTH = 1   # proprioception (full obs) stacked frames
DEPLOY_CRITIC_OBS_HISTORY_LENGTH = 1   # critic not used in deploy, keep for consistency
# Critic每帧比actor多5维: root_lin_vel(3) + feet_contact(2) (见base_env.py)
CRITIC_EXTRA_PER_FRAME = 5

# Derived map_scan_dim (same logic as g1_config.py G1_MAP_SCAN_DIM)
_DEPLOY_HS_NX = int(DEPLOY_HEIGHT_SCAN_SIZE[0] / DEPLOY_HEIGHT_SCAN_RESOLUTION) + 1
_DEPLOY_HS_NY = int(DEPLOY_HEIGHT_SCAN_SIZE[1] / DEPLOY_HEIGHT_SCAN_RESOLUTION) + 1
DEPLOY_MAP_SCAN_DIM = (_DEPLOY_HS_NX, _DEPLOY_HS_NY, DEPLOY_HEIGHT_SCAN_COORD_DIM)  # (16, 11, 3)

# =============================================================================
# Joint order constants
# =============================================================================

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
# PD gains from G1_CFG actuators (unitree.py original)
# =============================================================================

# PD gains 必须与 unitree.py G1_CFG (Isaac Lab 训练配置) 完全一致。
# 这些数值是 unitree.py 的部署镜像；修改训练侧 actuator 后必须同步此处。
_G1_PD_GROUPS = {
    "legs": {
        # "stiffness": {".*_hip_yaw_joint": 200, ".*_hip_roll_joint": 200,
        #                ".*_hip_pitch_joint": 200, ".*_knee_joint": 300,
        #                ".*waist.*": 200},
        # "damping": {".*_hip_yaw_joint": 5, ".*_hip_roll_joint": 5,
        #              ".*_hip_pitch_joint": 5, ".*_knee_joint": 6,
        #              ".*waist.*": 5},

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

def get_training_pd_gains(joint_names):
    kp = np.array([_resolve_pd(n, "stiffness") for n in joint_names], dtype=np.float32)
    kd = np.array([_resolve_pd(n, "damping") for n in joint_names], dtype=np.float32)
    return kp, kd

def get_training_effort_limits(joint_names):
    return np.array([_resolve_pd(n, "effort") for n in joint_names], dtype=np.float32)

# =============================================================================
# Projected gravity
# =============================================================================

def get_projected_gravity(mj_model, mj_data, body_name="pelvis"):
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found")
    rot = mj_data.xmat[body_id].reshape(3, 3)
    return (rot.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)).astype(np.float32)

# =============================================================================
# Height scanner (xyz output, matching _compute_elevation_map_xyz)
# =============================================================================

class HeightScannerXYZ:
    """
    Ray-based height scanner, output xyz in body yaw-aligned frame.

    z formula: (body_z - hit_z - offset) * scale  (same as original)
    x, y:       horizontal offsets in yaw-rotated body frame
    """

    def __init__(self, resolution=0.04, scan_size=(0.6, 0.4),
                 offset=(0.30, 0.0, 20.0), body_name="torso_link",
                 height_scale=1.0, height_offset=0.5):
        self.resolution = resolution
        self.scan_size = scan_size
        self.offset = offset
        self.body_name = body_name
        self.height_scale = height_scale
        self.height_offset = height_offset

        self.nx = int(scan_size[0] / resolution) + 1
        self.ny = int(scan_size[1] / resolution) + 1

        ox, oy, oz = offset
        hx, hy = scan_size[0] / 2, scan_size[1] / 2
        self.grid_x = np.linspace(ox - hx, ox + hx, self.nx)
        self.grid_y = np.linspace(oy - hy, oy + hy, self.ny)

        self._body_id = -1
        self._robot_geom_ids = set()
        self.ray_hits_w = np.full((self.nx * self.ny, 3), np.nan, dtype=np.float32)

    def bind(self, mj_model):
        self._body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, self.body_name)
        if self._body_id < 0:
            raise ValueError(f"Body '{self.body_name}' not found")
        root_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        robot_bodies = {root_id}
        for i in range(mj_model.nbody):
            if int(mj_model.body_parentid[i]) in robot_bodies:
                robot_bodies.add(i)
        self._robot_geom_ids = set()
        for i in range(mj_model.ngeom):
            if int(mj_model.geom_bodyid[i]) in robot_bodies:
                self._robot_geom_ids.add(i)

    def _yaw_rot_matrix(self, quat_wxyz):
        qw, qx, qy, qz = quat_wxyz
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

    def scan_xyz(self, mj_model, mj_data):
        """Returns: xyz flat array [nx*ny*3], interleaved as x0,y0,z0, x1,y1,z1, ..."""
        body_pos = mj_data.xpos[self._body_id].copy()
        body_quat = mj_data.xquat[self._body_id].copy()
        R_yaw = self._yaw_rot_matrix(body_quat)

        ray_dir = np.array([[0.0], [0.0], [-1.0]], dtype=np.float64)
        geom_id_buf = np.zeros((1, 1), dtype=np.int32)
        body_z = float(body_pos[2])
        ray_z = body_z + float(self.offset[2])

        hits_w = self.ray_hits_w
        world_hits = np.full((self.nx * self.ny, 3), np.nan, dtype=np.float64)
        xyz_flat = np.zeros(self.nx * self.ny * 3, dtype=np.float32)

        for ix, lx in enumerate(self.grid_x):
            for iy, ly in enumerate(self.grid_y):
                idx_flat = (iy * self.nx + ix) * 3
                idx = iy * self.nx + ix
                # world xy of this grid point
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
                    world_hits[idx] = hit_world

                # # x, y in body yaw frame (horizontal offsets)
                # rel_world = hit_world - np.append(body_pos[:2], 0.0)
                # rel_body = R_yaw.T @ rel_world
                # xyz_flat[idx_flat + 0] = float(rel_body[0]) * self.height_scale
                # xyz_flat[idx_flat + 1] = float(rel_body[1]) * self.height_scale
                # # z: same formula as original (body_z - hit_z - offset) * scale
                # xyz_flat[idx_flat + 2] = (body_z - float(hit_world[2]) - self.height_offset) * self.height_scale
                # # print(xyz_flat[idx_flat + 2])

                rel_world = hit_world - np.append(body_pos[:2], 0.0)
                rel_body = R_yaw.T @ rel_world
                xyz_flat[idx_flat + 0] = float(rel_body[0]) * self.height_scale
                xyz_flat[idx_flat + 1] = float(rel_body[1]) * self.height_scale
                # z: same formula as original (body_z - hit_z - offset) * scale
                z_raw = (body_z - float(hit_world[2]) - self.height_offset) * self.height_scale
                xyz_flat[idx_flat + 2] = np.clip(z_raw, -0.5, 0.7)
        self._world_hits = world_hits  # for visualization
        return xyz_flat
#
#   推导

#   TerrainScanner 用的是 RoboJuDo 的 z 公式：

#   TerrainScanner:  z = ground_z - body_z
#      裁剪范围:      clip(z, -1.2, 0.0)

#   HeightScannerXYZ 用的是训练格式的 z 公式：

#   HeightScannerXYZ:  train_z = (body_z - ground_z - 0.5) * 1.0

#   换算：

#   train_z = -z - 0.5

#   z = -1.2  →  train_z = -(-1.2) - 0.5 = 0.7
#   z =  0.0  →  train_z = -(0.0)  - 0.5 = -0.5

#   所以 clip(z_raw, -0.5, 0.7) 和 TertainScanner 的 clip(z, -1.2, 0.0) 是等价的。

#   这两个值代表什么物理意义

#   train_z = 0.7 对应 RoboJuDo 的 z = -1.2，即地面比身体低
#   1.2m。裁剪的作用是防止射线穿透平台打到地面的极端负值污染观测。

#   但 -1.2 / -0.5 这两个数本身没有特别的物理含义——它们是训练时为了截断异常值而设置的经验阈值，我们只是为了保持一致才做
#   了等价变换。如果训练时用的 clip 是别的值，部署时也应该随之调整。


#

# =============================================================================
# Observation builder
# =============================================================================

def build_actor_obs_ame(ang_vel, projected_gravity, command, joint_pos, joint_vel,
                         last_action, default_joint_pos, num_actions,
                         ang_vel_scale, dof_pos_scale, dof_vel_scale, cmd_scale,
                         height_scan_xyz):
    """
    Build single-frame actor observation for AME encoder.

    Single frame: ang_vel(3) + proj_grav(3) + cmd(3) + jpos(29) + jvel(29) + act(29) = 96
    Append height_scan_xyz: 11*11*3 = 363
    Total: 459
    """
    joint_pos_rel = (joint_pos - default_joint_pos) * dof_pos_scale

    single = np.zeros(96, dtype=np.float32)
    single[0:3] = ang_vel * ang_vel_scale
    single[3:6] = projected_gravity
    single[6:9] = command * cmd_scale
    single[9:9 + num_actions] = joint_pos_rel
    single[9 + num_actions:9 + 2 * num_actions] = joint_vel * dof_vel_scale
    single[9 + 2 * num_actions:9 + 3 * num_actions] = last_action

    if height_scan_xyz is not None:
        return np.concatenate([single, height_scan_xyz], dtype=np.float32)
    return single

# =============================================================================
# Policy wrapper (AME encoder — non-recurrent)
# =============================================================================

class PolicyRunner:
    def __init__(self, policy, device):
        self.policy = policy
        self.device = device

    def reset(self):
        pass

    def act(self, obs):
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
            ly = -joystick.get_axis(1)
            lx = -joystick.get_axis(0)
            rx = -joystick.get_axis(3)
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
    parser = argparse.ArgumentParser(description="MuJoCo sim2sim AME Encoder")
    parser.add_argument("--policy_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="g1_29dof60*40.yaml")
    parser.add_argument("--no_height_scan", action="store_true")
    parser.add_argument("--gamepad", action="store_true")
    args = parser.parse_args()

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
    use_training_pd = bool(cfg.get("use_training_pd", False))
    pd_scale = float(cfg.get("pd_scale", 1.0))
    cmd_init = np.array(cfg["cmd_init"], dtype=np.float32)
    use_gamepad = bool(cfg.get("use_gamepad", False)) or args.gamepad
    gamepad_deadzone = float(cfg.get("gamepad_deadzone", 0.15))
    gamepad_cmd_gain = np.array(cfg.get("gamepad_cmd_gain", [1.0, 0.6, 1.0]), dtype=np.float32)
    startup_blend_steps = int(cfg.get("startup_blend_steps", 30))
    startup_cmd_zero_steps = int(cfg.get("startup_cmd_zero_steps", 20))
    policy_delay_steps = int(cfg.get("policy_delay_steps", 0))
    action_ema_alpha = float(cfg.get("action_ema_alpha", 1))  # EMA smoothing, 0=off 1=raw

#   │ action_ema_alpha │                效果                │
#   ├──────────────────┼────────────────────────────────────┤
#   │ 0.2              │ 非常平滑，但响应慢，可能跟不上指令 │
#   ├──────────────────┼────────────────────────────────────┤
#   │ 0.4 (默认)       │ 中等平滑，推荐起点                 │
#   ├──────────────────┼────────────────────────────────────┤
#   │ 0.6              │ 轻度平滑，接近原始行为             │
#   ├──────────────────┼────────────────────────────────────┤
#   │ 1.0              │ 完全关闭 EMA                       │

    # ---- AME encoder dimensions (use configurable constants) ----
    HS_L, HS_W, HS_C = DEPLOY_MAP_SCAN_DIM  # (16, 11, 3)
    hs_dim = HS_L * HS_W * HS_C             # 528
    hs_history = DEPLOY_HEIGHT_SCAN_HISTORY_LENGTH
    actor_obs_history = DEPLOY_ACTOR_OBS_HISTORY_LENGTH
    num_proprio = 3 + 3 + 3 + num_actions * 3  # 96 for 29 dof
    # Single-frame obs dim; full obs = proprio_history * num_proprio + hs_history * hs_dim
    single_frame_dim = num_proprio + hs_dim   # 459 for history=1
    num_actor_obs = actor_obs_history * num_proprio + hs_history * hs_dim
    height_scan_scale = 1.0
    height_scan_offset = 0.5
    height_scan_frequency = float(cfg.get("height_scan_frequency", 30.0))
    if height_scan_frequency <= 0.0:
        raise ValueError("height_scan_frequency must be positive")
    height_scan_period = 1.0 / height_scan_frequency

    print("=" * 70)
    print("LeggedLab MuJoCo Sim2Sim — AME Encoder (ActorCriticEncoder)")
    print("=" * 70)
    print(f"  XML:          {xml_path}")
    print(f"  Policy:       {policy_path}")
    print(f"  Actions:      {num_actions}")
    print(f"  Proprio:      {num_proprio}")
    print(f"  Height scan:  {hs_dim} dims ({HS_L}×{HS_W} grid, coord_dim={HS_C})")
    print(f"  HS history:   {hs_history} frames")
    print(f"  Obs history:  {actor_obs_history} proprio frames")
    print(f"  Height scan frequency: {height_scan_frequency:.1f} Hz")
    print(f"  Actor obs:    {num_actor_obs}")
    print(f"  Action scale: {action_scale}")
    print(f"  EMA alpha:    {action_ema_alpha}" + (" (disabled)" if action_ema_alpha >= 1.0 else ""))
    print("=" * 70)

    # ---- Load MuJoCo model -------------------------------------------------
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = sim_dt

    mujoco_joint_names = []
    for i in range(mj_model.njnt):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name != "floating_base_joint":
            mujoco_joint_names.append(name)
    num_mujoco_joints = len(mujoco_joint_names)

    policy_to_mujoco = []
    for pname in POLICY_JOINT_ORDER:
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
        kps = np.array(cfg["stiffness"], dtype=np.float32)
        kds = np.array(cfg["damping"], dtype=np.float32)
        torque_limit = np.array(cfg["torque_limit"], dtype=np.float32)
    if pd_scale != 1.0:
        kps *= pd_scale; kds *= pd_scale

    default_joints_yaml = np.array(cfg.get("default_joint_angles",
                                   DEFAULT_JOINT_ANGLES_POLICY.tolist()), dtype=np.float32)

    # ---- Height scanner (xyz) -----------------------------------------------
    if not args.no_height_scan:
        scanner = HeightScannerXYZ(
            resolution=DEPLOY_HEIGHT_SCAN_RESOLUTION,
            scan_size=DEPLOY_HEIGHT_SCAN_SIZE,
            offset=DEPLOY_HEIGHT_SCAN_OFFSET,
            body_name="torso_link",
            height_scale=height_scan_scale, height_offset=height_scan_offset,
        )
        scanner.bind(mj_model)
    else:
        scanner = None

    # ---- Init state ---------------------------------------------------------
    mujoco.mj_resetData(mj_model, mj_data)
    mj_data.qpos[0:3] = [0.0, 0.0, 0.8]
    mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    init_q_mujoco = np.zeros(num_mujoco_joints, dtype=np.float32)
    for pi, mi in enumerate(policy_to_mujoco):
        if mi >= 0:
            init_q_mujoco[mi] = default_joints_yaml[pi]
    mj_data.qpos[7:7 + num_mujoco_joints] = init_q_mujoco
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    if scanner is not None:
        last_hs_xyz = scanner.scan_xyz(mj_model, mj_data)
        next_height_scan_time = mj_data.time + height_scan_period
    else:
        last_hs_xyz = None
        next_height_scan_time = np.inf

    # ---- Load policy --------------------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    from rsl_rl.modules import ActorCriticEncoder

    checkpoint = torch.load(policy_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))

    # num_critic_obs: critic 每帧比 actor 多 CRITIC_EXTRA_PER_FRAME 维
    #   (root_lin_vel[3] + feet_contact[2])，且按 actor 历史帧数堆叠。
    num_critic_obs = (actor_obs_history * (num_proprio + CRITIC_EXTRA_PER_FRAME)
                      + hs_history * hs_dim)

    policy = ActorCriticEncoder(
        num_actor_obs=num_actor_obs,
        num_critic_obs=num_critic_obs,  # dummy, critic not used
        num_actions=num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        map_scan_dim=(HS_L, HS_W, HS_C),
        map_scan_history_length=hs_history,
        mha_dim=64, num_heads=16,
        cnn_downsample=False,
        attach_global=False,
    ).to(device)

    policy.load_state_dict(state_dict, strict=False)
    policy.eval()
    print("Model loaded.")

    runner = PolicyRunner(policy, device)

    # ---- Warmup -------------------------------------------------------------
    warm_obs = np.zeros(num_actor_obs, dtype=np.float32)
    warm_obs[5] = -1.0  # gravity z ≈ -1
    for _ in range(3):
        runner.act(warm_obs)
    print("Warmup done.")

    # ---- Init last_action / EMA state / Obs history buffer --------------------
    qj = mj_data.qpos[7:]
    last_action = np.zeros(num_actions, dtype=np.float32)
    prev_smoothed_action = np.zeros(num_actions, dtype=np.float32)  # EMA memory
    for pi, mi in enumerate(policy_to_mujoco):
        if mi >= 0 and mi < len(qj):
            last_action[pi] = (qj[mi] - default_joints_yaml[pi]) / action_scale
    prev_smoothed_action[:] = last_action  # seed EMA with initial posture

    # Observation history buffer: circular buffer for stacked frames
    from collections import deque
    obs_history_buffer = deque(maxlen=actor_obs_history)
    hs_history_buffer = deque(maxlen=hs_history)
    _init_single_obs = np.zeros(single_frame_dim, dtype=np.float32)
    _init_single_obs[5] = -1.0  # gravity z seed
    for _ in range(actor_obs_history):
        obs_history_buffer.append(_init_single_obs.copy())
    _init_hs = np.zeros(hs_dim, dtype=np.float32)
    for _ in range(hs_history):
        hs_history_buffer.append(_init_hs.copy())

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

    # ---- Main loop ----------------------------------------------------------
    startup_step = 0
    policy_step = 0
    mj_per_step = sim_dt * control_decimation

    def reset_runtime_state():
        """Restore both MuJoCo state and deploy-side state after a viewer reset."""
        nonlocal last_action, prev_smoothed_action, target_dof_pos
        nonlocal last_hs_xyz, next_height_scan_time, startup_step, policy_step

        mujoco.mj_resetData(mj_model, mj_data)
        mj_data.qpos[0:3] = [0.0, 0.0, 0.8]
        mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        mj_data.qpos[7:7 + num_mujoco_joints] = init_q_mujoco
        mj_data.qvel[:] = 0.0
        mj_data.ctrl[:] = 0.0
        mujoco.mj_forward(mj_model, mj_data)

        last_action = np.zeros(num_actions, dtype=np.float32)
        prev_smoothed_action = last_action.copy()
        target_dof_pos = init_q_mujoco.copy()
        runner.reset()

        obs_history_buffer.clear()
        for _ in range(actor_obs_history):
            obs_history_buffer.append(_init_single_obs.copy())

        hs_history_buffer.clear()
        if scanner is not None:
            # Rescan immediately: otherwise next_height_scan_time still refers
            # to the pre-reset clock and the policy/green markers keep using
            # terrain samples from the robot's previous world position.
            last_hs_xyz = scanner.scan_xyz(mj_model, mj_data)
            for _ in range(hs_history):
                hs_history_buffer.append(last_hs_xyz.copy())
            next_height_scan_time = mj_data.time + height_scan_period
        else:
            last_hs_xyz = None
            for _ in range(hs_history):
                hs_history_buffer.append(_init_hs.copy())
            next_height_scan_time = np.inf

        startup_step = 0
        policy_step = 0
        print("[reset] Robot, height scan, observation history and action state reset.")

    print("\nStarting simulation...\n")

    with mujoco.viewer.launch_passive(mj_model, mj_data,
                                      show_left_ui=True, show_right_ui=True) as viewer:
        # The passive viewer's Reset button calls mj_resetData and moves the
        # simulation clock backwards. Detect that transition so deploy-side
        # buffers and the height scanner are reset together with MuJoCo.
        last_sim_time = float(mj_data.time)
        while viewer.is_running() and mj_data.time < sim_duration:
            step_start = time.time()

            if float(mj_data.time) + 0.5 * sim_dt < last_sim_time:
                reset_runtime_state()
                viewer.user_scn.ngeom = 0
                last_sim_time = float(mj_data.time)

            if policy_step >= policy_delay_steps:
                with cmd_lock:
                    cur_cmd = cmd.copy()
                if startup_step < startup_cmd_zero_steps:
                    cur_cmd[:] = 0.0

                qj = mj_data.qpos[7:]
                dqj = mj_data.qvel[6:]
                omega = mj_data.qvel[3:6]
                grav = get_projected_gravity(mj_model, mj_data)

                qj_pol = np.zeros(num_actions, dtype=np.float32)
                dqj_pol = np.zeros(num_actions, dtype=np.float32)
                for pi, mi in enumerate(policy_to_mujoco):
                    if mi >= 0 and mi < len(qj):
                        qj_pol[pi] = qj[mi]
                        dqj_pol[pi] = dqj[mi]

                if scanner is not None:
                    if mj_data.time >= next_height_scan_time:
                        last_hs_xyz = scanner.scan_xyz(mj_model, mj_data)
                        hs_history_buffer.append(last_hs_xyz.copy())
                        while mj_data.time >= next_height_scan_time:
                            next_height_scan_time += height_scan_period
                    hs_xyz = last_hs_xyz
                else:
                    hs_xyz = None
                    hs_history_buffer.append(np.zeros(hs_dim, dtype=np.float32))
                # Build single-frame observation
                single_obs = build_actor_obs_ame(
                    omega, grav, cur_cmd, qj_pol, dqj_pol, last_action,
                    default_joints_yaml, num_actions,
                    ang_vel_scale, dof_pos_scale, dof_vel_scale, cmd_scale,
                    hs_xyz,
                )
                # Push to history buffer
                obs_history_buffer.append(single_obs)
                # Stack history: proprio frames + height_scan frames
                # Layout: [proprio_t-N+1, ..., proprio_t, hs_t-M+1, ..., hs_t]
                obs_parts = []
                for h_obs in obs_history_buffer:
                    obs_parts.append(h_obs[:num_proprio])
                for h_hs in hs_history_buffer:
                    obs_parts.append(h_hs)
                obs = np.concatenate(obs_parts, dtype=np.float32)
                obs = np.clip(obs, -clip_obs, clip_obs)

                t0 = time.perf_counter()
                raw_action = runner.act(obs)
                inf_ms = (time.perf_counter() - t0) * 1000
                raw_action = np.clip(raw_action, -clip_act, clip_act)

                # --- startup blend (smooth transition from standing to walking) ---
                if startup_blend_steps > 0:
                    blend_alpha = min(float(startup_step + 1) / startup_blend_steps, 1.0)
                else:
                    blend_alpha = 1.0
                blended_action = (1.0 - blend_alpha) * last_action + blend_alpha * raw_action

                # --- EMA action smoothing (low-pass filter to suppress jitter) ---
                smoothed_action = (action_ema_alpha * blended_action
                                   + (1.0 - action_ema_alpha) * prev_smoothed_action)
                prev_smoothed_action = smoothed_action.copy()

                last_action = raw_action.copy()  # feed RAW to observation (matches training)

                loco_target = smoothed_action * action_scale + default_joints_yaml
                for pi, mi in enumerate(policy_to_mujoco):
                    if mi >= 0 and mi < num_mujoco_joints:
                        target_dof_pos[mi] = loco_target[pi]
                startup_step += 1

                if policy_step % 50 == 0:
                    print(f"[step {policy_step}] inf={inf_ms:.1f}ms  "
                          f"cmd=[{cur_cmd[0]:.2f},{cur_cmd[1]:.2f},{cur_cmd[2]:.2f}]  "
                          f"action_range=[{raw_action.min():.2f},{raw_action.max():.2f}]")

            # PD control + physics
            for _ in range(control_decimation):
                q_now = mj_data.qpos[7:7 + num_mujoco_joints]
                dq_now = mj_data.qvel[6:6 + num_mujoco_joints]
                tau = pd_control(target_dof_pos, q_now, kps,
                                 np.zeros(num_mujoco_joints), dq_now, kds)
                tau = np.clip(tau, -torque_limit, torque_limit)
                mj_data.ctrl[:num_mujoco_joints] = tau
                mujoco.mj_step(mj_model, mj_data)

            # --- 地形扫描可视化: 绿色球标记命中点 ---
            if scanner is not None and hasattr(scanner, '_world_hits'):
                pts = scanner._world_hits
                valid = np.isfinite(pts).all(axis=1)
                pts_valid = pts[valid]
                ngeom = min(len(pts_valid), viewer.user_scn.maxgeom)
                viewer.user_scn.ngeom = ngeom
                for i in range(ngeom):
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[i],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=np.array([0.012, 0.0, 0.0], dtype=np.float64),
                        pos=pts_valid[i],
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32),
                    )
            else:
                viewer.user_scn.ngeom = 0

            # Save the time before sync: viewer input is processed by sync(),
            # so a Reset click is observed as a backwards time jump next loop.
            last_sim_time = float(mj_data.time)
            viewer.sync()
            elapsed = time.time() - step_start
            if elapsed < mj_per_step:
                time.sleep(mj_per_step - elapsed)
            policy_step += 1

    running[0] = False
    print("Simulation ended.")

if __name__ == "__main__":
    main()
