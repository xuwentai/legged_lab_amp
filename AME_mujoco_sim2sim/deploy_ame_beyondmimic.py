#!/usr/bin/env python3
"""
MuJoCo sim2sim for LeggedLab-trained G1 policies (AME Encoder + BeyondMimic PD).

Loads an ActorCriticEncoder checkpoint and deploys in MuJoCo with BeyondMimic
frequency-based PD gains (armature * omega_n^2).  Height scan outputs xyz.

Usage:
  cd AME_mujoco_sim2sim
  python deploy_ame_beyondmimic.py --policy_path ../logs/g1_rough/.../model_XX.pt --gamepad
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
# BeyondMimic motor parameters
# =============================================================================
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10.0 * 2.0 * math.pi
DAMPING_RATIO = 2.0

STIFFNESS_5020    = ARMATURE_5020 * NATURAL_FREQ ** 2       # ≈ 14.25
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ ** 2     # ≈ 40.18
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ ** 2     # ≈ 99.10
STIFFNESS_4010    = ARMATURE_4010 * NATURAL_FREQ ** 2        # ≈ 16.78

DAMPING_5020    = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ    # ≈ 0.907
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ  # ≈ 2.558
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ  # ≈ 6.309
DAMPING_4010    = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ     # ≈ 1.068

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

# BeyondMimic default joint angles (deeper crouch, different arm pose)
_DEFAULT_JOINT_PATTERNS = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
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
# PD gains — BeyondMimic (frequency-based)
# =============================================================================

_G1_PD_GROUPS = {
    "legs": {
        "stiffness": {
            ".*_hip_pitch_joint": STIFFNESS_7520_14,
            ".*_hip_roll_joint":  STIFFNESS_7520_22,
            ".*_hip_yaw_joint":   STIFFNESS_7520_14,
            ".*_knee_joint":      STIFFNESS_7520_22,
        },
        "damping": {
            ".*_hip_pitch_joint": DAMPING_7520_14,
            ".*_hip_roll_joint":  DAMPING_7520_22,
            ".*_hip_yaw_joint":   DAMPING_7520_14,
            ".*_knee_joint":      DAMPING_7520_22,
        },
        "effort": {
            ".*_hip_yaw_joint": 88, ".*_hip_roll_joint": 139,
            ".*_hip_pitch_joint": 88, ".*_knee_joint": 139,
        },
    },
    "feet": {
        "stiffness": {
            ".*_ankle_pitch_joint": 2.0 * STIFFNESS_5020,
            ".*_ankle_roll_joint":  2.0 * STIFFNESS_5020,
        },
        "damping": {
            ".*_ankle_pitch_joint": 2.0 * DAMPING_5020,
            ".*_ankle_roll_joint":  2.0 * DAMPING_5020,
        },
        "effort": {".*_ankle_pitch_joint": 50, ".*_ankle_roll_joint": 50},
    },
    "waist": {
        "stiffness": {
            "waist_roll_joint":  2.0 * STIFFNESS_5020,
            "waist_pitch_joint": 2.0 * STIFFNESS_5020,
        },
        "damping": {
            "waist_roll_joint":  2.0 * DAMPING_5020,
            "waist_pitch_joint": 2.0 * DAMPING_5020,
        },
        "effort": {"waist_roll_joint": 50, "waist_pitch_joint": 50},
    },
    "waist_yaw": {
        "stiffness": {"waist_yaw_joint": STIFFNESS_7520_14},
        "damping":   {"waist_yaw_joint": DAMPING_7520_14},
        "effort":    {"waist_yaw_joint": 88},
    },
    "arms": {
        "stiffness": {
            ".*_shoulder_pitch_joint": STIFFNESS_5020,
            ".*_shoulder_roll_joint":  STIFFNESS_5020,
            ".*_shoulder_yaw_joint":   STIFFNESS_5020,
            ".*_elbow_joint":          STIFFNESS_5020,
            ".*_wrist_roll_joint":     STIFFNESS_5020,
            ".*_wrist_pitch_joint":    STIFFNESS_4010,
            ".*_wrist_yaw_joint":      STIFFNESS_4010,
        },
        "damping": {
            ".*_shoulder_pitch_joint": DAMPING_5020,
            ".*_shoulder_roll_joint":  DAMPING_5020,
            ".*_shoulder_yaw_joint":   DAMPING_5020,
            ".*_elbow_joint":          DAMPING_5020,
            ".*_wrist_roll_joint":     DAMPING_5020,
            ".*_wrist_pitch_joint":    DAMPING_4010,
            ".*_wrist_yaw_joint":      DAMPING_4010,
        },
        "effort": {
            ".*_shoulder_pitch_joint": 25, ".*_shoulder_roll_joint": 25,
            ".*_shoulder_yaw_joint": 25, ".*_elbow_joint": 25,
            ".*_wrist_roll_joint": 25, ".*_wrist_pitch_joint": 5,
            ".*_wrist_yaw_joint": 5,
        },
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
# Height scanner (xyz output)
# =============================================================================

class HeightScannerXYZ:
    def __init__(self, resolution=0.04, scan_size=(0.4, 0.4),
                 offset=(0.20, 0.0, 20.0), body_name="torso_link",
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
        body_pos = mj_data.xpos[self._body_id].copy()
        body_quat = mj_data.xquat[self._body_id].copy()
        R_yaw = self._yaw_rot_matrix(body_quat)
        ray_dir = np.array([[0.0], [0.0], [-1.0]], dtype=np.float64)
        geom_id_buf = np.zeros((1, 1), dtype=np.int32)
        body_z = float(body_pos[2])
        ray_z = body_z + float(self.offset[2])
        xyz_flat = np.zeros(self.nx * self.ny * 3, dtype=np.float32)

        for ix, lx in enumerate(self.grid_x):
            for iy, ly in enumerate(self.grid_y):
                idx_flat = (iy * self.nx + ix) * 3
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

        #         rel_world = hit_world - np.append(body_pos[:2], 0.0)
        #         rel_body = R_yaw.T @ rel_world
        #         xyz_flat[idx_flat + 0] = float(rel_body[0]) * self.height_scale
        #         xyz_flat[idx_flat + 1] = float(rel_body[1]) * self.height_scale
        #         xyz_flat[idx_flat + 2] = (body_z - float(hit_world[2]) - self.height_offset) * self.height_scale
        # return xyz_flat
                rel_world = hit_world - np.append(body_pos[:2], 0.0)
                rel_body = R_yaw.T @ rel_world
                xyz_flat[idx_flat + 0] = float(rel_body[0]) * self.height_scale
                xyz_flat[idx_flat + 1] = float(rel_body[1]) * self.height_scale
                # z: same formula as original (body_z - hit_z - offset) * scale
                z_raw = (body_z - float(hit_world[2]) - self.height_offset) * self.height_scale
                xyz_flat[idx_flat + 2] = np.clip(z_raw, -0.5, 0.7)
        # self._world_hits = world_hits  # for visualization
        return xyz_flat

# =============================================================================
# Observation builder
# =============================================================================

def build_actor_obs_ame(ang_vel, projected_gravity, command, joint_pos, joint_vel,
                         last_action, default_joint_pos, num_actions,
                         ang_vel_scale, dof_pos_scale, dof_vel_scale, cmd_scale,
                         height_scan_xyz):
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
# Policy wrapper
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
    parser = argparse.ArgumentParser(description="MuJoCo sim2sim AME Encoder + BeyondMimic PD")
    parser.add_argument("--policy_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="g1_29dof_beyondmimic.yaml")
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
    use_training_pd = bool(cfg.get("use_training_pd", True))
    pd_scale = float(cfg.get("pd_scale", 1.0))
    cmd_init = np.array(cfg["cmd_init"], dtype=np.float32)
    use_gamepad = bool(cfg.get("use_gamepad", False)) or args.gamepad
    gamepad_deadzone = float(cfg.get("gamepad_deadzone", 0.15))
    gamepad_cmd_gain = np.array(cfg.get("gamepad_cmd_gain", [1.0, 0.6, 1.0]), dtype=np.float32)
    startup_blend_steps = int(cfg.get("startup_blend_steps", 30))
    startup_cmd_zero_steps = int(cfg.get("startup_cmd_zero_steps", 20))
    policy_delay_steps = int(cfg.get("policy_delay_steps", 0))

    # ---- AME encoder dimensions ----
    HS_L, HS_W, HS_C = 11, 11, 3
    hs_dim = HS_L * HS_W * HS_C
    num_proprio = 3 + 3 + 3 + num_actions * 3
    num_actor_obs = num_proprio + hs_dim
    height_scan_scale = 1.0
    height_scan_offset = 0.5

    print("=" * 70)
    print("LeggedLab MuJoCo Sim2Sim — AME Encoder + BeyondMimic PD")
    print("=" * 70)
    print(f"  XML:          {xml_path}")
    print(f"  Policy:       {policy_path}")
    print(f"  Actions:      {num_actions}")
    print(f"  Proprio:      {num_proprio}")
    print(f"  Height scan:  {hs_dim} dims ({HS_L}×{HS_W} grid, coord_dim={HS_C})")
    print(f"  Actor obs:    {num_actor_obs}")
    print(f"  Action scale: {action_scale}")
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

    # ---- PD gains (BeyondMimic) --------------------------------------------
    if use_training_pd:
        kps, kds = get_training_pd_gains(mujoco_joint_names)
        torque_limit = get_training_effort_limits(mujoco_joint_names)
    else:
        kps = np.array(cfg["stiffness"], dtype=np.float32)
        kds = np.array(cfg["damping"], dtype=np.float32)
        torque_limit = np.array(cfg["torque_limit"], dtype=np.float32)
    if pd_scale != 1.0:
        kps *= pd_scale; kds *= pd_scale

    # BeyondMimic defaults
    default_joints_yaml = np.array(cfg.get("default_joint_angles",
                                   DEFAULT_JOINT_ANGLES_POLICY.tolist()), dtype=np.float32)

    # ---- Height scanner (xyz) -----------------------------------------------
    if not args.no_height_scan:
        scanner = HeightScannerXYZ(
            resolution=0.04, scan_size=(0.4, 0.4),
            offset=(0.20, 0.0, 20.0), body_name="torso_link",
            height_scale=height_scan_scale, height_offset=height_scan_offset,
        )
        scanner.bind(mj_model)
    else:
        scanner = None

    # ---- Init state ---------------------------------------------------------
    mujoco.mj_resetData(mj_model, mj_data)
    mj_data.qpos[0:3] = [0.0, 0.0, 0.76]  # BeyondMimic lower init height
    mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
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

    from rsl_rl.modules import ActorCriticEncoder

    checkpoint = torch.load(policy_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))

    policy = ActorCriticEncoder(
        num_actor_obs=num_actor_obs,
        num_critic_obs=num_actor_obs + 5,
        num_actions=num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        map_scan_dim=(HS_L, HS_W, HS_C),
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
    warm_obs[5] = -1.0
    for _ in range(3):
        runner.act(warm_obs)
    print("Warmup done.")

    # ---- Init last_action ---------------------------------------------------
    qj = mj_data.qpos[7:]
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

    # ---- Main loop ----------------------------------------------------------
    startup_step = 0
    policy_step = 0
    mj_per_step = sim_dt * control_decimation

    print("\nStarting simulation...\n")

    with mujoco.viewer.launch_passive(mj_model, mj_data,
                                      show_left_ui=True, show_right_ui=True) as viewer:
        while viewer.is_running() and mj_data.time < sim_duration:
            step_start = time.time()

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

                hs_xyz = scanner.scan_xyz(mj_model, mj_data) if scanner is not None else None

                obs = build_actor_obs_ame(
                    omega, grav, cur_cmd, qj_pol, dqj_pol, last_action,
                    default_joints_yaml, num_actions,
                    ang_vel_scale, dof_pos_scale, dof_vel_scale, cmd_scale,
                    hs_xyz,
                )
                obs = np.clip(obs, -clip_obs, clip_obs)

                t0 = time.perf_counter()
                raw_action = runner.act(obs)
                inf_ms = (time.perf_counter() - t0) * 1000
                raw_action = np.clip(raw_action, -clip_act, clip_act)

                if startup_blend_steps > 0:
                    alpha = min(float(startup_step + 1) / startup_blend_steps, 1.0)
                else:
                    alpha = 1.0
                action = (1.0 - alpha) * last_action + alpha * raw_action
                last_action = raw_action.copy()

                loco_target = action * action_scale + default_joints_yaml
                for pi, mi in enumerate(policy_to_mujoco):
                    if mi >= 0 and mi < num_mujoco_joints:
                        target_dof_pos[mi] = loco_target[pi]
                startup_step += 1

                if policy_step % 50 == 0:
                    print(f"[step {policy_step}] inf={inf_ms:.1f}ms  "
                          f"cmd=[{cur_cmd[0]:.2f},{cur_cmd[1]:.2f},{cur_cmd[2]:.2f}]  "
                          f"action_range=[{raw_action.min():.2f},{raw_action.max():.2f}]")

            for _ in range(control_decimation):
                q_now = mj_data.qpos[7:7 + num_mujoco_joints]
                dq_now = mj_data.qvel[6:6 + num_mujoco_joints]
                tau = pd_control(target_dof_pos, q_now, kps,
                                 np.zeros(num_mujoco_joints), dq_now, kds)
                tau = np.clip(tau, -torque_limit, torque_limit)
                mj_data.ctrl[:num_mujoco_joints] = tau
                mujoco.mj_step(mj_model, mj_data)

            viewer.sync()
            elapsed = time.time() - step_start
            if elapsed < mj_per_step:
                time.sleep(mj_per_step - elapsed)
            policy_step += 1

    running[0] = False
    print("Simulation ended.")

if __name__ == "__main__":
    main()
