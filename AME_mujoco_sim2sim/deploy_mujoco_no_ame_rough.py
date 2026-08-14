#!/usr/bin/env python3
"""Run the current LeggedLab G1 AMP rough-terrain policy in MuJoCo.

The policy input mirrors ``LeggedLab-Isaac-AMP-Rough-G1-Play-v0``:

    base_ang_vel history      5 x 3
    root rotation history     5 x 6
    velocity command history  5 x 3
    joint position history    5 x 29
    joint velocity history    5 x 29
    previous action history   5 x 29
    terrain height map        1 x 11 x 11

The ``--policy`` argument accepts either the TorchScript actor exported by
``scripts/rsl_rl/play.py`` or a training checkpoint such as ``model_20200.pt``.
"""

from __future__ import annotations

import argparse
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent


def _resolve_path(path: str | Path, base_dir: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = base_dir / resolved
    return resolved.resolve()


def _resolve_policy_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Policy file does not exist: {path}")
    return path


def _as_float_array(values: Any, expected_size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (expected_size,):
        raise ValueError(f"{name} must contain {expected_size} values, got shape {array.shape}.")
    return array


def _rotation_about_z(yaw: float) -> np.ndarray:
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def root_local_rot_tan_norm(data: mujoco.MjData, body_id: int) -> np.ndarray:
    """Match ``mdp.root_local_rot_tan_norm`` for the floating pelvis."""
    rotation_wb = data.xmat[body_id].reshape(3, 3)
    yaw = np.arctan2(rotation_wb[1, 0], rotation_wb[0, 0])
    rotation_yaw = _rotation_about_z(float(yaw))
    rotation_local = rotation_yaw.T @ rotation_wb
    return np.concatenate((rotation_local[:, 0], rotation_local[:, 2])).astype(np.float32)


def body_angular_velocity(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
) -> np.ndarray:
    """Return angular velocity in the body frame, matching ``mdp.base_ang_vel``."""
    velocity = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
        velocity,
        1,
    )
    return velocity[:3].astype(np.float32)


class JointMap:
    """Address policy-ordered joints in MuJoCo without relying on XML order."""

    def __init__(self, model: mujoco.MjModel, policy_joint_names: list[str]) -> None:
        self.names = policy_joint_names
        qpos_addresses: list[int] = []
        dof_addresses: list[int] = []
        actuator_ids: list[int] = []

        for name in policy_joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"MuJoCo joint not found: {name}")
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id < 0:
                raise ValueError(f"MuJoCo actuator not found: {name}")
            qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
            dof_addresses.append(int(model.jnt_dofadr[joint_id]))
            actuator_ids.append(actuator_id)

        self.qpos_addresses = np.asarray(qpos_addresses, dtype=np.int32)
        self.dof_addresses = np.asarray(dof_addresses, dtype=np.int32)
        self.actuator_ids = np.asarray(actuator_ids, dtype=np.int32)

    def positions(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray(data.qpos[self.qpos_addresses], dtype=np.float32).copy()

    def velocities(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray(data.qvel[self.dof_addresses], dtype=np.float32).copy()

    def set_positions(self, data: mujoco.MjData, positions: np.ndarray) -> None:
        data.qpos[self.qpos_addresses] = positions

    def set_torques(self, data: mujoco.MjData, torques: np.ndarray) -> None:
        data.ctrl[self.actuator_ids] = torques


class TerrainScanner:
    """MuJoCo ray caster matching the yaw-aligned Isaac Lab ``RayCaster``."""

    def __init__(
        self,
        model: mujoco.MjModel,
        body_name: str,
        resolution: float,
        scan_size: tuple[float, float],
        position_offset: tuple[float, float, float],
        height_offset: float,
    ) -> None:
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if self.body_id < 0:
            raise ValueError(f"Height-scanner body not found: {body_name}")

        self.resolution = resolution
        self.scan_size = scan_size
        self.position_offset = position_offset
        self.height_offset = height_offset
        self.num_x = round(scan_size[0] / resolution) + 1
        self.num_y = round(scan_size[1] / resolution) + 1

        expected_x_size = (self.num_x - 1) * resolution
        expected_y_size = (self.num_y - 1) * resolution
        if not np.isclose(expected_x_size, scan_size[0]) or not np.isclose(expected_y_size, scan_size[1]):
            raise ValueError("height_scan_size must be an integer multiple of height_scan_resolution.")

        offset_x, offset_y, _ = position_offset
        self.local_x = np.linspace(
            offset_x - scan_size[0] / 2.0,
            offset_x + scan_size[0] / 2.0,
            self.num_x,
            dtype=np.float64,
        )
        self.local_y = np.linspace(
            offset_y - scan_size[1] / 2.0,
            offset_y + scan_size[1] / 2.0,
            self.num_y,
            dtype=np.float64,
        )

        robot_body_ids = {self.body_id}
        pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        robot_body_ids.add(pelvis_id)
        changed = True
        while changed:
            changed = False
            for candidate_id in range(model.nbody):
                if int(model.body_parentid[candidate_id]) in robot_body_ids and candidate_id not in robot_body_ids:
                    robot_body_ids.add(candidate_id)
                    changed = True
        self.robot_geom_ids = {
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) in robot_body_ids
        }
        self.world_hits = np.full((self.num_x * self.num_y, 3), np.nan, dtype=np.float64)

    @property
    def observation_size(self) -> int:
        return self.num_x * self.num_y

    def scan(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        body_position = data.xpos[self.body_id].copy()
        body_rotation = data.xmat[self.body_id].reshape(3, 3)
        yaw = np.arctan2(body_rotation[1, 0], body_rotation[0, 0])
        yaw_rotation = _rotation_about_z(float(yaw))

        _, _, offset_z = self.position_offset
        ray_z = float(body_position[2] + offset_z)
        ray_direction = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
        geom_id = np.full(1, -1, dtype=np.int32)
        observation = np.zeros((self.num_y, self.num_x), dtype=np.float32)
        world_hits = np.full((self.num_y, self.num_x, 3), np.nan, dtype=np.float64)

        # Isaac Lab GridPatternCfg(ordering="xy") flattens x fastest and y slowest.
        for y_index, local_y in enumerate(self.local_y):
            for x_index, local_x in enumerate(self.local_x):
                local_xy = np.asarray([local_x, local_y], dtype=np.float64)
                world_xy = body_position[:2] + yaw_rotation[:2, :2] @ local_xy
                ray_origin = np.asarray([world_xy[0], world_xy[1], ray_z], dtype=np.float64)
                hit_position: np.ndarray | None = None

                # Skip visual/collision geometry belonging to the robot itself.
                for _ in range(64):
                    geom_id[0] = -1
                    distance = mujoco.mj_ray(
                        model,
                        data,
                        ray_origin,
                        ray_direction,
                        None,
                        1,
                        -1,
                        geom_id,
                    )
                    if distance < 0.0:
                        break
                    candidate_hit = ray_origin + ray_direction * distance
                    if int(geom_id[0]) not in self.robot_geom_ids:
                        hit_position = candidate_hit
                        break
                    ray_origin = candidate_hit + ray_direction * 1.0e-3

                if hit_position is None:
                    continue

                relative_world = hit_position - body_position
                relative_yaw = yaw_rotation.T @ relative_world
                observation[y_index, x_index] = -relative_yaw[2] - self.height_offset
                world_hits[y_index, x_index] = hit_position

        # Match Isaac Lab's scalar height_scan protection and per-term clipping.
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        observation = np.clip(observation, -1.0, 1.0)
        self.world_hits = world_hits.reshape(-1, 3)
        return observation.reshape(-1)


class PolicyObservation:
    """Build term-major histories exactly like Isaac Lab ObservationManager."""

    TERM_DIMS = {
        "base_ang_vel": 3,
        "root_local_rot_tan_norm": 6,
        "velocity_commands": 3,
        "joint_pos": 29,
        "joint_vel": 29,
        "actions": 29,
    }

    def __init__(self, history_length: int, height_scan_history_length: int) -> None:
        self.history_length = history_length
        self.height_scan_history_length = height_scan_history_length
        self.term_history: dict[str, deque[np.ndarray]] = {}
        self.height_scan_history: deque[np.ndarray] = deque(maxlen=height_scan_history_length)

    def reset(self) -> None:
        self.term_history.clear()
        self.height_scan_history.clear()

    @staticmethod
    def _append_or_fill(history: deque[np.ndarray], value: np.ndarray) -> None:
        if not history:
            for _ in range(history.maxlen or 1):
                history.append(value.copy())
        else:
            history.append(value.copy())

    def build(
        self,
        *,
        base_ang_vel: np.ndarray,
        root_rotation: np.ndarray,
        command: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        last_action: np.ndarray,
        height_scan: np.ndarray,
    ) -> np.ndarray:
        current_terms = {
            "base_ang_vel": base_ang_vel,
            "root_local_rot_tan_norm": root_rotation,
            "velocity_commands": command,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "actions": last_action,
        }

        observation_parts: list[np.ndarray] = []
        for term_name, expected_dim in self.TERM_DIMS.items():
            value = _as_float_array(current_terms[term_name], expected_dim, term_name)
            history = self.term_history.setdefault(term_name, deque(maxlen=self.history_length))
            self._append_or_fill(history, value)
            observation_parts.append(np.concatenate(tuple(history)))

        self._append_or_fill(self.height_scan_history, np.asarray(height_scan, dtype=np.float32))
        observation_parts.append(np.concatenate(tuple(self.height_scan_history)))
        observation = np.concatenate(observation_parts).astype(np.float32, copy=False)
        if not np.isfinite(observation).all():
            raise FloatingPointError("Non-finite value found in the MuJoCo policy observation.")
        return observation


class JitPolicy:
    def __init__(self, policy_path: Path, device: torch.device, expected_obs_dim: int, num_actions: int) -> None:
        self.device = device
        self.expected_obs_dim = expected_obs_dim
        self.num_actions = num_actions
        try:
            self.module = torch.jit.load(str(policy_path), map_location=device)
        except Exception as error:
            raise RuntimeError(
                f"Unable to load {policy_path} as a TorchScript policy. "
                "Use the exported/policy.pt generated by scripts/rsl_rl/play.py."
            ) from error
        self.module.eval()

    def act(self, observation: np.ndarray) -> np.ndarray:
        if observation.shape != (self.expected_obs_dim,):
            raise ValueError(
                f"Policy observation has shape {observation.shape}; expected ({self.expected_obs_dim},)."
            )
        observation_tensor = torch.from_numpy(observation).to(self.device).unsqueeze(0)
        with torch.inference_mode():
            output = self.module(observation_tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Unexpected TorchScript policy output type: {type(output).__name__}")
        action = output.detach().to("cpu").numpy().reshape(-1).astype(np.float32)
        if action.shape != (self.num_actions,):
            raise ValueError(f"Policy returned {action.shape}; expected ({self.num_actions},).")
        if not np.isfinite(action).all():
            raise FloatingPointError("TorchScript policy returned a non-finite action.")
        return action


class TrainingCheckpointPolicy:
    """Load a plain MLP training checkpoint and expose the JitPolicy inference API."""

    def __init__(
        self,
        checkpoint_path: Path,
        device: torch.device,
        expected_obs_dim: int,
        num_actions: int,
    ) -> None:
        self.device = device
        self.expected_obs_dim = expected_obs_dim
        self.num_actions = num_actions

        checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"{checkpoint_path} is not a supported RSL-RL training checkpoint.")

        critic_obs_dim = self._infer_critic_obs_dim(checkpoint, expected_obs_dim)

        try:
            from rsl_rl.modules import ActorCritic
        except ImportError as error:
            raise RuntimeError(
                "Loading a training checkpoint requires rsl_rl.modules.ActorCritic in the active Python "
                "environment. Use an exported TorchScript policy.pt or activate the Isaac Lab environment."
            ) from error

        self.module = ActorCritic(
            num_actor_obs=expected_obs_dim,
            num_critic_obs=critic_obs_dim,
            num_actions=num_actions,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        ).to(device)

        state_dict = self._checkpoint_to_actor_critic_state_dict(checkpoint)
        loaded = self.module.load_state_dict(state_dict, strict=False)
        if loaded is False:
            raise RuntimeError(f"Unable to load actor weights from training checkpoint: {checkpoint_path}")
        self.module.eval()

    @staticmethod
    def _infer_critic_obs_dim(checkpoint: dict[str, Any], fallback: int) -> int:
        critic_state_dict = checkpoint.get("critic_state_dict")
        if isinstance(critic_state_dict, dict):
            for key in ("critic.0.weight", "mlp.0.weight"):
                weight = critic_state_dict.get(key)
                if isinstance(weight, torch.Tensor) and weight.ndim == 2:
                    return int(weight.shape[1])
        return fallback

    @staticmethod
    def _checkpoint_to_actor_critic_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
        actor_state_dict = checkpoint.get("actor_state_dict")
        if isinstance(actor_state_dict, dict):
            mapped_state_dict: dict[str, torch.Tensor] = {}
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

        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
        if isinstance(state_dict, dict):
            return {key: value for key, value in state_dict.items() if isinstance(value, torch.Tensor)}

        raise RuntimeError(
            "Unsupported checkpoint format. Expected actor_state_dict, model_state_dict, or state_dict."
        )

    def act(self, observation: np.ndarray) -> np.ndarray:
        if observation.shape != (self.expected_obs_dim,):
            raise ValueError(
                f"Policy observation has shape {observation.shape}; expected ({self.expected_obs_dim},)."
            )
        observation_tensor = torch.from_numpy(observation).to(self.device).unsqueeze(0)
        with torch.inference_mode():
            output = self.module.act_inference(observation_tensor)
        action = output.detach().to("cpu").numpy().reshape(-1).astype(np.float32)
        if action.shape != (self.num_actions,):
            raise ValueError(f"Policy returned {action.shape}; expected ({self.num_actions},).")
        if not np.isfinite(action).all():
            raise FloatingPointError("Training checkpoint policy returned a non-finite action.")
        return action


def load_policy(
    policy_path: Path,
    device: torch.device,
    expected_obs_dim: int,
    num_actions: int,
) -> JitPolicy | TrainingCheckpointPolicy:
    if policy_path.name.startswith("model_"):
        return TrainingCheckpointPolicy(
            policy_path,
            device,
            expected_obs_dim,
            num_actions,
        )
    try:
        return JitPolicy(policy_path, device, expected_obs_dim, num_actions)
    except RuntimeError:
        return TrainingCheckpointPolicy(
            policy_path,
            device,
            expected_obs_dim,
            num_actions,
        )


def pd_control(
    target_position: np.ndarray,
    joint_position: np.ndarray,
    stiffness: np.ndarray,
    joint_velocity: np.ndarray,
    damping: np.ndarray,
) -> np.ndarray:
    return (target_position - joint_position) * stiffness - joint_velocity * damping


def _deadzone(value: float, threshold: float) -> float:
    return 0.0 if abs(value) < threshold else value


def _run_gamepad(
    command: np.ndarray,
    command_lock: threading.Lock,
    running: list[bool],
    deadzone: float,
    gains: np.ndarray,
) -> None:
    try:
        import pygame
    except ImportError as error:
        raise RuntimeError("Gamepad control requires pygame: pip install pygame") from error

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() < 1:
        raise RuntimeError("--gamepad was requested, but no gamepad was detected.")
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"[INFO] Gamepad: {joystick.get_name()}")

    while running[0]:
        pygame.event.pump()
        forward = _deadzone(-float(joystick.get_axis(1)), deadzone)
        lateral = _deadzone(-float(joystick.get_axis(0)), deadzone)
        yaw_rate = _deadzone(-float(joystick.get_axis(3)), deadzone)
        with command_lock:
            command[:] = np.asarray([forward, lateral, yaw_rate], dtype=np.float32) * gains
        time.sleep(0.01)


def _draw_scan_hits(viewer: Any, scanner: TerrainScanner) -> None:
    valid_hits = scanner.world_hits[np.isfinite(scanner.world_hits).all(axis=1)]
    num_geoms = min(len(valid_hits), viewer.user_scn.maxgeom)
    viewer.user_scn.ngeom = num_geoms
    for index in range(num_geoms):
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[index],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.asarray([0.012, 0.0, 0.0], dtype=np.float64),
            pos=valid_hits[index],
            mat=np.eye(3, dtype=np.float64).reshape(-1),
            rgba=np.asarray([0.1, 0.9, 0.2, 0.8], dtype=np.float32),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        "--policy_path",
        dest="policy",
        required=True,
        help="Exported TorchScript policy.pt or an RSL-RL training checkpoint such as model_*.pt.",
    )
    parser.add_argument("--config", default="g1_29dof.yaml", help="Deployment YAML path.")
    parser.add_argument("--device", default=None, help="Torch device. Defaults to cuda:0 when available.")
    parser.add_argument("--duration", type=float, default=None, help="Override simulation duration in seconds.")
    parser.add_argument(
        "--command",
        type=float,
        nargs=3,
        metavar=("VX", "VY", "WZ"),
        default=None,
        help="Override the fixed velocity command.",
    )
    parser.add_argument("--gamepad", action="store_true", help="Control the velocity command with a gamepad.")
    parser.add_argument("--headless", action="store_true", help="Run without the MuJoCo viewer.")
    parser.add_argument("--no-realtime", action="store_true", help="Do not pace the simulation to wall time.")
    parser.add_argument("--hide-scan", action="store_true", help="Hide terrain-ray hit markers.")
    parser.add_argument(
        "--zero-height-scan",
        action="store_true",
        help="Feed a zero map while preserving the required policy dimension.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Load assets and run one policy inference, then exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve_path(args.config, SCRIPT_DIR)
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    config_dir = config_path.parent

    xml_path = _resolve_path(config["xml_path"], config_dir)
    requested_policy_path = _resolve_path(args.policy, Path.cwd())
    policy_path = _resolve_policy_file(requested_policy_path)

    num_actions = int(config["num_actions"])
    history_length = int(config["policy_history_length"])
    height_history_length = int(config["height_scan_history_length"])
    action_scale = float(config["action_scale"])
    action_clip = float(config["clip_actions"])
    sim_dt = float(config["simulation_dt"])
    control_decimation = int(config["control_decimation"])
    control_dt = sim_dt * control_decimation
    sim_duration = float(args.duration if args.duration is not None else config["simulation_duration"])

    if num_actions != 29:
        raise ValueError(f"This G1 deployment expects 29 actions, got {num_actions}.")
    if history_length != 5:
        raise ValueError(f"The current policy configuration requires history length 5, got {history_length}.")
    if height_history_length != 1:
        raise ValueError(
            f"The current terrain-map configuration requires history length 1, got {height_history_length}."
        )

    joint_names = list(config["policy_joint_order"])
    if len(joint_names) != num_actions or len(set(joint_names)) != num_actions:
        raise ValueError("policy_joint_order must contain 29 unique joint names.")
    default_joint_pos = _as_float_array(config["default_joint_angles"], num_actions, "default_joint_angles")
    stiffness = _as_float_array(config["stiffness"], num_actions, "stiffness")
    damping = _as_float_array(config["damping"], num_actions, "damping")
    effort_limits = _as_float_array(config["effort_limits"], num_actions, "effort_limits")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    model.opt.timestep = sim_dt
    joint_map = JointMap(model, joint_names)
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id < 0:
        raise ValueError("MuJoCo body 'pelvis' was not found.")

    scan_resolution = float(config["height_scan_resolution"])
    scan_size = tuple(float(value) for value in config["height_scan_size"])
    scan_position_offset = tuple(float(value) for value in config["height_scan_position_offset"])
    scanner = TerrainScanner(
        model=model,
        body_name=str(config["height_scan_body"]),
        resolution=scan_resolution,
        scan_size=scan_size,
        position_offset=scan_position_offset,
        height_offset=float(config["height_scan_height_offset"]),
    )
    if (scanner.num_x, scanner.num_y) != (11, 11):
        raise ValueError(
            "The current rough MLP actor requires an 11x11 height map, "
            f"but the deployment config produces {scanner.num_x}x{scanner.num_y}."
        )

    expected_obs_dim = (
        history_length * sum(PolicyObservation.TERM_DIMS.values())
        + height_history_length * scanner.observation_size
    )
    if expected_obs_dim != 616:
        raise ValueError(f"The current G1 AMP rough MLP actor requires 616 observations, got {expected_obs_dim}.")

    device_name = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    policy = load_policy(
        policy_path,
        device,
        expected_obs_dim,
        num_actions,
    )
    observation_builder = PolicyObservation(history_length, height_history_length)

    command = np.asarray(args.command if args.command is not None else config["command"], dtype=np.float32)
    gamepad_gains = _as_float_array(config["gamepad_command_gains"], 3, "gamepad_command_gains")
    command_lock = threading.Lock()
    running = [True]
    gamepad_thread: threading.Thread | None = None

    root_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    if root_joint_id < 0:
        raise ValueError("MuJoCo free joint 'floating_base_joint' was not found.")
    root_qpos_address = int(model.jnt_qposadr[root_joint_id])
    base_position = _as_float_array(config["initial_base_position"], 3, "initial_base_position")

    def reset_simulation() -> tuple[np.ndarray, np.ndarray]:
        mujoco.mj_resetData(model, data)
        data.qpos[root_qpos_address : root_qpos_address + 3] = base_position
        data.qpos[root_qpos_address + 3 : root_qpos_address + 7] = np.asarray(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        joint_map.set_positions(data, default_joint_pos)
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)
        observation_builder.reset()
        initial_action = np.zeros(num_actions, dtype=np.float32)
        initial_target = default_joint_pos.copy()
        return initial_action, initial_target

    last_action, target_joint_pos = reset_simulation()
    height_scan_frequency = float(config["height_scan_frequency"])
    if height_scan_frequency <= 0.0:
        raise ValueError("height_scan_frequency must be positive.")
    height_scan_period = 1.0 / height_scan_frequency
    next_height_scan_time = float(data.time)
    height_scan = np.zeros(scanner.observation_size, dtype=np.float32)

    action_ema_alpha = float(config.get("action_ema_alpha", 1.0))
    if not 0.0 < action_ema_alpha <= 1.0:
        raise ValueError("action_ema_alpha must be in (0, 1].")
    smoothed_action = last_action.copy()

    print("=" * 72)
    print("LeggedLab G1 AMP Rough -> MuJoCo sim2sim")
    print(f"  XML:             {xml_path}")
    print(f"  Policy:          {policy_path}")
    print(f"  Torch device:    {device}")
    print(f"  Control rate:    {1.0 / control_dt:.1f} Hz ({control_decimation} x {sim_dt:g} s)")
    print(f"  Policy input:    {expected_obs_dim} = 495 proprio history + 121 terrain heights")
    print(f"  Terrain map:     {scanner.num_x} x {scanner.num_y} z @ {scan_resolution:.3f} m")
    print(f"  Command:         [{command[0]:.2f}, {command[1]:.2f}, {command[2]:.2f}]")
    print("=" * 72)

    def build_observation() -> np.ndarray:
        nonlocal height_scan, next_height_scan_time
        if args.zero_height_scan:
            height_scan.fill(0.0)
        elif data.time + 1.0e-9 >= next_height_scan_time:
            height_scan = scanner.scan(model, data)
            while next_height_scan_time <= data.time + 1.0e-9:
                next_height_scan_time += height_scan_period

        with command_lock:
            current_command = command.copy()
        return observation_builder.build(
            base_ang_vel=body_angular_velocity(model, data, pelvis_id),
            root_rotation=root_local_rot_tan_norm(data, pelvis_id),
            command=current_command,
            joint_pos=joint_map.positions(data),
            joint_vel=joint_map.velocities(data),
            last_action=last_action,
            height_scan=height_scan,
        )

    first_observation = build_observation()
    first_action = policy.act(first_observation)
    print(
        f"[INFO] Policy validation passed: obs={first_observation.shape}, "
        f"action={first_action.shape}, action_range=[{first_action.min():.3f}, {first_action.max():.3f}]"
    )
    if args.dry_run:
        running[0] = False
        return

    if args.gamepad:
        gamepad_thread = threading.Thread(
            target=_run_gamepad,
            args=(command, command_lock, running, float(config["gamepad_deadzone"]), gamepad_gains),
            daemon=True,
        )
        gamepad_thread.start()

    def control_step() -> float:
        nonlocal last_action, smoothed_action, target_joint_pos
        step_start = time.perf_counter()
        observation = build_observation()
        action = np.clip(policy.act(observation), -action_clip, action_clip)
        smoothed_action = action_ema_alpha * action + (1.0 - action_ema_alpha) * smoothed_action
        last_action = action.copy()
        target_joint_pos = default_joint_pos + action_scale * smoothed_action
        return (time.perf_counter() - step_start) * 1000.0

    def physics_step() -> None:
        joint_position = joint_map.positions(data)
        joint_velocity = joint_map.velocities(data)
        torque = pd_control(target_joint_pos, joint_position, stiffness, joint_velocity, damping)
        joint_map.set_torques(data, np.clip(torque, -effort_limits, effort_limits))
        mujoco.mj_step(model, data)

    policy_step = 0
    previous_sim_time = float(data.time)

    def run_iteration(viewer: Any | None) -> None:
        nonlocal policy_step, previous_sim_time, last_action, target_joint_pos
        nonlocal smoothed_action, next_height_scan_time

        if data.time + 0.5 * sim_dt < previous_sim_time:
            last_action, target_joint_pos = reset_simulation()
            smoothed_action = last_action.copy()
            next_height_scan_time = float(data.time)
            policy_step = 0
            if viewer is not None:
                viewer.user_scn.ngeom = 0
            print("[INFO] MuJoCo and policy histories reset.")

        inference_ms = control_step()
        for _ in range(control_decimation):
            physics_step()

        if viewer is not None:
            if not args.hide_scan and not args.zero_height_scan:
                _draw_scan_hits(viewer, scanner)
            else:
                viewer.user_scn.ngeom = 0
            previous_sim_time = float(data.time)
            viewer.sync()
        else:
            previous_sim_time = float(data.time)

        if policy_step % int(config["print_interval"]) == 0:
            with command_lock:
                current_command = command.copy()
            print(
                f"[step {policy_step:06d}] t={data.time:7.2f}s inference={inference_ms:6.2f}ms "
                f"cmd=[{current_command[0]:+.2f},{current_command[1]:+.2f},{current_command[2]:+.2f}] "
                f"action=[{last_action.min():+.2f},{last_action.max():+.2f}]"
            )
        policy_step += 1

    try:
        if args.headless:
            while data.time < sim_duration:
                iteration_start = time.perf_counter()
                run_iteration(None)
                if not args.no_realtime:
                    elapsed = time.perf_counter() - iteration_start
                    if elapsed < control_dt:
                        time.sleep(control_dt - elapsed)
        else:
            with mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as viewer:
                while viewer.is_running() and data.time < sim_duration:
                    iteration_start = time.perf_counter()
                    run_iteration(viewer)
                    if not args.no_realtime:
                        elapsed = time.perf_counter() - iteration_start
                        if elapsed < control_dt:
                            time.sleep(control_dt - elapsed)
    finally:
        running[0] = False
        if gamepad_thread is not None:
            gamepad_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
