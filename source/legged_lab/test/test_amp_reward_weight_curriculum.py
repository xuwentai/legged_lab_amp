from __future__ import annotations

import importlib.util
import unittest
from dataclasses import dataclass
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).parents[1]
    / "legged_lab"
    / "tasks"
    / "locomotion"
    / "amp"
    / "mdp"
    / "curriculums.py"
)
SPEC = importlib.util.spec_from_file_location("amp_curriculums", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
AMP_CURRICULUMS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AMP_CURRICULUMS)
reward_weight_from_tracking = AMP_CURRICULUMS.reward_weight_from_tracking


@dataclass
class _RewardCfg:
    weight: float


class _RewardManager:
    def __init__(self, intrusion_weight: float, tracking_score: float, num_envs: int, episode_length_s: float):
        self._cfgs = {
            "foot_stair_intrusion": _RewardCfg(intrusion_weight),
            "track_lin_vel_xy_exp": _RewardCfg(1.0),
        }
        self._episode_sums = {
            "track_lin_vel_xy_exp": torch.full((num_envs,), tracking_score * episode_length_s),
        }

    def get_term_cfg(self, term_name: str) -> _RewardCfg:
        return self._cfgs[term_name]

    def set_term_cfg(self, term_name: str, cfg: _RewardCfg) -> None:
        self._cfgs[term_name] = cfg


class _Env:
    def __init__(self, intrusion_weight: float, tracking_score: float):
        self.num_envs = 4
        self.device = "cpu"
        self.max_episode_length_s = 20.0
        self.reward_manager = _RewardManager(
            intrusion_weight,
            tracking_score,
            self.num_envs,
            self.max_episode_length_s,
        )


def _update(env: _Env) -> dict[str, torch.Tensor]:
    return reward_weight_from_tracking(
        env,
        slice(None),
        reward_term_name="foot_stair_intrusion",
        tracking_reward_term_name="track_lin_vel_xy_exp",
        initial_weight=-0.1,
        final_weight=-1.0,
        tracking_thresholds=(0.7, 0.9),
        step_size=0.03,
    )


class RewardWeightFromTrackingTest(unittest.TestCase):
    def test_increases_penalty_after_strong_tracking(self) -> None:
        env = _Env(intrusion_weight=-0.1, tracking_score=0.95)

        state = _update(env)

        self.assertAlmostEqual(env.reward_manager.get_term_cfg("foot_stair_intrusion").weight, -0.127)
        self.assertAlmostEqual(state["weight"].item(), -0.127)
        self.assertAlmostEqual(state["tracking_score"].item(), 0.95)

    def test_relaxes_penalty_after_poor_tracking(self) -> None:
        env = _Env(intrusion_weight=-0.5, tracking_score=0.6)

        _update(env)

        self.assertAlmostEqual(env.reward_manager.get_term_cfg("foot_stair_intrusion").weight, -0.488)

    def test_keeps_penalty_inside_threshold_band(self) -> None:
        env = _Env(intrusion_weight=-0.4, tracking_score=0.8)

        _update(env)

        self.assertAlmostEqual(env.reward_manager.get_term_cfg("foot_stair_intrusion").weight, -0.4)


if __name__ == "__main__":
    unittest.main()
