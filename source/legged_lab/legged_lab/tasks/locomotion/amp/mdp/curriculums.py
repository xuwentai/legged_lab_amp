"""Curriculum terms specific to AMP locomotion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reward_weight_from_tracking(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | slice,
    reward_term_name: str,
    tracking_reward_term_name: str,
    initial_weight: float,
    final_weight: float,
    tracking_thresholds: tuple[float, float] = (0.7, 0.9),
    step_size: float = 0.03,
) -> dict[str, torch.Tensor]:
    """Adapt a global reward weight using completed-episode tracking performance.

    Poor tracking moves the weight toward ``initial_weight``. Strong tracking
    moves it toward ``final_weight``. Values in the threshold band leave the
    current weight unchanged.
    """
    if not 0.0 < step_size <= 1.0:
        raise ValueError(f"step_size must be in (0, 1], got {step_size}.")
    lower_threshold, upper_threshold = tracking_thresholds
    if lower_threshold > upper_threshold:
        raise ValueError(f"tracking_thresholds must be ordered, got {tracking_thresholds}.")

    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).reshape(-1)

    reward_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    tracking_cfg = env.reward_manager.get_term_cfg(tracking_reward_term_name)
    if tracking_cfg.weight == 0.0:
        raise ValueError(f"Tracking reward '{tracking_reward_term_name}' must have a non-zero weight.")

    episode_tracking = env.reward_manager._episode_sums[tracking_reward_term_name][env_ids]
    tracking_score = torch.mean(episode_tracking) / env.max_episode_length_s / tracking_cfg.weight

    current_weight = float(reward_cfg.weight)
    score = float(tracking_score.item())
    if score > upper_threshold:
        target_weight = final_weight
    elif score < lower_threshold:
        target_weight = initial_weight
    else:
        target_weight = current_weight

    updated_weight = current_weight + (target_weight - current_weight) * step_size
    reward_cfg.weight = float(updated_weight)
    env.reward_manager.set_term_cfg(reward_term_name, reward_cfg)

    return {
        "weight": torch.tensor(updated_weight, device=env.device),
        "tracking_score": tracking_score,
    }
