from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import MISSING

import torch
import trimesh

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.configclass import configclass


@configclass
class VirtualObstacleCfg:
    """Configuration for a virtual obstacle generated from terrain geometry."""

    class_type: type = MISSING
    visualizer: VisualizationMarkersCfg = MISSING


class VirtualObstacleBase(ABC):
    def __init__(self, cfg: VirtualObstacleCfg):
        self.cfg = cfg

    @abstractmethod
    def generate(self, mesh: trimesh.Trimesh, device: torch.device | str = "cpu") -> None:
        raise NotImplementedError

    @abstractmethod
    def disable_visualizer(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def visualize(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_points_penetration_offset(self, points: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

