from dataclasses import MISSING
from typing import Callable

from isaaclab.utils import configclass

from .points_generator import grid3d_points_generator


@configclass
class PointsGeneratorCfg:
    """Configuration for volume-point generation."""

    func: Callable = MISSING


@configclass
class Grid3dPointsGeneratorCfg(PointsGeneratorCfg):
    func: Callable = grid3d_points_generator

    x_min: float = -1.0
    x_max: float = 1.0
    x_num: int = 10
    y_min: float = -1.0
    y_max: float = 1.0
    y_num: int = 10
    z_min: float = -1.0
    z_max: float = 1.0
    z_num: int = 10

