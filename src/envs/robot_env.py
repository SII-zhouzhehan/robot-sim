"""Project robot environment.

The project has one physics backend: MuJoCo. This module keeps the original
import path working for notebooks and third-party scripts while exposing the
MuJoCo implementation directly.
"""

from .mujoco_robot_env import MujocoRobotEnv

MorphingRobotEnv = MujocoRobotEnv

__all__ = ["MorphingRobotEnv", "MujocoRobotEnv"]
