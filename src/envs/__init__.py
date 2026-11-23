# src/envs/__init__.py 内容：
from .mujoco_robot_env import MujocoRobotEnv
from .mujoco_benchmark_env import MujocoBenchmarkEnv
from .mujoco_brain_gym_env import MujocoBrainGymEnv
from .configs import ROBOT_CONFIGS

from .tars_grounded import TarsGroundedEnv

MorphingRobotEnv = TarsGroundedEnv
BenchmarkEnv = MujocoBenchmarkEnv
BrainGymEnv = MujocoBrainGymEnv

__all__ = [
    "MujocoRobotEnv", "MujocoBenchmarkEnv", "MujocoBrainGymEnv",
    "MorphingRobotEnv", "TarsGroundedEnv", "BenchmarkEnv", "BrainGymEnv", "ROBOT_CONFIGS",
]
