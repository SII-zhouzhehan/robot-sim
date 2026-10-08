# src/envs/__init__.py 内容：
from .mujoco_robot_env import MujocoRobotEnv
from .mujoco_benchmark_env import MujocoBenchmarkEnv
from .mujoco_brain_gym_env import MujocoBrainGymEnv
from .configs import ROBOT_CONFIGS

MorphingRobotEnv = MujocoRobotEnv
BenchmarkEnv = MujocoBenchmarkEnv
BrainGymEnv = MujocoBrainGymEnv

__all__ = [
    "MujocoRobotEnv", "MujocoBenchmarkEnv", "MujocoBrainGymEnv",
    "MorphingRobotEnv", "BenchmarkEnv", "BrainGymEnv", "ROBOT_CONFIGS",
]
