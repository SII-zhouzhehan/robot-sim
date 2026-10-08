"""Compatibility import for the MuJoCo Gymnasium environment."""

from .mujoco_brain_gym_env import MujocoBrainGymEnv

BrainGymEnv = MujocoBrainGymEnv

__all__ = ["BrainGymEnv", "MujocoBrainGymEnv"]
