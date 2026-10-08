"""Compatibility import for the MuJoCo benchmark environment."""

from .mujoco_benchmark_env import MujocoBenchmarkEnv

BenchmarkEnv = MujocoBenchmarkEnv

__all__ = ["BenchmarkEnv", "MujocoBenchmarkEnv"]
