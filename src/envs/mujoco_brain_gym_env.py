"""Gymnasium/PPO environment for the independent MuJoCo backend."""
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from brains.scripted_brain import ScriptedBrain
from controllers.mpc_controller import KinematicMPC
from controllers.mujoco_control import MujocoDriveController
from .mujoco_benchmark_env import MujocoBenchmarkEnv


class MujocoBrainGymEnv(gym.Env):
    """21 observations, three actions; one action advances one simulated second.

    Actions are local endpoint dx/dy in [-1, 1] metres and Phi in [-1, 1]
    mapped to [0, 3]. MPC runs at 20 Hz, swerve/PID at 60 Hz, physics at 240 Hz.
    Rewards and physical dynamics are defined by the MuJoCo backend.
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, model_path=None, gui=False, scene_id="C", max_steps=100,
                 source="gnn", checkpoint=None, imitation_weight=1.0, seed=42,
                 expert_lookahead_idx=5, **kwargs):
        super().__init__()
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected MuJoCo environment arguments: {unknown}")
        self.gui, self.max_steps, self.imitation_weight = gui, max_steps, imitation_weight
        self.env = MujocoBenchmarkEnv(model_path, gui=gui, scene_id=scene_id, control_steps=4, seed=seed)
        self.controller = MujocoDriveController(self.env.core_env, source, checkpoint)
        self.mpc = KinematicMPC(dt=0.05, N=10)
        self.teacher = ScriptedBrain()
        self.action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        self.observation_space = spaces.Box(-100.0, 100.0, (21,), dtype=np.float32)
        self.current_step = 0
        self.current_phi = 1.5
        self.last_cmd = np.zeros(3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset(seed=seed)
        self.controller.env = self.env.core_env
        self.controller.reset()
        self.current_step, self.current_phi = 0, 1.5
        self.last_cmd = np.zeros(3)
        self.mpc.u_prev[:] = 0
        self.teacher = ScriptedBrain()
        self.last_dist_to_goal = self.env.get_info()["dist_to_goal"]
        self._wall_start, self._sim_start = time.perf_counter(), float(self.env.core_env.data.time)
        if self.gui:
            self.env.core_env.open_viewer()
        return self._get_semantic_obs(), self.env.get_info()

    def _get_semantic_obs(self):
        x, y, yaw = self.env.core_env.get_robot_pose()
        gx, gy, _ = self.env.task.goal_pose
        values = [x, y, yaw, gx, gy, self.controller.phi, np.hypot(gx - x, gy - y)]
        candidates = []
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        for obstacle in self.env.obstacles:
            dx, dy = obstacle["x"] - x, obstacle.get("y", 0) - y
            local_x, local_y = dx * cos_y + dy * sin_y, -dx * sin_y + dy * cos_y
            distance = np.hypot(dx, dy)
            if distance < 5 and local_x > -1:
                kind = obstacle["type"]
                width = obstacle.get("width", 2 * obstacle.get("radius", 0.2))
                onehot = [float(kind == name) for name in ("narrow_gate", "low_platform", "bridge", "cylinders")]
                candidates.append((distance, [local_x, local_y, width, *onehot]))
        candidates.sort(key=lambda item: item[0])
        for i in range(2):
            values.extend(candidates[i][1] if i < len(candidates) else [0] * 7)
        return np.clip(np.asarray(values, dtype=np.float32), -100, 100)

    def _pace_gui(self):
        core = self.env.core_env
        core.render(force=True)
        deadline = self._wall_start + float(core.data.time) - self._sim_start
        remaining = deadline - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
        elif remaining < -0.25:
            self._wall_start = time.perf_counter()
            self._sim_start = float(core.data.time)

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError("Expected three finite actions")
        action = np.clip(action, -1, 1)
        x, y, yaw = self.env.core_env.get_robot_pose()
        dx = action[0] * np.cos(yaw) - action[1] * np.sin(yaw)
        dy = action[0] * np.sin(yaw) + action[1] * np.cos(yaw)
        target_phi = float(1.5 * (action[2] + 1))
        heading = np.arctan2(dy, dx) if np.hypot(dx, dy) > 0.01 else yaw
        yaw_delta = (heading - yaw + np.pi) % (2 * np.pi) - np.pi
        fractions = np.linspace(0, 1, 21)
        trajectory = np.column_stack([x + fractions * dx, y + fractions * dy, yaw + fractions * yaw_delta])
        collision, max_force, max_penetration = False, 0.0, 0.0
        terminated = False
        info = self.env.get_info()
        for tick in range(60):
            core = self.env.core_env
            if tick % 3 == 0:
                reference = trajectory[min(tick // 3 + 1, 20):]
                self.last_cmd = self.mpc.solve(np.array(core.get_robot_pose()), reference, self.last_cmd)
            current_yaw = core.get_robot_yaw()
            vx, vy, omega = self.last_cmd
            world_vx = vx * np.cos(current_yaw) - vy * np.sin(current_yaw)
            world_vy = vx * np.sin(current_yaw) + vy * np.cos(current_yaw)
            # Settle folding before advancing into a tight obstacle.
            joint_error = np.max(np.abs(core.joint_positions(core.block_ids)
                                       - self.controller.predictor.predict(target_phi)))
            if abs(target_phi - self.controller.phi) > 0.02 or joint_error > 0.08:
                world_vx = world_vy = omega = 0.0
            self.controller.advance(world_vx, world_vy, omega, target_phi)
            self.current_phi = self.controller.phi
            info = self.env.get_info()
            collision |= info["obstacle_collision"]
            max_force = max(max_force, info["max_obstacle_force_n"])
            max_penetration = max(max_penetration, info["max_penetration_m"])
            fallen = core.data.xpos[core.base_body_id, 2] < 0.04
            terminated = bool(info["success"] or fallen or collision)
            if self.gui:
                self._pace_gui()
                if not core.viewer.is_running():
                    info["viewer_closed"] = True
                    terminated = True
            if terminated:
                break
        self.current_step += 1
        progress = self.last_dist_to_goal - info["dist_to_goal"]
        self.last_dist_to_goal = info["dist_to_goal"]
        reward = 10 * progress - 0.1 - 0.05 * abs(target_phi - self.current_phi)
        if info["success"]:
            reward += 100
        if collision or fallen:
            reward -= 20
        if self.imitation_weight > 0 and not terminated:
            observed = self.env.get_obs()
            reference, teacher_phi = self.teacher.get_action(observed, self.env.task.global_traj)
            point = reference[min(5, len(reference) - 1), :2] - np.array(observed["pose"][:2])
            yaw = observed["pose"][2]
            expert_local = np.array([point[0] * np.cos(yaw) + point[1] * np.sin(yaw),
                                     -point[0] * np.sin(yaw) + point[1] * np.cos(yaw)])
            alignment = np.dot(action[:2], expert_local) / (np.linalg.norm(action[:2]) * np.linalg.norm(expert_local) + 1e-8)
            shape_reward = np.clip(1 - abs(target_phi - teacher_phi), -1, 1)
            reward += self.imitation_weight * (3 * alignment + shape_reward)
        info.update(obstacle_collision=bool(collision), max_obstacle_force_n=max_force,
                    max_penetration_m=max_penetration, phi=self.current_phi, is_success=bool(info["success"] and not collision))
        truncated = bool(self.current_step >= self.max_steps and not terminated)
        return self._get_semantic_obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.gui:
            self.env.core_env.render(force=True)

    def close(self):
        self.env.close()
