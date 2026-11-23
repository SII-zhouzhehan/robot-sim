"""Compose the existing MorphGNN, PID and swerve algorithms for MuJoCo."""
from pathlib import Path

import numpy as np

from .pid_controller import VelocityPIDController
from .swerve_kinematics import SwerveDriveController


def angles_from_phi(phi):
    phi = float(np.clip(phi, 0, 3))
    if phi <= 1:
        a = 1.57 * (1 - phi)
        return np.array([a, 1.57, a, -1.57, a, 1.57, a])
    if phi <= 2:
        a = -1.57 * (2 * (2 - phi) - 1)
        return np.array([0, -a, 0, a, 0, -a, 0])
    a = -1.57 * (phi - 2)
    return np.array([a, -1.57, a, 1.57, a, -1.57, a])


class MorphologyPredictor:
    def __init__(self, source="gnn", checkpoint=None):
        if source not in ("gnn", "formula"):
            raise ValueError("source must be gnn or formula")
        self.source = source
        self.model = self.converter = self.edge_index = None
        self.checkpoint = None
        self._cache = {}
        if source == "gnn":
            import torch
            from models.model import MorphGNN
            from utils.graph_converter import RobotGraphConverter
            torch.set_num_threads(1)
            if checkpoint is None:
                root = Path(__file__).resolve().parents[2]
                candidates = sorted((root / "experiments").glob("phi_run_*/best_model.pt"), reverse=True)
                if not candidates:
                    raise FileNotFoundError("No MorphGNN checkpoint; use --source formula or --checkpoint PATH")
                checkpoint = candidates[0]
            self.checkpoint = Path(checkpoint).resolve()
            self.model = MorphGNN().cpu().eval()
            weights = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            self.model.load_state_dict(weights["model_state_dict"])
            self.converter = RobotGraphConverter()
            self.edge_index = self.converter.edge_index_7

    def predict(self, phi, source=None):
        source = source or self.source
        if source not in ("gnn", "formula"):
            raise ValueError("Unknown morphology source")
        phi = float(np.clip(phi, 0, 3))
        key = source, round(phi, 6)
        if key not in self._cache:
            if source == "formula":
                angles = angles_from_phi(phi)
            else:
                if self.model is None:
                    raise ValueError("GNN was not loaded; launch with --source gnn")
                import torch
                x = torch.from_numpy(self.converter.get_feature_matrix(phi))
                with torch.inference_mode():
                    final, _ = self.model(x, self.edge_index)
                angles = np.asarray(self.converter.graph_to_env_angles(final.squeeze(-1).numpy()))
            if not np.isfinite(angles).all():
                raise FloatingPointError("Non-finite morphology prediction")
            # A bounded cache retains stable shapes without growing throughout training.
            if len(self._cache) >= 512:
                self._cache.clear()
            self._cache[key] = np.clip(angles, -1.57, 1.57)
        return self._cache[key].copy()


class MujocoDriveController:
    def __init__(self, env, source="gnn", checkpoint=None, use_pid=True, phi_rate=0.5):
        self.env = env
        self.predictor = MorphologyPredictor(source, checkpoint)
        self.swerve = SwerveDriveController(env.WHEEL_RADIUS)
        self.pid = VelocityPIDController(dt=env.control_dt, max_linear_v=env.MAX_LINEAR_VELOCITY,
                                          ema_alpha=0.5)
        self.use_pid = use_pid
        self.phi_rate = phi_rate
        self.phi = 1.5

    def reset(self):
        self.phi = 1.5
        self.pid.reset()
        self.swerve.last_angles.update(dict(zip(self.env.STEER_CORNER_ORDER, self.env.get_steering_positions())))

    def advance(self, vx=0.0, vy=0.0, omega=0.0, phi_target=1.5, source=None, use_pid=None,
                *, physics_callback=None):
        target = float(np.clip(phi_target, 0, 3))
        self.phi += float(np.clip(target - self.phi, -self.phi_rate * self.env.control_dt,
                                  self.phi_rate * self.env.control_dt))
        self.pid.set_enabled(self.use_pid if use_pid is None else use_pid)
        self.pid.update_gains_for_phi(self.phi)
        actual = self.env.get_world_velocity()
        command = self.pid.compute(vx, vy, omega, *actual)
        angles = self.env.get_steering_positions()
        self.swerve.last_angles.update(dict(zip(self.env.STEER_CORNER_ORDER, angles)))
        if np.max(np.abs(command)) < 0.01:
            steering, velocity = angles, np.zeros(4)
        else:
            positions, corners = self.env.get_wheel_positions_for_kinematics()
            wheel_commands = self.swerve.compute_swerve_kinematics(*command, positions,
                                                                  self.env.get_steering_zero_yaws_imu())
            steering = [wheel_commands[c]["angle"] for c in corners]
            velocity = [wheel_commands[c]["speed"] for c in corners]
        blocks = self.predictor.predict(self.phi, source)
        settled = np.max(np.abs(self.env.joint_positions(self.env.block_ids) - blocks)) < 0.05
        self.env.set_pivot_lock(abs(target - self.phi) < 0.005 and settled)
        action = {"blocks": blocks, "steering": steering, "velocity": velocity}
        result = (self.env.step(action) if physics_callback is None else
                  self.env.step(action, physics_callback=physics_callback))
        return result
