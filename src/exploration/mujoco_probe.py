"""MuJoCo adapter: bounded position ramps, position/velocity observations only.

The existing position servos and their force limits are retained. Identified
relations therefore describe this control setup; they are not passive-body
constraints. Each local trial restores a simulation snapshot for comparability.
Snapshot restoration is not a proposed real-hardware reset procedure.
"""
import time

import mujoco
import numpy as np

from controllers.mujoco_control import angles_from_phi
from envs.mujoco_robot_env import MujocoRobotEnv


def physical_joint_edges(model, joint_ids):
    """Joint line graph, collapsing rigid CAD frames into their welded body."""
    bodies = model.jnt_bodyid[np.asarray(joint_ids, dtype=int)]
    endpoints = [{int(model.body_weldid[body]),
                  int(model.body_weldid[model.body_parentid[body]])}
                 for body in bodies]
    return [(i, j) for i in range(len(bodies)) for j in range(i + 1, len(bodies))
            if endpoints[i] & endpoints[j]]


class MujocoJointProbe:
    def __init__(self, model_path=None, *, duration=.25, settle_seconds=2.0,
                 radius=.06, gui=False):
        if duration <= 0 or settle_seconds < 0 or radius <= 0:
            raise ValueError("Invalid probe timing or local radius")
        self.env = MujocoRobotEnv(model_path, control_steps=1, passive_pivots=True)
        self.env.set_configuration("8_shape_translate")
        self.env.reset(settle_seconds=settle_seconds)
        # Phi is used ONCE to specify the requested initial pose, never as a
        # learner feature, motion label, locking mask, or subsequent trajectory.
        self.anchor = angles_from_phi(0)
        np.testing.assert_allclose(self.env.current_block_angles, self.anchor)
        self.ids = self.env.block_ids.copy()
        self.names = tuple(mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_JOINT, int(j))
                           for j in self.ids)
        limits = self.env.model.jnt_range[self.ids]
        self.lower = np.maximum(limits[:, 0] - self.anchor, -radius)
        self.upper = np.minimum(limits[:, 1] - self.anchor, radius)
        self.steps = max(1, round(duration / self.env.dt))
        self.duration = self.steps * self.env.dt
        self.physical_edges = self._joint_adjacency()
        self.snapshot = mujoco.MjData(self.env.model)
        mujoco.mj_copyData(self.snapshot, self.env.model, self.env.data)
        self.initial_q = self.env.joint_positions(self.ids)
        self.gui = gui
        if gui:
            self.env.open_viewer()
        self.baseline = self.run(np.zeros(len(self.names)))

    def _joint_adjacency(self):
        # Line graph of the actual model's link connections; no fixed chain.
        return physical_joint_edges(self.env.model, self.ids)

    def restore(self):
        mujoco.mj_copyData(self.env.data, self.env.model, self.snapshot)
        self.env.current_block_angles = self.anchor.copy()
        self.env._pivot_anchors = None
        mujoco.mj_forward(self.env.model, self.env.data)

    def run(self, increment):
        increment = np.asarray(increment, dtype=float)
        if (increment.shape != self.anchor.shape or not np.isfinite(increment).all()
                or np.any(increment < self.lower - 1e-10) or np.any(increment > self.upper + 1e-10)):
            raise ValueError("Probe is outside local/actuator angle limits")
        self.restore()
        start = self.env.joint_positions(self.ids)
        positions, velocities, targets = [], [], []
        steering = self.env.get_steering_positions()
        for tick in range(self.steps):
            target = self.anchor + increment * (tick + 1) / self.steps
            self.env.step({"blocks": target, "steering": steering, "velocity": np.zeros(4)})
            positions.append(self.env.joint_positions(self.ids))
            velocities.append(self.env.joint_velocities(self.ids))
            targets.append(target.copy())
            if self.gui:
                if not self.env.viewer.is_running():
                    raise KeyboardInterrupt
                self.env.render()
                time.sleep(self.env.dt)
        q, qdot = np.asarray(positions), np.asarray(velocities)
        return {"increment": increment.copy(), "position_rad": q, "velocity_rad_s": qdot,
                "target_rad": np.asarray(targets), "displacement": q[-1] - start,
                "integrated_velocity": qdot.sum(axis=0) * self.env.dt}

    def response(self, trial, feedback="velocity"):
        if feedback == "position":
            key = "displacement"
        elif feedback == "velocity":
            key = "integrated_velocity"
        else:
            raise ValueError("Feedback must be position or velocity")
        # Paired null-input trial removes the repeatable initial drift, using
        # exactly the same observation channels as each perturbation trial.
        return trial[key] - self.baseline[key]

    def metadata(self):
        return {"model_path": str(self.env.model_path), "initial_phi": 0,
                "joint_names": list(self.names), "morph_joint_count": len(self.names),
                "equality_count": int(self.env.model.neq),
                "nominal_initial_angles_rad": self.anchor.tolist(),
                "actual_initial_angles_rad": self.initial_q.tolist(),
                "initial_tracking_max_rad": float(np.max(abs(self.initial_q - self.anchor))),
                "increment_lower_rad": self.lower.tolist(), "increment_upper_rad": self.upper.tolist(),
                "duration_seconds": self.duration, "physics_dt": self.env.dt,
                "baseline_drift_rad": self.baseline["displacement"].tolist(),
                "reset_protocol": "identical MuJoCo snapshot, simulation only",
                "actuation": "existing bounded position servos; smooth target ramps",
                "pivot_actuation": "passive", "force_or_current_feedback_used": False}

    def close(self):
        self.env.close()
