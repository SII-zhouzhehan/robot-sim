"""Eight-motor experiment adapter; observations are angles and velocities.

Assembly welds are used only to construct the known physical graph. The loop
Jacobian is never used by the motor controller or passed to the learner.
The existing compatible Phi trajectory supplies the initial configuration only.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib

import mujoco
import numpy as np

from envs.mujoco_closed_loop import ClosedLoopKinematics, DEFAULT_CLOSED_MODEL, MORPH_NAMES
from envs.mujoco_robot_env import MujocoRobotEnv


def closed_physical_edges(model, joint_ids):
    """Line graph of joints, including known fixed assembly weld connections."""
    parent = list(range(model.nbody))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        parent[root(a)] = root(b)

    for body in range(model.nbody):
        union(body, int(model.body_weldid[body]))
    for eq in range(model.neq):
        if (model.eq_type[eq] == mujoco.mjtEq.mjEQ_WELD
                and model.eq_objtype[eq] == mujoco.mjtObj.mjOBJ_BODY
                and model.eq_active0[eq]):
            union(int(model.eq_obj1id[eq]), int(model.eq_obj2id[eq]))
    endpoints = []
    for joint in joint_ids:
        child = int(model.jnt_bodyid[joint])
        endpoints.append({root(child), root(int(model.body_parentid[child]))})
    return [(i, j) for i in range(len(endpoints)) for j in range(i + 1, len(endpoints))
            if endpoints[i] & endpoints[j]]


@dataclass
class ProbeState:
    data: object
    target: np.ndarray
    q: np.ndarray


class ClosedLoopProbe:
    def __init__(self, path=DEFAULT_CLOSED_MODEL, *, hz=1920, duration=.6, kp=100., kd=4.,
                 torque_limit=10., initial_angles=None):
        if hz < 960 or hz % 240 or duration <= 0 or min(kp, kd, torque_limit) <= 0:
            raise ValueError("Invalid probe settings")
        self.path = Path(path).resolve()
        self.source_hash = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.env = MujocoRobotEnv(self.path, passive_pivots=True)
        self.model, self.data = self.env.model, self.env.data
        m = self.model
        if m.neq < 1:
            raise ValueError("Expected an assembled closed-loop model")
        self.names = tuple(MORPH_NAMES)
        self.ids = np.array([m.joint(n).id for n in self.names])
        self.qadr, self.dofs = m.jnt_qposadr[self.ids], m.jnt_dofadr[self.ids]
        self.actuators = np.array([m.actuator(n + "_position").id for n in self.names])
        self.edges = closed_physical_edges(m, self.ids)
        self.kinematics = ClosedLoopKinematics(m)  # setup and evaluator, NOT policy
        self.reference = (self.kinematics.trajectory(121)[1][0] if initial_angles is None
                          else np.asarray(initial_angles, dtype=float).copy())
        if self.reference.shape != (len(self.names),) or not np.isfinite(self.reference).all():
            raise ValueError("Invalid initial angles")
        m.opt.timestep = self.env.dt = 1 / hz
        self.hz, self.duration = hz, duration
        self.steps = round(duration * hz)
        self.kp, self.kd, self.torque_limit = kp, kd, torque_limit
        # Independent bounded PD servos. No exact tangent-space allocation.
        m.actuator_gainprm[self.actuators] = 0
        m.actuator_gainprm[self.actuators, 0] = kp
        m.actuator_biasprm[self.actuators] = 0
        m.actuator_biasprm[self.actuators, 1] = -kp
        m.actuator_biasprm[self.actuators, 2] = -kd
        m.actuator_forcelimited[self.actuators] = 1
        m.actuator_forcerange[self.actuators] = [-torque_limit, torque_limit]
        m.jnt_actfrclimited[self.ids] = 1
        m.jnt_actfrcrange[self.ids] = [-torque_limit, torque_limit]
        self.lower = np.where(m.jnt_limited[self.ids], m.jnt_range[self.ids, 0], -np.inf)
        self.upper = np.where(m.jnt_limited[self.ids], m.jnt_range[self.ids, 1], np.inf)
        self.env.set_configuration("8_shape_translate")
        self.env.reset(settle_seconds=0)
        self.data.qpos[self.qadr] = self.reference
        self.data.ctrl[self.actuators] = self.reference
        mujoco.mj_forward(m, self.data)
        for _ in range(3 * hz):
            mujoco.mj_step(m, self.data)
        mujoco.mj_forward(m, self.data)
        self.env._check_state()
        self.initial = self.capture(self.reference)

    def capture(self, target):
        snapshot = mujoco.MjData(self.model)
        mujoco.mj_copyData(snapshot, self.model, self.data)
        return ProbeState(snapshot, np.asarray(target).copy(), self.data.qpos[self.qadr].copy())

    def bounds(self, state, amplitude, radius=.3):
        # Radius also bounds accumulated target mismatch / preload during search.
        lower = np.maximum(self.lower, self.initial.target - radius) - state.target
        upper = np.minimum(self.upper, self.initial.target + radius) - state.target
        return np.maximum(lower, -amplitude), np.minimum(upper, amplitude)

    def run(self, state, increment):
        increment = np.asarray(increment, dtype=float)
        if increment.shape != state.target.shape or not np.isfinite(increment).all():
            raise ValueError("Invalid eight-motor command")
        target = state.target + increment
        if np.any(target < self.lower - 1e-9) or np.any(target > self.upper + 1e-9):
            raise ValueError("Command outside mechanical joint limits")
        m, d = self.model, self.data
        mujoco.mj_copyData(d, m, state.data)
        peak_speed = peak_torque = max_gap = max_angle = 0.
        positions, velocities, timestamps = [], [], []
        integrated_velocity = np.zeros(len(self.names))
        valid, reason = True, "completed"
        for tick in range(self.steps):
            d.ctrl[self.actuators] = state.target + increment * min(1., (tick + 1) / (self.steps / 2))
            mujoco.mj_step(m, d)
            v = d.qvel[self.dofs]
            integrated_velocity += v / self.hz
            peak_speed = max(peak_speed, float(np.max(abs(v))))
            peak_torque = max(peak_torque, float(np.max(abs(d.qfrc_actuator[self.dofs]))))
            if tick % (self.hz // 240) == 0 or tick == self.steps - 1:
                mujoco.mj_forward(m, d)
                error = self.kinematics.errors(d)
                max_gap = max(max_gap, float(np.linalg.norm(error[:3]) * .2 * 1000))
                max_angle = max(max_angle, float(np.rad2deg(np.linalg.norm(error[3:]))))
                positions.append(d.qpos[self.qadr].copy())
                velocities.append(v.copy())
                timestamps.append((tick + 1) / self.hz)
                if not np.isfinite(d.qpos).all() or peak_speed > .4 or max_gap > .5 or max_angle > 1.5:
                    valid, reason = False, "simulation_guard"
                    break
        self.env._check_state()
        mujoco.mj_forward(m, d)
        # If a trial aborts, its partial result is recorded but cannot become an edge.
        end = self.capture(d.ctrl[self.actuators])
        return {"state": end, "command": increment.copy(), "delta": end.q - state.q,
                "integrated_velocity": integrated_velocity, "valid": valid, "reason": reason,
                "q": np.asarray(positions), "qdot": np.asarray(velocities),
                "time": np.asarray(timestamps), "peak_speed_rad_s": peak_speed,
                "peak_motor_torque_nm": peak_torque, "max_gap_mm": max_gap,
                "max_angle_deg": max_angle}

    def metadata(self):
        return {"model": str(self.path), "sha256": self.source_hash, "names": self.names,
                "neq": int(self.model.neq), "physical_edges": self.edges,
                "reference_q_deg": np.rad2deg(self.reference).tolist(),
                "settled_q_deg": np.rad2deg(self.initial.q).tolist(),
                "hz": self.hz, "duration": self.duration, "kp": self.kp, "kd": self.kd,
                "torque_limit_nm": self.torque_limit,
                "learner_feedback": ["q", "qdot"], "oracle_projection_in_controller": False,
                "initialization": "existing Phi=0-compatible IK; no subsequent Phi guidance",
                "collision_settings": "all existing exclusions and masks preserved",
                "reset_protocol": "MuJoCo snapshots; hardware return costs are not measured"}

    def close(self):
        self.env.close()
