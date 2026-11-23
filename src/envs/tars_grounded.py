"""Control interface for the imported eight-motor TARS scene (SI units)."""
from pathlib import Path

import mujoco
import numpy as np

from controllers.swerve_kinematics import SwerveDriveController

SCENE = Path(__file__).resolve().parents[2] / "data/my_robot/mujoco/tars_mjcf_grounded/scene.xml"
MORPH_NAMES = tuple(f"dof_{i}" for i in range(1, 9))


class TarsGroundedEnv:
    """Use CAD module order (pivot 9,12,15,18) throughout, independent of pose."""

    def __init__(self, model_path=SCENE, control_steps=5):
        if not isinstance(control_steps, int) or control_steps < 1:
            raise ValueError("control_steps must be a positive integer")
        self.model = m = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(m)
        self.control_steps = control_steps
        self.dt = m.opt.timestep
        self.control_dt = self.dt * control_steps

        def joints(numbers):
            return np.array([m.joint(f"dof_{n}").id for n in numbers])

        def actuators(ids):
            result = []
            for joint in ids:
                act = m.actuator(m.joint(joint).name + "_position").id
                if m.actuator_trnid[act, 0] != joint or not m.actuator_forcelimited[act]:
                    raise ValueError("Expected a torque-limited joint servo")
                result.append(act)
            return np.array(result)

        self.block_ids = joints(range(1, 9))
        self.pivot_ids = joints(range(9, 21, 3))
        self.steering_ids = joints(range(10, 21, 3))
        self.wheel_ids = joints(range(11, 21, 3))
        self.block_actuators = actuators(self.block_ids)
        self.steering_actuators = actuators(self.steering_ids)
        self.wheel_actuators = actuators(self.wheel_ids)
        self.base_body = m.jnt_bodyid[np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE).item()]
        # Approximate rolling radius from axle height above the ground at rest.
        self.swerve = SwerveDriveController(wheel_radius=.028)
        self.reset()

    def joint_positions(self, ids):
        return self.data.qpos[self.model.jnt_qposadr[ids]].copy()

    def joint_velocities(self, ids):
        return self.data.qvel[self.model.jnt_dofadr[ids]].copy()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_step(self.model, self.data, nstep=round(.6 / self.dt))
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def observe(self):
        return {name: self.joint_positions(ids) for name, ids in (
            ("blocks", self.block_ids), ("pivots", self.pivot_ids),
            ("steering", self.steering_ids), ("wheels", self.wheel_ids))}

    def get_robot_pose(self):
        center = self.data.xanchor[self.wheel_ids].mean(axis=0)
        rotation = self.data.xmat[self.base_body].reshape(3, 3)
        return np.array([*center[:2], np.arctan2(rotation[1, 0], rotation[0, 0])])

    def step(self, action):
        """blocks and steering in joint radians, velocity in wheel rad/s.

        Omitted position targets hold their previous commands. Each wheel's
        speed request advances its position reference over this control period.
        """
        if action.keys() - {"blocks", "steering", "velocity"}:
            raise ValueError("Unknown TARS action channel")
        d, m = self.data, self.model
        targets = []
        for key, ids in (("blocks", self.block_actuators), ("steering", self.steering_actuators),
                         ("velocity", self.wheel_actuators)):
            default = np.zeros(len(ids)) if key == "velocity" else d.ctrl[ids]
            value = np.asarray(action.get(key, default), dtype=float)
            if value.shape != ids.shape or not np.isfinite(value).all():
                raise ValueError(f"{key} requires {len(ids)} finite values")
            targets.append(value)
        blocks, steering, speed = targets
        d.ctrl[self.block_actuators] = np.clip(blocks, *m.actuator_ctrlrange[self.block_actuators].T)
        d.ctrl[self.steering_actuators] = steering
        # Bound tracking lag by the position error at torque saturation. This
        # avoids accumulated position error when a wheel stalls or reverses.
        wheel = self.wheel_actuators
        lag = m.actuator_forcerange[wheel, 1] / m.actuator_gainprm[wheel, 0]
        for _ in range(self.control_steps):
            actual = self.joint_positions(self.wheel_ids)
            d.ctrl[wheel] = actual + np.clip(d.ctrl[wheel] + self.dt * speed - actual, -lag, lag)
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        if np.any(d.warning.number) or not np.isfinite(d.qvel).all():
            raise FloatingPointError("TARS simulation reported invalid dynamics")
        return self.observe()

    def drive(self, vx=0., vy=0., omega=0., *, blocks=None):
        """World-frame twist about the four-wheel center, on level ground."""
        pos = self.data.xanchor[self.wheel_ids, :2]
        axes = self.data.xaxis[self.wheel_ids]
        steer = self.joint_positions(self.steering_ids)
        # Positive wheel rotation rolls along axle cross world-up. Recompute
        # the mounting directions and module spacing as the chassis changes.
        heading = np.arctan2(-axes[:, 0], axes[:, 1])
        if np.min(self.data.xaxis[self.steering_ids, 2]) < .9:
            raise ValueError("Planar drive requires upright steering modules")
        names = range(4)
        self.swerve.last_angles.update(zip(names, steer))
        cmd = self.swerve.compute_swerve_kinematics(vx, vy, omega,
                    dict(zip(names, pos - pos.mean(axis=0))), dict(zip(names, heading - steer)))
        angles = np.array([cmd[i]["angle"] for i in names])
        angle_error = np.arctan2(np.sin(angles - steer), np.cos(angles - steer))
        # Wait for steering servos to align before driving. The final steering
        # target is not the angle reached during this control step.
        speeds = np.array([cmd[i]["speed"] for i in names])
        speeds[np.abs(angle_error) > .06] = 0.0
        action = {"steering": steer + np.clip(angle_error, -.10, .10), "velocity": speeds}
        if blocks is not None:
            action["blocks"] = blocks
        return self.step(action)
