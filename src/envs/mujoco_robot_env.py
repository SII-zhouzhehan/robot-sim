"""MuJoCo backend with the project's morphology/steering/wheel action interface."""
from pathlib import Path
import time

import mujoco
import numpy as np

from .configs import ROBOT_CONFIGS
from .mujoco_model import (BLOCK_NAMES, DEFAULT_MODEL, DEFAULT_URDF,
                           DEFAULT_PIVOT_DAMPING, DEFAULT_PIVOT_FRICTIONLOSS, PIVOT_NAMES,
                           STEERING_NAMES, WHEEL_NAMES, build_model)


class MujocoRobotEnv:
    WHEEL_RADIUS = 0.0255
    WHEEL_DIAMETER = 2 * WHEEL_RADIUS
    MAX_RPM = 160.0
    MAX_WHEEL_OMEGA = MAX_RPM * 2 * np.pi / 60
    MAX_LINEAR_VELOCITY = MAX_WHEEL_OMEGA * WHEEL_RADIUS
    STEER_CORNER_ORDER = ("RR", "FR", "RL", "FL")

    def __init__(self, model_path=None, gui=False, control_steps=1, urdf_path=DEFAULT_URDF, spec=None, *,
                 pivot_frictionloss=DEFAULT_PIVOT_FRICTIONLOSS, pivot_damping=DEFAULT_PIVOT_DAMPING,
                 passive_pivots=False):
        if not isinstance(control_steps, int) or control_steps < 1:
            raise ValueError("control_steps must be a positive integer")
        # Keep the historical constructor call ``Env(urdf_path, ...)`` usable
        # while routing it through the MuJoCo XML builder.  An URDF is an
        # input asset, never the runtime physics model.
        if model_path is not None and Path(model_path).suffix.lower() == ".urdf":
            urdf_path, model_path = model_path, None
        source_urdf = Path(urdf_path).resolve()
        if model_path is None:
            self.model_path = (DEFAULT_MODEL if source_urdf == DEFAULT_URDF.resolve()
                               else source_urdf.parent / "mujoco" / f"{source_urdf.stem}.xml").resolve()
        else:
            self.model_path = Path(model_path).resolve()
        if spec is None:
            if not self.model_path.exists():
                build_model(source_urdf, self.model_path)
            self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        else:
            self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.gui = gui
        self.control_steps = control_steps
        self.dt = float(self.model.opt.timestep)
        self.control_dt = self.dt * control_steps
        self.viewer = None
        self._next_render = 0.0
        self.current_config_name = "O_straight"
        self.current_block_angles = np.zeros(7)
        self.joint_name_to_id = {name: self._id(mujoco.mjtObj.mjOBJ_JOINT, name)
                                 for name in (*BLOCK_NAMES, *STEERING_NAMES, *WHEEL_NAMES, *PIVOT_NAMES)}
        self.block_ids = np.array([self.joint_name_to_id[n] for n in BLOCK_NAMES])
        self.steering_ids = np.array([self.joint_name_to_id[n] for n in STEERING_NAMES])
        self.wheel_ids = np.array([self.joint_name_to_id[n] for n in WHEEL_NAMES])
        self.pivot_ids = np.array([self.joint_name_to_id[n] for n in PIVOT_NAMES])
        self.block_actuators = self._actuators(BLOCK_NAMES, "position")
        self.steering_actuators = self._actuators(STEERING_NAMES, "position")
        self.wheel_actuators = self._actuators(WHEEL_NAMES, "velocity")
        self.pivot_actuators = self._actuators(PIVOT_NAMES, "position")
        self.base_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "base")
        self.steering_body_ids = self.model.jnt_bodyid[self.steering_ids]
        self.wheel_geom_ids = np.array([self._id(mujoco.mjtObj.mjOBJ_GEOM, f"wheel{i}_collision") for i in range(1, 5)])
        self.robot_geom_ids = np.flatnonzero(self.model.geom_group == 2)
        self.ground_geom_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self._robot_geom_set = set(self.robot_geom_ids.tolist())
        self._wheel_geom_set = set(self.wheel_geom_ids.tolist())
        self._steering_signs = np.ones(4)
        self._pivot_anchors = None
        self._pivot_drive_gain = self.model.actuator_gainprm[self.pivot_actuators].copy()
        self._pivot_drive_bias = self.model.actuator_biasprm[self.pivot_actuators].copy()
        self.passive_pivots = False
        # Apply defaults even to previously generated XML caches. Passing None
        # explicitly preserves custom XML resistance for that parameter.
        self.configure_pivot_resistance(frictionloss=pivot_frictionloss, damping=pivot_damping)
        self.set_passive_pivots(passive_pivots)

    def _id(self, kind, name):
        value = mujoco.mj_name2id(self.model, kind, name)
        if value < 0:
            raise ValueError(f"MuJoCo model is missing {kind.name}: {name}")
        return value

    def _actuators(self, names, suffix):
        return np.array([self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_{suffix}") for n in names])

    def joint_positions(self, ids):
        return self.data.qpos[self.model.jnt_qposadr[ids]].copy()

    def joint_velocities(self, ids):
        return self.data.qvel[self.model.jnt_dofadr[ids]].copy()

    def get_steering_positions(self):
        # Swerve's steering convention is positive rotation around world +Z.
        return self.joint_positions(self.steering_ids) * self._steering_signs

    def configure_pivot_resistance(self, *, frictionloss=None, damping=None):
        """Set passive hinge resistance before reset (Nm and Nm*s/rad).

        None preserves the XML value; zero disables that resistance. Dry
        friction is a bounded constraint torque, not a position-hold servo.
        """
        values = {"frictionloss": frictionloss, "damping": damping}
        for name, value in values.items():
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError(f"pivot {name} must be finite and nonnegative")
        dofs = self.model.jnt_dofadr[self.pivot_ids]
        if frictionloss is not None:
            self.model.dof_frictionloss[dofs] = frictionloss
        if damping is not None:
            self.model.dof_damping[dofs] = damping
        mujoco.mj_forward(self.model, self.data)

    def set_passive_pivots(self, enabled=True):
        """Disable the legacy upright servo while retaining hinge resistance."""
        self.passive_pivots = bool(enabled)
        self._pivot_anchors = None
        ids = self.pivot_actuators
        self.model.actuator_gainprm[ids] = 0 if enabled else self._pivot_drive_gain
        self.model.actuator_biasprm[ids] = 0 if enabled else self._pivot_drive_bias
        if enabled:
            self.data.ctrl[ids] = 0
        mujoco.mj_forward(self.model, self.data)

    def set_configuration(self, name):
        if name not in ROBOT_CONFIGS:
            raise ValueError(f"Unknown robot configuration: {name}")
        self.current_config_name = name

    def reset(self, *, pose=(0.0, 0.0, 0.0), settle_seconds=1.0):
        mujoco.mj_resetData(self.model, self.data)
        self._pivot_anchors = None
        free = self._id(mujoco.mjtObj.mjOBJ_JOINT, "floating_base")
        address = self.model.jnt_qposadr[free]
        yaw = float(pose[2])
        self.data.qpos[address:address + 3] = [0, 0, 0.35]
        self.data.qpos[address + 3:address + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        config = ROBOT_CONFIGS[self.current_config_name]
        self.data.qpos[self.model.jnt_qposadr[self.block_ids]] = config["block_angles"]
        self.data.qpos[self.model.jnt_qposadr[self.steering_ids]] = config["steering_init"]
        self.current_block_angles = np.array(config["block_angles"], dtype=float)
        self.data.ctrl[self.block_actuators] = self.current_block_angles
        self.data.ctrl[self.steering_actuators] = config["steering_init"]
        mujoco.mj_forward(self.model, self.data)
        self._steering_signs = np.where(self.data.xaxis[self.steering_ids, 2] >= 0, 1.0, -1.0)
        for _ in range(round(settle_seconds / self.dt)):
            self.update_pivot_hold()
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        center = self.get_support_center()
        self.data.qpos[address:address + 2] += np.asarray(pose[:2]) - center[:2]
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self._check_state()
        return self._get_obs()

    def set_pivot_lock(self, enabled):
        """Capture a finite-torque hold target; never teleport joint positions."""
        if self.passive_pivots:
            self._pivot_anchors = None
            return
        if enabled and self._pivot_anchors is None:
            self._pivot_anchors = self.joint_positions(self.pivot_ids)
        elif not enabled:
            self._pivot_anchors = None

    def update_pivot_hold(self):
        if self.passive_pivots:
            self.data.ctrl[self.pivot_actuators] = 0
            return
        if self._pivot_anchors is not None:
            targets = self._pivot_anchors
        else:
            # Keep steering columns upright during folding with bounded physical
            # torques. Retargeting to the current angle lets gravity ratchet the
            # legs sideways; resetting qpos to lock them would bypass physics.
            axis = self.data.xaxis[self.pivot_ids]
            columns = self.data.xaxis[self.steering_ids]
            up = np.array([0.0, 0.0, 1.0])
            projected = columns - np.sum(columns * axis, axis=1)[:, None] * axis
            desired = up - (axis @ up)[:, None] * axis
            delta = np.arctan2(np.sum(axis * np.cross(projected, desired), axis=1),
                               np.sum(projected * desired, axis=1))
            limits = self.model.jnt_range[self.pivot_ids]
            targets = np.clip(self.joint_positions(self.pivot_ids) + delta, limits[:, 0], limits[:, 1])
        self.data.ctrl[self.pivot_actuators] = targets

    @staticmethod
    def _command(value, count, name):
        array = np.asarray(value, dtype=float)
        if array.shape != (count,) or not np.isfinite(array).all():
            raise ValueError(f"{name} requires {count} finite values")
        return array

    def step(self, action, *, physics_callback=None):
        """Advance control, optionally observing every pre-integration physics state.

        The callback receives this environment after actuator targets are set.
        Observers should copy MjData before recomputing derived physics fields.
        """
        blocks = self._command(action.get("blocks", self.current_block_angles), 7, "blocks")
        steering = self._command(action.get("steering", self.get_steering_positions()), 4, "steering")
        velocity = self._command(action.get("velocity", np.zeros(4)), 4, "velocity")
        limits = self.model.jnt_range[self.block_ids]
        blocks = np.clip(blocks, limits[:, 0], limits[:, 1])
        self.current_block_angles = blocks.copy()
        self.data.ctrl[self.block_actuators] = blocks
        self.data.ctrl[self.steering_actuators] = steering * self._steering_signs
        self.data.ctrl[self.wheel_actuators] = np.clip(velocity, -self.MAX_WHEEL_OMEGA, self.MAX_WHEEL_OMEGA)
        for _ in range(self.control_steps):
            self.update_pivot_hold()
            if physics_callback is not None:
                physics_callback(self)
            mujoco.mj_step(self.model, self.data)
        # mj_step leaves Cartesian kinematics at the pre-integration state.
        mujoco.mj_forward(self.model, self.data)
        self._check_state()
        return self._get_obs(), 0.0, False, {"simulation_time": float(self.data.time)}

    def _check_state(self):
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("Non-finite MuJoCo state")
        if np.max(np.abs(self.data.qvel)) > 1e4:
            raise FloatingPointError("MuJoCo velocity diverged")
        for warning in (mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQPOS,
                        mujoco.mjtWarning.mjWARN_BADQVEL):
            if self.data.warning[warning].number:
                raise FloatingPointError(f"MuJoCo reported {warning.name}")

    def _get_obs(self):
        return np.concatenate([self.joint_positions(self.block_ids), self.joint_positions(self.pivot_ids),
                               self.get_steering_positions(), self.joint_positions(self.wheel_ids)]).astype(np.float32)

    def get_support_center(self):
        return np.mean(self.data.xipos[self.steering_body_ids], axis=0)

    def get_robot_yaw(self):
        rotation = self.data.xmat[self.base_body_id].reshape(3, 3)
        # Phi=3 pitches the base's local X axis almost vertically. Euler yaw
        # from that axis becomes ill-conditioned and can jump by pi. The local
        # transverse Y axis remains horizontal; Y x world-up gives forward.
        return float(np.arctan2(-rotation[0, 1], rotation[1, 1]))

    def get_robot_pose(self):
        center = self.get_support_center()
        return float(center[0]), float(center[1]), self.get_robot_yaw()

    def get_world_velocity(self):
        velocity = np.zeros(6)
        linear = []
        for body_id in self.steering_body_ids:
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, int(body_id), velocity, 0)
            linear.append(velocity[3:].copy())
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base_body_id, velocity, 0)
        mean = np.mean(linear, axis=0)
        return float(mean[0]), float(mean[1]), float(velocity[2])

    def get_wheel_positions_world_frame(self):
        xy = self.data.xipos[self.steering_body_ids, :2]
        return (xy - xy.mean(axis=0)).tolist()

    def get_wheel_positions_for_kinematics(self):
        return dict(zip(self.STEER_CORNER_ORDER, self.get_wheel_positions_world_frame())), list(self.STEER_CORNER_ORDER)

    def get_steering_zero_yaws_imu(self):
        # The rolling direction for positive wheel speed is axle × world up.
        # Derive it from the imported joint axes, avoiding fixed RPY magic offsets.
        axle = self.data.xaxis[self.wheel_ids]
        direction = np.cross(axle, np.array([0.0, 0.0, 1.0]))
        heading = np.arctan2(direction[:, 1], direction[:, 0])
        return dict(zip(self.STEER_CORNER_ORDER, heading - self.get_steering_positions()))

    def get_steering_joint_names(self):
        return list(STEERING_NAMES)

    def get_robot_bounds(self):
        ids = self.robot_geom_ids
        rotations = self.data.geom_xmat[ids].reshape(-1, 3, 3)
        local = self.model.geom_aabb[ids]
        centers = self.data.geom_xpos[ids] + np.einsum("nij,nj->ni", rotations, local[:, :3])
        extents = np.einsum("nij,nj->ni", np.abs(rotations), local[:, 3:])
        return np.stack([(centers - extents).min(axis=0), (centers + extents).max(axis=0)])

    def get_contact_summary(self, obstacle_geom_ids=()):
        obstacles = set(obstacle_geom_ids)
        wheel_contacts = set()
        collision = False
        max_force, max_penetration = 0.0, 0.0
        force = np.zeros(6)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if self.ground_geom_id in pair:
                wheel_contacts.update(pair & self._wheel_geom_set)
            if pair & self._robot_geom_set and pair & obstacles:
                mujoco.mj_contactForce(self.model, self.data, index, force)
                normal_force = abs(float(force[0]))
                penetration = max(0.0, -float(contact.dist))
                max_force = max(max_force, normal_force)
                max_penetration = max(max_penetration, penetration)
                collision |= normal_force > 1.0 or penetration > 0.002
        return {"wheel_contacts": len(wheel_contacts), "supported": len(wheel_contacts) >= 2,
                "obstacle_collision": bool(collision), "max_obstacle_force_n": max_force,
                "max_penetration_m": max_penetration}

    def open_viewer(self, key_callback=None):
        import mujoco.viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=key_callback,
                                                   show_left_ui=False, show_right_ui=False)
        with self.viewer.lock():
            self.viewer.opt.geomgroup[2] = 0
            self.viewer.cam.distance = 2.2
            self.viewer.cam.azimuth = 125
            self.viewer.cam.elevation = -25
            self.viewer.cam.lookat[:] = self.get_support_center()
            self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self.viewer.sync()
        return self.viewer

    def render(self, *, force=False):
        now = time.perf_counter()
        if self.viewer is not None and self.viewer.is_running() and (force or now >= self._next_render):
            with self.viewer.lock():
                self.viewer.cam.lookat[:] = self.get_support_center()
            self.viewer.sync()
            self._next_render = now + 1 / 60

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
