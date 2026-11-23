"""Torque coordination for the provisional eight-motor closed mechanism."""
import mujoco
import numpy as np

from .mujoco_control import angles_from_phi


class MorphTorqueController:
    def __init__(self, env, trajectory=None, *, kp=1050.0, kd=2 * np.sqrt(30),
                 interpolate_targets=True):
        # Lazy import avoids a dependency cycle in the original seven-joint CLI.
        from envs.mujoco_closed_loop import MORPH_NAMES, tangent_projector
        self.env, self.trajectory = env, trajectory
        self.closed = trajectory is not None
        self.projector = tangent_projector
        names = MORPH_NAMES if self.closed else MORPH_NAMES[:7]
        self.joints = np.array([env.model.joint(n).id for n in names])
        self.qadr = env.model.jnt_qposadr[self.joints]
        self.dofs = env.model.jnt_dofadr[self.joints]
        self.actuators = np.array([env.model.actuator(f"{n}_position").id for n in names])
        self.kp, self.kd = kp, kd
        self.interpolate_targets = interpolate_targets
        # Bias damping stays implicit in MuJoCo. The command includes its
        # normal-space compensation, so total instantaneous torque is P @ PD.
        env.model.actuator_gainprm[self.actuators] = 0
        env.model.actuator_gainprm[self.actuators, 0] = 1
        env.model.actuator_biasprm[self.actuators] = 0
        env.model.actuator_biasprm[self.actuators, 2] = -kd
        self.target = self.target_at(1.5)
        self.steering = env.get_steering_positions()

    def target_at(self, phi):
        if not self.closed:
            return angles_from_phi(phi)
        phis, q = self.trajectory
        return np.array([np.interp(phi, phis, column) for column in q.T])

    def update(self, env):
        mujoco.mj_forward(env.model, env.data)
        q, v = env.data.qpos[self.qadr], env.data.qvel[self.dofs]
        torque = self.kp * (self.target - q) - self.kd * v
        if self.closed:
            torque = self.projector(env.model, env.data, self.dofs) @ torque
        env.data.ctrl[self.actuators] = torque + self.kd * v

    def step(self, phi, observer=None):
        previous = self.target.copy()
        target = self.target_at(phi)
        tick = 0

        def callback(env):
            nonlocal tick
            tick += 1
            # Preserve the 240 Hz endpoint commands, but avoid applying the
            # entire endpoint jump at the first physics tick. Each interval
            # ends at its requested target without adding a control delay.
            fraction = tick / env.control_steps if self.interpolate_targets else 1
            self.target = previous + (target - previous) * fraction
            self.update(env)
            if observer is not None:
                observer(env)

        result = self.env.step({"blocks": target[:7], "steering": self.steering,
                               "velocity": np.zeros(4)}, physics_callback=callback)
        self.target = target
        return result
