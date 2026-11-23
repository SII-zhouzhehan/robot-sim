"""Closed-chain geometry, actual actuation, and coordinated torque checks."""
import hashlib
import xml.etree.ElementTree as ET

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from controllers.closed_loop_control import MorphTorqueController
from envs.mujoco_closed_loop import (CLOSURE_JOINT, ClosedLoopKinematics,
                                     build_closed_loop_model, tangent_projector)
from envs.mujoco_model import DEFAULT_MODEL
from envs.mujoco_robot_env import MujocoRobotEnv


@pytest.fixture(scope="module")
def closure(tmp_path_factory):
    if not DEFAULT_MODEL.exists():
        pytest.skip("Private robot model is unavailable")
    digest = hashlib.sha256(DEFAULT_MODEL.read_bytes()).hexdigest()
    path, _ = build_closed_loop_model(output=tmp_path_factory.mktemp("closed_loop") / "closed.xml")
    assert hashlib.sha256(DEFAULT_MODEL.read_bytes()).hexdigest() == digest
    model = mujoco.MjModel.from_xml_path(str(path))
    ik = ClosedLoopKinematics(model)
    return path, model, ik, ik.trajectory(121)


def test_real_eighth_hinge_and_motor(closure):
    _, model, _, _ = closure
    assert (model.nq, model.nv, model.nu, model.neq) == (27, 26, 20, 1)
    assert model.joint("joint8").id != model.joint(CLOSURE_JOINT).id
    assert model.joint(CLOSURE_JOINT).type == mujoco.mjtJoint.mjJNT_HINGE
    data = mujoco.MjData(model)
    actuator = model.actuator(f"{CLOSURE_JOINT}_position").id
    data.ctrl[actuator] = 1 / 35
    mujoco.mj_forward(model, data)
    assert data.qfrc_actuator[model.joint(CLOSURE_JOINT).dofadr[0]] == pytest.approx(1)
    # The motor belongs to the physical loop, not a free visual joint.
    assert np.linalg.norm(data.qfrc_constraint) > 0.1
    assert np.linalg.norm(data.qacc[6:]) > 0.1


def test_manual_collision_exclusions_and_existing_masks_preserved(closure):
    path, closed, _, _ = closure
    source = mujoco.MjModel.from_xml_path(str(DEFAULT_MODEL))

    def exclusions(xml):
        return {frozenset((e.get("body1"), e.get("body2")))
                for e in ET.parse(xml).getroot().findall("contact/exclude")}

    assert exclusions(path) == exclusions(DEFAULT_MODEL)
    for g in range(source.ngeom):
        copied = closed.geom(source.geom(g).name).id
        assert closed.geom_contype[copied] == source.geom_contype[g]
        assert closed.geom_conaffinity[copied] == source.geom_conaffinity[g]


def test_projected_trajectory_closes_and_respects_existing_limits(closure):
    _, _, ik, (phis, q) = closure
    residual = np.array([ik.residual(row) for row in q])
    assert np.max(np.linalg.norm(residual[:, :3], axis=1)) * .2 < 2e-6
    assert np.max(np.linalg.norm(residual[:, 3:], axis=1)) < 2e-5
    assert np.all(q >= ik.lower - 1e-8) and np.all(q <= ik.upper + 1e-8)
    assert np.max(abs(np.diff(q, axis=0))) < np.deg2rad(10)
    np.testing.assert_allclose(q[np.argmin(abs(phis - 1.5))], 0, atol=1e-7)


def test_allocator_excludes_internal_preload(closure):
    _, model, ik, (phis, q) = closure
    data = mujoco.MjData(model)
    data.qpos[ik.qadr] = q[np.argmin(abs(phis - 2.25))]
    mujoco.mj_forward(model, data)
    dofs = model.jnt_dofadr[ik.ids]
    projector = tangent_projector(model, data, dofs)
    np.testing.assert_allclose(projector, projector.T, atol=1e-10)
    np.testing.assert_allclose(projector @ projector, projector, atol=1e-10)
    jac = data.efc_J.reshape(data.nefc, model.nv)[:6, dofs]
    np.testing.assert_allclose(jac @ projector, 0, atol=1e-9)
    assert np.trace(projector) == pytest.approx(2, abs=1e-6)


def test_eight_motor_dynamics_and_passive_pivots(closure):
    path, _, ik, trajectory = closure
    env = MujocoRobotEnv(path, passive_pivots=True, control_steps=8)
    env.model.opt.timestep = env.dt = 1 / 1920
    env.control_dt = 1 / 240
    env.model.actuator_forcelimited[:] = 0
    env.model.jnt_actfrclimited[:] = 0
    try:
        env.reset()
        controller = MorphTorqueController(env, trajectory)
        for phi in np.linspace(1.5, 1.7, 240):
            controller.step(phi)
        assert np.isfinite(env.data.qpos).all()
        assert np.max(abs(env.data.qpos[controller.qadr] - controller.target)) < np.deg2rad(3)
        assert np.linalg.norm(ik.errors(env.data)[:3]) * .2 < .0002
        np.testing.assert_allclose(env.data.actuator_force[env.pivot_actuators], 0, atol=1e-12)
        assert abs(env.data.qfrc_actuator[controller.dofs[-1]]) > .01
    finally:
        env.close()


def test_target_interpolation_reduces_command_spike_without_control_delay(closure):
    path, _, _, trajectory = closure
    peaks = []
    for interpolate in (False, True):
        env = MujocoRobotEnv(path, passive_pivots=True, control_steps=8)
        env.model.opt.timestep = env.dt = 1 / 1920
        env.control_dt = 1 / 240
        env.model.actuator_forcelimited[:] = 0
        env.model.jnt_actfrclimited[:] = 0
        try:
            env.reset()
            controller = MorphTorqueController(env, trajectory, interpolate_targets=interpolate)
            # Switch from native reset servos to the coordinated controller
            # and settle before isolating a single command-update transient.
            for _ in range(240):
                controller.step(1.5)
            torques = []

            def observe(e):
                mujoco.mj_forward(e.model, e.data)
                torques.append(e.data.qfrc_actuator[controller.dofs].copy())

            start = float(env.data.time)
            phi = 1.5 + .5 / 240
            controller.step(phi, observe)
            peaks.append(float(np.max(abs(np.asarray(torques)))))
            assert env.data.time - start == pytest.approx(1 / 240)
            np.testing.assert_allclose(controller.target, controller.target_at(phi), atol=1e-14)
        finally:
            env.close()
    # Actual actuator loads, from identical rest states; no changed gains,
    # reduced command endpoint, extra time, or disabled contacts.
    assert peaks[1] < .6 * peaks[0]


def test_invalid_or_overwriting_build_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        build_closed_loop_model(DEFAULT_MODEL, DEFAULT_MODEL)
    with pytest.raises(ValueError):
        build_closed_loop_model(output=tmp_path / "x.xml", added_mass_kg=-.1)
