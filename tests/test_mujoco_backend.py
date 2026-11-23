"""Physical integration checks; private CAD assets are optional on public CI."""
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from controllers.mujoco_control import MujocoDriveController
from envs.mujoco_model import DEFAULT_URDF, build_model
from envs.mujoco_robot_env import MujocoRobotEnv
from envs.mujoco_benchmark_env import MujocoBenchmarkEnv
from envs.mujoco_brain_gym_env import MujocoBrainGymEnv
from utils.mujoco_control_panel import ControlPanel, ControlState


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    if not DEFAULT_URDF.exists():
        pytest.skip("Private robot URDF/STL assets are not present")
    path = tmp_path_factory.mktemp("mujoco_model") / "robot.xml"
    build_model(DEFAULT_URDF, path)
    return path


@pytest.fixture
def robot(model_path):
    env = MujocoRobotEnv(model_path, control_steps=12)
    env.reset()
    yield env
    env.close()


def test_portable_model_and_actuator_limits(robot):
    assert robot.model.nq == 26 and robot.model.nv == 25
    assert robot.model.nu == 19
    assert np.all(robot.model.jnt_qposadr[robot.block_ids] >= 7)
    robot.step({"velocity": [100, -100, 100, -100]})
    assert np.max(np.abs(robot.data.ctrl[robot.wheel_actuators])) == pytest.approx(robot.MAX_WHEEL_OMEGA)
    assert np.all(np.abs(robot.data.actuator_force[robot.wheel_actuators]) <= 0.35 + 1e-6)
    with pytest.raises(ValueError):
        robot.step({"blocks": [float("nan")] * 7})


def test_stationary_robot_remains_supported(robot):
    controller = MujocoDriveController(robot, source="formula")
    initial = np.array(robot.get_robot_pose())
    for _ in range(100):
        controller.advance()
    assert np.linalg.norm(np.array(robot.get_robot_pose())[:2] - initial[:2]) < 0.01
    assert robot.get_contact_summary()["wheel_contacts"] == 4


def test_physics_observer_sees_every_tick_without_changing_trajectory(robot):
    from evaluation.mujoco_loads import JointLoadSampler
    sampler = JointLoadSampler(robot.model)
    controller = MujocoDriveController(robot, source="formula")
    saved = mujoco.MjData(robot.model)
    mujoco.mj_copyData(saved, robot.model, robot.data)
    times, readings = [], []

    def observe(env):
        times.append(float(env.data.time))
        readings.append(sampler.sample(env.data))

    controller.advance(phi_target=1.51, physics_callback=observe)
    observed_qpos = robot.data.qpos.copy()
    observed_qvel = robot.data.qvel.copy()
    mujoco.mj_copyData(robot.data, robot.model, saved)
    robot._pivot_anchors = None
    controller = MujocoDriveController(robot, source="formula")
    controller.advance(phi_target=1.51)
    np.testing.assert_array_equal(robot.data.qpos, observed_qpos)
    np.testing.assert_array_equal(robot.data.qvel, observed_qvel)
    assert len(times) == robot.control_steps
    np.testing.assert_allclose(np.diff(times), robot.dt)
    assert np.isfinite(readings[0]["torque_nm"]).all()


@pytest.mark.parametrize("command,axis", [((.12, 0, 0), 0), ((0, .12, 0), 1), ((0, 0, .3), 2)])
def test_swerve_moves_in_commanded_world_direction(robot, command, axis):
    controller = MujocoDriveController(robot, source="formula")
    initial = np.array(robot.get_robot_pose())
    for _ in range(60):
        controller.advance(*command)
    delta = np.array(robot.get_robot_pose()) - initial
    assert delta[axis] > (0.2 if axis < 2 else 0.5)
    if axis < 2:
        assert abs(delta[1 - axis]) < 0.03
    assert robot.get_contact_summary()["supported"]


@pytest.mark.parametrize("phi", [0, 1, 1.5, 2, 2.5, 3])
def test_morphology_settles_and_can_drive(robot, phi):
    controller = MujocoDriveController(robot, source="formula")
    for _ in range(200):
        controller.advance(phi_target=phi)
    target = controller.predictor.predict(phi)
    assert np.max(np.abs(robot.joint_positions(robot.block_ids) - target)) < 0.08
    assert robot.get_contact_summary()["supported"]
    initial_x = robot.get_robot_pose()[0]
    if phi == 3:
        headings = []
        for _ in range(20):
            controller.advance(phi_target=phi)
            headings.append(robot.get_robot_yaw())
        assert np.max(np.abs(np.diff(np.unwrap(headings)))) < 0.03
    for _ in range(60):
        controller.advance(.1, 0, 0, phi_target=phi)
    assert robot.get_robot_pose()[0] - initial_x > 0.15
    assert robot.get_contact_summary()["supported"]


@pytest.mark.parametrize("scene,kind", [("A", "narrow_gate"), ("B", "none"), ("C", "narrow_gate")])
def test_scene_reset_preserves_world_rng_and_centroid(model_path, scene, kind):
    state = np.random.get_state()
    env = MujocoBenchmarkEnv(model_path, scene_id=scene, seed=123)
    try:
        observation = env.reset()
        after = np.random.get_state()
        assert np.array_equal(state[1], after[1]) and state[2:] == after[2:]
        assert observation["closest_type"] == kind
        assert np.linalg.norm(observation["pose"][:2]) < 1e-6
        bounds = env.core_env.get_robot_bounds()
        assert np.all(bounds[0] < bounds[1])
        assert env.get_info()["engine"] == "mujoco"
        if scene == "B":
            values = np.r_[0.0, env.task.waypoints[:, 1]]
            assert env.task.global_traj[:, 1].min() >= values.min() - 1e-9
            assert env.task.global_traj[:, 1].max() <= values.max() + 1e-9
    finally:
        env.close()


def test_gymnasium_reset_step_and_time_limit(model_path):
    env = MujocoBrainGymEnv(model_path, scene_id="B", max_steps=1, source="formula", imitation_weight=0)
    try:
        observation, info = env.reset(seed=42)
        assert env.observation_space.contains(observation)
        start = float(env.env.core_env.data.time)
        observation, reward, terminated, truncated, info = env.step(np.zeros(3))
        assert env.observation_space.contains(observation) and np.isfinite(reward)
        assert not terminated and truncated
        assert float(env.env.core_env.data.time) - start == pytest.approx(1.0)
        assert info["engine"] == "mujoco"
    finally:
        env.close()


def test_local_slider_panel_controls_and_rejects_nonfinite_commands():
    state = ControlState(source="formula")
    panel = ControlPanel(state, open_browser=False)
    try:
        with urlopen(panel.url) as response:
            assert "机器人控制" in response.read().decode()
        request = Request(panel.url + "command", json.dumps({"phi": 2, "vx": .1}).encode(),
                          {"Content-Type": "application/json"}, method="POST")
        with urlopen(request) as response:
            assert response.status == 204
        with urlopen(panel.url + "state") as response:
            command = json.load(response)
        assert command["phi"] == 2 and command["vx"] == .1
        bad = Request(panel.url + "command", b'{"vx": NaN}', method="POST")
        with pytest.raises(HTTPError):
            urlopen(bad)
        state.key(ord(" "))
        assert state.snapshot()["vx"] == 0
    finally:
        panel.close()
