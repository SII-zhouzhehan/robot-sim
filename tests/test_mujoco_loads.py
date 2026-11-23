"""Analytical validation of load units, wrench origin and observer isolation."""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from evaluation.mujoco_loads import JointLoadSampler, audit_masses, incline_robot_and_ground


def pendulum(gear=1):
    return mujoco.MjModel.from_xml_string(f"""
    <mujoco><option gravity="0 0 -9.8"/>
    <worldbody><body name="link" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <inertial pos="1 0 0" mass="2" diaginertia="0.1 0.1 0.1"/>
      <geom type="sphere" size="0.01" contype="0" conaffinity="0"/>
    </body></worldbody>
    <actuator><motor name="motor" joint="hinge" gear="{gear}"/></actuator>
    </mujoco>""")


def test_static_gravity_and_wrench_at_hinge_without_mutating_live_state():
    m = pendulum()
    d = mujoco.MjData(m)
    d.ctrl[0] = -19.6  # 2 kg at 1 m, rotation about +Y: gravity drives +Y.
    mujoco.mj_forward(m, d)
    before = {name: getattr(d, name).copy() for name in ("qpos", "qvel", "ctrl", "qacc", "qacc_warmstart")}
    r = JointLoadSampler(m, ["hinge"]).sample(d)
    assert r["gravity_nm"][0] == pytest.approx(-19.6)
    assert r["torque_nm"][0] == pytest.approx(-19.6)
    assert r["reaction_axis_nm"][0] == pytest.approx(-19.6)
    assert r["reaction_radial_n"][0] == pytest.approx(19.6)
    assert r["reaction_bending_nm"][0] == pytest.approx(0)
    assert r["full_balance_residual"] < 1e-10
    assert r["inertia_nm"][0] == pytest.approx(0, abs=1e-10)
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(d, name), value)


def test_actuator_force_is_converted_to_joint_output_torque_by_gear():
    m = pendulum(gear=4)
    d = mujoco.MjData(m)
    d.ctrl[0] = -3
    d.qpos[0] = .4
    d.qvel[0] = 1.2
    r = JointLoadSampler(m, ["hinge"]).sample(d)
    assert r["actuator_force"][0] == pytest.approx(-3)
    assert r["torque_nm"][0] == pytest.approx(-12)
    assert r["gravity_nm"][0] == pytest.approx(-19.6 * np.cos(.4))
    assert r["full_balance_residual"] < 1e-10


def test_passive_dry_friction_and_damping_are_separate_from_motor_torque():
    m = mujoco.MjModel.from_xml_string('''
    <mujoco><option gravity="0 0 0"/><worldbody><body>
      <joint name="hinge" type="hinge" frictionloss=".05" damping=".05"/>
      <geom type="sphere" size=".05" mass="1"/>
    </body></worldbody><actuator><motor name="motor" joint="hinge"/></actuator></mujoco>''')
    d = mujoco.MjData(m)
    sampler = JointLoadSampler(m, ["hinge"])
    for speed in (-2., 2.):
        d.qvel[0] = speed
        r = sampler.sample(d)
        assert r["torque_nm"][0] == 0
        assert r["joint_friction_nm"][0] == pytest.approx(-np.sign(speed) * .05)
        assert r["constraint_nm"][0] == pytest.approx(r["joint_friction_nm"][0])
        assert r["passive_nm"][0] == pytest.approx(-.05 * speed)
        assert r["inertia_nm"][0] == pytest.approx(-np.sign(speed) * .15)
        assert r["full_balance_residual"] < 1e-10


def test_world_applied_force_is_included_in_generalized_balance():
    m = pendulum()
    d = mujoco.MjData(m)
    d.xfrc_applied[1, 2] = 4  # Upward 4 N applied at the body COM, 1 m from hinge.
    d.ctrl[0] = -15.6
    r = JointLoadSampler(m, ["hinge"]).sample(d)
    assert r["applied_nm"][0] == pytest.approx(-4)
    assert r["inertia_nm"][0] == pytest.approx(0, abs=1e-10)
    assert r["full_balance_residual"] < 1e-10


def test_mass_audit_reads_urdf_inertial_values(tmp_path):
    path = tmp_path / "source.urdf"
    path.write_text('''<robot name="pendulum"><link name="link"><inertial>
      <origin xyz="1 0 0" rpy="0 0 0"/><mass value="2"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial></link></robot>''')
    audit = audit_masses(pendulum(), path)
    assert audit["urdf_total_mass_kg"] == pytest.approx(2)
    assert audit["mujoco_total_mass_kg"] == pytest.approx(2)
    assert audit["links"][0]["urdf_com_xyz_m"] == [1., 0., 0.]
    assert audit["links"][0]["mass_difference_kg"] == 0


def test_inclined_plane_preserves_ground_distance_and_resolves_slope_gravity():
    from types import SimpleNamespace
    m = mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 -9.8"/>
      <worldbody><geom name="ground" type="plane" size="3 3 .1"/>
      <body name="robot" pos=".3 .2 .4"><freejoint/>
        <geom type="sphere" size=".05" mass="2"/>
      </body></worldbody></mujoco>''')
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    e = SimpleNamespace(model=m, data=d, ground_geom_id=m.geom('ground').id)
    old_height = d.qpos[2]
    rotation = incline_robot_and_ground(e, 15)
    plane_normal = d.geom_xmat[e.ground_geom_id].reshape(3, 3)[:, 2]
    uphill = rotation[:, 0]
    assert plane_normal @ d.qpos[:3] == pytest.approx(old_height)
    assert m.opt.gravity @ uphill == pytest.approx(-9.8 * np.sin(np.deg2rad(15)))
    assert m.opt.gravity @ plane_normal == pytest.approx(-9.8 * np.cos(np.deg2rad(15)))
    # This detects stale MuJoCo identity-frame shortcuts leaving the plane flat.
    np.testing.assert_allclose(plane_normal, [-np.sin(np.deg2rad(15)), 0, np.cos(np.deg2rad(15))])
    np.testing.assert_allclose(d.qvel, 0)
