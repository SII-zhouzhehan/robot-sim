"""Joint-output loads and CAD mass audit; measurement never mutates live physics."""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from envs.mujoco_model import BLOCK_NAMES, PIVOT_NAMES, STEERING_NAMES, WHEEL_NAMES


JOINT_GROUPS = {"block": BLOCK_NAMES, "pivot": PIVOT_NAMES,
                "steering": STEERING_NAMES, "wheel": WHEEL_NAMES}
JOINT_NAMES = tuple(name for names in JOINT_GROUPS.values() for name in names)


def incline_robot_and_ground(env, degrees):
    """Rotate a prepared free-base robot and its plane before a load test."""
    angle = np.deg2rad(degrees)
    quat = np.array([np.cos(angle / 2), 0, -np.sin(angle / 2), 0])
    flat_rotation = np.zeros(9)
    mujoco.mju_quat2Mat(flat_rotation, quat)
    rotation = flat_rotation.reshape(3, 3)
    m, d = env.model, env.data
    m.geom_quat[env.ground_geom_id] = quat
    # Compiled identity rotation must no longer bypass the new quaternion.
    m.geom_sameframe[env.ground_geom_id] = 0
    d.qpos[:3] = rotation @ d.qpos[:3]
    previous = d.qpos[3:7].copy()
    mujoco.mju_mulQuat(d.qpos[3:7], quat, previous)
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    return rotation


class JointLoadSampler:
    """Read output torque and parent-child wrench for directly driven hinge joints.

    cfrc_int is a world-oriented [moment, force] wrench about the root subtree
    COM. Shift its moment to the joint anchor before splitting axial/bending
    components. Joint-limit reactions can contribute to its axial moment, so
    this structural moment is distinct from motor output torque.
    """

    def __init__(self, model, names=JOINT_NAMES):
        self.model = model
        self.names = tuple(names)
        self.jids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                              for n in names])
        if np.any(self.jids < 0) or np.any(model.jnt_type[self.jids] != mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("Load measurement requires named hinge joints")
        self.dofs = model.jnt_dofadr[self.jids]
        self.qpos = model.jnt_qposadr[self.jids]
        self.bodies = model.jnt_bodyid[self.jids]
        self.actuators = []
        for jid in self.jids:
            ids = np.flatnonzero((model.actuator_trnid[:, 0] == jid)
                                & (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT))
            if len(ids) != 1:
                raise ValueError("Each measured joint must have one joint-transmission actuator")
            self.actuators.append(int(ids[0]))
        self.actuators = np.asarray(self.actuators)
        self.data = mujoco.MjData(model)
        self.gravity_data = mujoco.MjData(model)

    def sample(self, live_data):
        m, d = self.model, self.data
        mujoco.mj_copyData(d, m, live_data)
        mujoco.mj_forward(m, d)
        mujoco.mj_rnePostConstraint(m, d)
        mujoco.mj_copyData(self.gravity_data, m, d)
        self.gravity_data.qvel[:] = 0
        mujoco.mj_fwdVelocity(m, self.gravity_data)
        inertia = np.zeros(m.nv)
        mujoco.mj_mulM(m, d, inertia, d.qacc)
        applied = d.qfrc_applied.copy()
        for body in np.flatnonzero(np.any(d.xfrc_applied != 0, axis=1)):
            mujoco.mj_applyFT(m, d, d.xfrc_applied[body, :3], d.xfrc_applied[body, 3:],
                             d.xipos[body], int(body), applied)
        residual = inertia + d.qfrc_bias - d.qfrc_passive - d.qfrc_constraint - applied - d.qfrc_actuator
        # Dry joint friction is a constraint force, not qfrc_passive or motor
        # effort. Separate it from ground/limit reactions for resistance audits.
        friction_force = np.where(d.efc_type == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF,
                                  d.efc_force, 0.)
        joint_friction = np.zeros(m.nv)
        mujoco.mj_mulJacTVec(m, d, joint_friction, friction_force)
        wrench = d.cfrc_int[self.bodies]
        force = wrench[:, 3:].copy()
        origin = d.subtree_com[m.body_rootid[self.bodies]]
        moment = wrench[:, :3] + np.cross(origin - d.xanchor[self.jids], force)
        axis = d.xaxis[self.jids]
        axial_moment = np.sum(moment * axis, axis=1)
        axial_force = np.sum(force * axis, axis=1)
        gear = m.actuator_gear[self.actuators, 0]
        position_target = d.ctrl[self.actuators] / gear
        # Wheel velocity targets are not position targets.
        position_target[m.actuator_biastype[self.actuators] == mujoco.mjtBias.mjBIAS_NONE] = np.nan
        for index, actuator in enumerate(self.actuators):
            if m.actuator_biasprm[actuator, 1] == 0:
                position_target[index] = np.nan
        return {
            "position_rad": d.qpos[self.qpos].copy(), "speed_rad_s": d.qvel[self.dofs].copy(),
            "target_rad": position_target, "torque_nm": d.qfrc_actuator[self.dofs].copy(),
            "actuator_force": d.actuator_force[self.actuators].copy(),
            "gravity_nm": self.gravity_data.qfrc_bias[self.dofs].copy(),
            "velocity_bias_nm": (d.qfrc_bias - self.gravity_data.qfrc_bias)[self.dofs].copy(),
            "inertia_nm": inertia[self.dofs].copy(), "passive_nm": d.qfrc_passive[self.dofs].copy(),
            "joint_friction_nm": joint_friction[self.dofs].copy(),
            "constraint_nm": d.qfrc_constraint[self.dofs].copy(), "applied_nm": applied[self.dofs].copy(),
            "reaction_axis_nm": axial_moment, "reaction_bending_nm": np.linalg.norm(
                moment - axial_moment[:, None] * axis, axis=1),
            "reaction_axial_n": axial_force, "reaction_radial_n": np.linalg.norm(
                force - axial_force[:, None] * axis, axis=1),
            "reaction_force_world_n": force, "reaction_moment_world_nm": moment,
            "balance_residual_nm": residual[self.dofs].copy(),
            "full_balance_residual": float(np.max(np.abs(residual))), "contacts": int(d.ncon),
        }


def audit_masses(model, urdf_path):
    """Preserve source inertial fields alongside compiled principal inertias."""
    root = ET.parse(Path(urdf_path)).getroot()
    parents = {j.find("child").get("link"): j for j in root.findall("joint")}
    rows = []
    for link in root.findall("link"):
        name = link.get("name")
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        inertial = link.find("inertial")
        if bid < 0 or inertial is None:
            raise ValueError(f"Missing compiled body or source inertia for {name}")
        origin, inertia = inertial.find("origin"), inertial.find("inertia")
        parent_joint = parents.get(name)
        source_mass = float(inertial.find("mass").get("value"))
        rows.append({
            "link": name, "parent_joint": parent_joint.get("name") if parent_joint is not None else "floating_base",
            "parent_link": parent_joint.find("parent").get("link") if parent_joint is not None else "world",
            "group": next((g for g in ("block", "pivot", "steering", "wheel") if name.startswith(g)), "base"),
            "urdf_mass_kg": source_mass, "mujoco_mass_kg": float(model.body_mass[bid]),
            "mass_difference_kg": float(model.body_mass[bid] - source_mass),
            "urdf_com_xyz_m": [float(v) for v in origin.get("xyz", "0 0 0").split()] if origin is not None else [0, 0, 0],
            "urdf_inertial_rpy_rad": [float(v) for v in origin.get("rpy", "0 0 0").split()] if origin is not None else [0, 0, 0],
            "urdf_inertia_kg_m2": {key: float(inertia.get(key)) for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")},
            "mujoco_com_xyz_m": model.body_ipos[bid].tolist(),
            "mujoco_principal_inertia_kg_m2": model.body_inertia[bid].tolist(),
            "mujoco_inertia_quat_wxyz": model.body_iquat[bid].tolist(),
        })
    groups = {g: sum(r["mujoco_mass_kg"] for r in rows if r["group"] == g)
              for g in ("base", "block", "pivot", "steering", "wheel")}
    return {"source_urdf": str(Path(urdf_path).resolve()), "links": rows, "group_mass_kg": groups,
            "urdf_total_mass_kg": sum(r["urdf_mass_kg"] for r in rows),
            "mujoco_total_mass_kg": float(model.body_mass.sum())}
