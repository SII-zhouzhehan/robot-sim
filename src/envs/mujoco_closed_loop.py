"""Provisional block4--block8 revolute closure, separate from the CAD model.

URDF is a tree. A real hinge on block4 carries a rotor that is welded to
block8, providing an actuated closed-chain hinge without welding the hinge
itself shut. Anchor/axis are inferred from the existing formula, not CAD.
"""
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from controllers.mujoco_control import angles_from_phi
from .mujoco_model import BLOCK_NAMES, DEFAULT_MODEL, PROJECT_ROOT, build_model

CLOSURE_JOINT = "joint_closure"  # joint8 already names a passive wheel pivot.
MORPH_NAMES = (*BLOCK_NAMES, CLOSURE_JOINT)
DEFAULT_CLOSED_MODEL = DEFAULT_MODEL.with_name("robot_closed_loop.xml")


def _values(x):
    return " ".join(f"{v:.12g}" for v in np.asarray(x).ravel())


def infer_closure(model):
    """Fit a common point and hinge axis over the original 0 <= Phi <= 3."""
    data = mujoco.MjData(model)
    ids = [model.joint(n).id for n in BLOCK_NAMES]
    bodies = [model.body(n).id for n in ("block4", "block8")]
    matrices, offsets = [], []
    for phi in np.linspace(0, 3, 301):
        data.qpos[model.jnt_qposadr[ids]] = angles_from_phi(phi)
        mujoco.mj_kinematics(model, data)
        r4, r8 = [data.xmat[b].reshape(3, 3).copy() for b in bodies]
        matrices.append(np.column_stack((r4, -r8)))
        offsets.append(data.xpos[bodies[1]] - data.xpos[bodies[0]])
    matrix, offset = np.concatenate(matrices), np.concatenate(offsets)
    anchors = np.linalg.lstsq(matrix, offset, rcond=None)[0]
    axis_pair = np.linalg.svd(matrix, full_matrices=False)[2][-1]
    axis = axis_pair[:3] / np.linalg.norm(axis_pair[:3])
    if axis[0] < 0:
        axis = -axis
    gaps = (matrix @ anchors - offset).reshape(-1, 3)
    return {"anchor_block4_m": anchors[:3].tolist(),
            "best_fit_anchor_block8_m": anchors[3:].tolist(),
            "axis_block4": axis.tolist(),
            "original_formula_max_gap_mm": float(np.linalg.norm(gaps, axis=1).max() * 1000)}


def build_closed_loop_model(source=DEFAULT_MODEL, output=DEFAULT_CLOSED_MODEL, *, added_mass_kg=0.15):
    """Add one hinge/actuator and six weld equations; preserve the source XML.

    Added motor/bracket mass is a declared assumption (half on each side).
    Zero requests a negligible 1e-6 kg rotor needed for nonzero inertia.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("Closed-loop output must differ from the source model")
    if not np.isfinite(added_mass_kg) or added_mass_kg < 0:
        raise ValueError("added_mass_kg must be finite and nonnegative")
    if not source.exists():
        build_model(output_path=source)
    original = mujoco.MjModel.from_xml_path(str(source))
    geometry = infer_closure(original)
    root = ET.parse(source).getroot()
    root.set("model", "morphing_robot_eight_motor_closed_loop_provisional")
    compiler = root.find("compiler")
    # Keep generated MJCF portable.  MuJoCo resolves ``meshdir`` relative to
    # the XML file, so express the source mesh directory relative to the
    # requested output instead of embedding the path of the machine that
    # generated the model.
    source_meshdir = Path(compiler.get("meshdir", ""))
    if not source_meshdir.is_absolute():
        source_meshdir = source.parent / source_meshdir
    compiler.set("meshdir", Path(os.path.relpath(source_meshdir.resolve(), output.parent)).as_posix())
    root.find("option").set("iterations", "100")
    root.find("option").set("jacobian", "dense")
    parent = root.find(".//body[@name='block4']")
    anchor, axis = geometry["anchor_block4_m"], geometry["axis_block4"]
    mass = max(added_mass_kg / 2, 1e-6)
    rotor = ET.SubElement(parent, "body", name="closure_rotor", pos=_values(anchor))
    ET.SubElement(rotor, "inertial", pos="0 0 0", mass=str(mass),
                  diaginertia=_values(np.full(3, mass * 0.00025)))
    ET.SubElement(rotor, "joint", name=CLOSURE_JOINT, type="hinge", axis=_values(axis),
                  limited="false", damping="0.01", armature="0.0001")
    # An orange disk marks the inferred joint; it introduces no contact support.
    quat = Rotation.align_vectors([axis], [[0, 0, 1]])[0].as_quat()
    ET.SubElement(rotor, "geom", name="closure_motor_visual", type="cylinder", size="0.025 0.018",
                  quat=_values(quat[[3, 0, 1, 2]]), group="1", contype="0", conaffinity="0",
                  density="0", rgba="1 0.45 0.08 1")
    if added_mass_kg > 0:
        stator = ET.SubElement(parent, "body", name="closure_stator", pos=_values(anchor))
        ET.SubElement(stator, "inertial", pos="0 0 0", mass=str(added_mass_kg / 2),
                      diaginertia=_values(np.full(3, added_mass_kg / 2 * 0.00025)))
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    ET.SubElement(equality, "weld", name="closure_weld", body1="closure_rotor", body2="block8",
                  solref="0.002 1", solimp="0.9999 0.9999 0.001", torquescale="1")
    ET.SubElement(root.find("actuator"), "position", name=f"{CLOSURE_JOINT}_position",
                  joint=CLOSURE_JOINT, kp="35", kv="2")
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    output.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(output))
    reference = ClosedLoopKinematics(model)
    try:
        source_reference = source.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        # Tests and callers may intentionally use an asset outside the
        # workspace; retain an absolute reference in that case.
        source_reference = str(source)
    geometry.update(added_mass_kg=float(added_mass_kg), source_model=source_reference,
                    joint_name=CLOSURE_JOINT, nq=model.nq, nv=model.nv, actuators=model.nu,
                    anchor_block8_reference_m=reference.mate_anchor.tolist(), reference_phi=1.5,
                    motor_sizing_valid=False,
                    geometry_validation="Hinge geometry inferred; intentional collision exclusions preserved. Assembly calibration and torque revalidation pending.",
                    note="Provisional inferred geometry; added motor/bracket mass is assumed.")
    output.with_suffix(".json").write_text(json.dumps(geometry, indent=2) + "\n")
    return output, geometry


class ClosedLoopKinematics:
    """Project the seven-angle formula onto a compatible eight-angle loop."""
    def __init__(self, model):
        self.model, self.data = model, mujoco.MjData(model)
        self.ids = np.array([model.joint(n).id for n in MORPH_NAMES])
        self.qadr = model.jnt_qposadr[self.ids]
        self.lower = np.r_[model.jnt_range[self.ids[:7], 0], -3.2]
        self.upper = np.r_[model.jnt_range[self.ids[:7], 1], 3.2]
        self.rotor, self.mate = [model.body(n).id for n in ("closure_rotor", "block8")]
        mujoco.mj_kinematics(model, self.data)
        r8 = self.data.xmat[self.mate].reshape(3, 3)
        rr = self.data.xmat[self.rotor].reshape(3, 3)
        self.mate_anchor = r8.T @ (self.data.xpos[self.rotor] - self.data.xpos[self.mate])
        self.relative_rotation = r8.T @ rr

    def residual(self, q):
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.errors(self.data)

    def errors(self, data):
        r8 = data.xmat[self.mate].reshape(3, 3)
        rr = data.xmat[self.rotor].reshape(3, 3)
        position = data.xpos[self.rotor] - data.xpos[self.mate] - r8 @ self.mate_anchor
        orientation = Rotation.from_matrix(r8.T @ rr @ self.relative_rotation.T).as_rotvec()
        return np.r_[position / 0.2, orientation]

    def solve(self, phi, seed=None):
        target = np.r_[angles_from_phi(phi), -angles_from_phi(phi)[3]]
        if seed is None:
            seed = target
        # Exact closure dominates tracking. A tiny regularizer selects a nearby
        # solution in the redundant mechanism without fighting the loop in sim.
        result = least_squares(lambda q: np.r_[self.residual(q), 1e-5 * (q - target)], seed,
                               bounds=(self.lower, self.upper),
                               ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=500)
        residual = self.residual(result.x)
        if np.linalg.norm(residual) > 1e-5:
            raise ValueError(f"No compatible loop solution at Phi={phi:.4f}: {residual}")
        return result.x

    def trajectory(self, samples=301):
        phis = np.linspace(0, 3, samples)
        # Continue outwards from the exact reference configuration, avoiding
        # branch jumps near the flat, nearly singular folded configurations.
        q = np.empty((samples, 8))
        middle = int(np.argmin(abs(phis - 1.5)))
        q[middle] = self.solve(phis[middle], np.zeros(8))
        for indices in (range(middle + 1, samples), range(middle - 1, -1, -1)):
            previous = q[middle]
            for i in indices:
                q[i] = self.solve(phis[i], previous)
                previous = q[i]
        return phis, q


def tangent_projector(model, data, dofs):
    """Minimum-norm motor allocation excluding weld preload directions.

    The weld constrains six relative velocities. Torque along its Jacobian's
    row space only loads the loop internally. This symmetric projector keeps
    torques in the attainable motion space; it does not equalize all torques.
    """
    rows = np.flatnonzero((data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)) &
                          (data.efc_id == model.equality("closure_weld").id))
    if len(rows) != 6:
        raise ValueError("Expected six active closure weld equations")
    if mujoco.mj_isSparse(model):
        # Use a dense Jacobian in this small (26-DoF) model.
        raise ValueError("Closed-loop allocation requires option jacobian='dense'")
    jacobian = data.efc_J.reshape(data.nefc, model.nv)[rows][:, dofs]
    _, singular, vt = np.linalg.svd(jacobian, full_matrices=True)
    rank = int(np.count_nonzero(singular > singular[0] * 1e-8))
    tangent = vt[rank:].T
    return tangent @ tangent.T
