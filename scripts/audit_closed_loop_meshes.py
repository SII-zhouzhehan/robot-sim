"""Check original CAD triangle surfaces independently of MuJoCo exclusions.

Optional dependency: python-fcl. Use --deps-dir for a temporary installation.
Triangle intersection is a geometry warning, not a force/volume measurement.
The user intentionally excluded certain CAD contacts because of approximation
error. This script does not establish physical interference or enable pairs.
"""
import argparse
from datetime import datetime
import itertools
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import mujoco
import numpy as np
from envs.mujoco_closed_loop import ClosedLoopKinematics, DEFAULT_CLOSED_MODEL
from controllers.mujoco_control import angles_from_phi
from envs.mujoco_model import DEFAULT_URDF, read_stl


def crosses_both_planes(first, second, tolerance=1e-5):
    """Reject coplanar/touch-only FCL contacts (10 micrometre threshold).

    FCL has already established triangle intersection. Requiring each
    triangle to straddle the other's plane distinguishes transverse surface
    crossings from coincident CAD faces; this is not a penetration depth.
    """
    for a, b in ((first, second), (second, first)):
        normal = np.cross(a[1] - a[0], a[2] - a[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-16:
            return False
        signed = (b - a[0]) @ (normal / norm)
        if signed.min() >= -tolerance or signed.max() <= tolerance:
            return False
    return True


def render_crossings(model, data, ik, trajectory, rows, output):
    """Illustrate the intentionally excluded pair using simplified visuals.

    The close-up camera is centered on the original-triangle crossing region.
    This picture is a mesh diagnostic, not evidence of physical interference.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Patch
    font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    if font.exists():
        plt.rcParams["font.family"] = FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    phi = 2.85
    row = min((r for r in rows if r["mode"] == "closed_projection"), key=lambda r: abs(r["phi"] - phi))
    pair = next(p for p in row["intersections"] if p["pair"] == ["block2", "block6"])
    points = np.array(pair["transverse_points_world_m"])
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 640
    renderer = mujoco.Renderer(model, width=960, height=640)
    option = mujoco.MjvOption()
    option.geomgroup[2] = 0
    colors = {model.body("block2").id: [.06, .52, .82, 1],
              model.body("block6").id: [1, .56, .06, 1]}
    visual = np.flatnonzero(model.geom_group == 1)
    rgba = model.geom_rgba.copy()
    materials = model.geom_matid.copy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4), layout="constrained")
    try:
        data.qpos[:] = model.qpos0
        data.qpos[ik.qadr] = [np.interp(row["phi"], trajectory["phi"], c) for c in trajectory["position_rad"].T]
        mujoco.mj_forward(model, data)
        center = np.mean([data.xpos[model.body(f"block{i}").id] for i in range(1, 9)], axis=0)
        for close, ax in zip((False, True), axes):
            model.geom_rgba[:] = rgba
            model.geom_matid[visual] = -1
            for g in visual:
                body = int(model.geom_bodyid[g])
                model.geom_rgba[g] = colors.get(body, [.66, .70, .74, 0 if close else .25])
                if close and body in colors:
                    model.geom_rgba[g, 3] = .4
            option.geomgroup[0] = not close
            camera = mujoco.MjvCamera()
            camera.distance, camera.azimuth, camera.elevation = (.36 if close else 1.05), 110, -40
            camera.lookat[:] = points.mean(axis=0) if close else center
            renderer.update_scene(data, camera=camera, scene_option=option)
            ax.imshow(renderer.render())
            ax.axis("off")
        axes[0].set_title(f"推算闭环轨迹 · Φ={row['phi']:.2f}", fontsize=17)
        axes[1].set_title("已手动排除接触的网格交叉 · 不代表实物干涉", fontsize=17)
        fig.legend(handles=[Patch(color=colors[b], label=model.body(b).name) for b in colors],
                   loc="outside lower center", ncol=2, fontsize=13, frameon=False)
        fig.savefig(output, dpi=220)
    finally:
        model.geom_rgba[:] = rgba
        model.geom_matid[:] = materials
        renderer.close()
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deps-dir", type=Path)
    parser.add_argument("--samples", type=int, default=61)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / f"closed_loop_contact_audit_{datetime.now():%Y%m%d}")
    parser.add_argument("--trajectory", type=Path, default=ROOT / "experiments/closed_loop_20261020/compatible_trajectory.npz")
    parser.add_argument("--states", type=Path, help="Also check the largest dynamic torque sample and Phi near 2.85")
    parser.add_argument("--render", action="store_true", help="Render a block2/block6 interference illustration")
    args = parser.parse_args()
    if args.deps_dir:
        sys.path.insert(0, str(args.deps_dir.resolve()))
    import fcl
    source = ET.parse(DEFAULT_URDF).getroot()
    objects, meshes, triangle_counts = {}, {}, {}
    for i in range(1, 9):
        name = f"block{i}"
        link = source.find(f"link[@name='{name}']")
        origin = link.find("visual/origin")
        if origin is not None and any(float(v) != 0 for attribute in ("xyz", "rpy")
                                      for v in origin.get(attribute, "0 0 0").split()):
            raise ValueError("This audit requires zero visual origins, as in the current CAD")
        triangles = read_stl(DEFAULT_URDF.parent / link.find("visual/geometry/mesh").get("filename")).astype(float)
        vertices, index = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
        bvh = fcl.BVHModel()
        bvh.beginModel(len(vertices), len(triangles))
        bvh.addSubModel(vertices, index.reshape(-1, 3))
        bvh.endModel()
        objects[i] = fcl.CollisionObject(bvh)
        meshes[i] = triangles
        triangle_counts[name] = len(triangles)
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_CLOSED_MODEL))
    excluded = {frozenset((e.get("body1"), e.get("body2")))
                for e in ET.parse(DEFAULT_CLOSED_MODEL).getroot().findall("contact/exclude")}
    data = mujoco.MjData(model)
    ik = ClosedLoopKinematics(model)
    trajectory = np.load(args.trajectory)
    # Adjacent joint mates may share CAD actuator/bearing interfaces and are
    # not validated here; all other pairs are tested despite exclusions.
    adjacent = {(1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (7, 8), (4, 8)}
    poses = []
    for phi in np.linspace(0, 3, args.samples):
        for mode in ("original_formula", "closed_projection"):
            qpos = model.qpos0.copy()
            qpos[ik.qadr] = (np.r_[angles_from_phi(phi), 0] if mode == "original_formula" else
                            np.array([np.interp(phi, trajectory["phi"], c) for c in trajectory["position_rad"].T]))
            poses.append((float(phi), mode, qpos))
    if args.states:
        states = np.load(args.states)
        peak = np.unravel_index(np.argmax(abs(states["torque_nm"])), states["torque_nm"].shape)[0]
        near = int(np.argmin(abs(states["phi"] - 2.85)))
        for label, index in (("dynamic_peak", peak), ("dynamic_phi_2.85", near)):
            poses.append((float(states["phi"][index]), label, states["full_qpos"][index]))
    rows = []
    for phi, mode, qpos in poses:
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        for i, obj in objects.items():
            b = model.body(f"block{i}").id
            obj.setTransform(fcl.Transform(data.xmat[b].reshape(3, 3), data.xpos[b]))
        intersections = []
        for i, j in itertools.combinations(range(1, 9), 2):
            if (i, j) in adjacent:
                continue
            result = fcl.CollisionResult()
            fcl.collide(objects[i], objects[j], fcl.CollisionRequest(num_max_contacts=100, enable_contact=True), result)
            if result.is_collision:
                world_triangles = []
                for number in (i, j):
                    body = model.body(f"block{number}").id
                    world_triangles.append(meshes[number] @ data.xmat[body].reshape(3, 3).T + data.xpos[body])
                strict = [c for c in result.contacts if crosses_both_planes(
                    world_triangles[0][c.b1], world_triangles[1][c.b2])]
                intersections.append({"pair": [f"block{i}", f"block{j}"],
                                      "excluded_in_saved_model": frozenset((f"block{i}", f"block{j}")) in excluded,
                                      "contact_count_capped_100": len(result.contacts),
                                      "points_world_m": [c.pos.tolist() for c in result.contacts[:8]],
                                      "transverse_crossing_count_capped_100": len(strict),
                                      "transverse_points_world_m": [c.pos.tolist() for c in strict[:20]],
                                      "crossing_face_indices": [[int(c.b1), int(c.b2)] for c in strict[:20]]})
        rows.append({"phi": float(phi), "mode": mode, "intersections": intersections})
        if intersections:
            print(mode, f"Phi={phi:.2f}", [(r["pair"], r["transverse_crossing_count_capped_100"]) for r in intersections], flush=True)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"original_mesh_triangles": triangle_counts, "samples": args.samples,
              "user_context": "Manual collision exclusions compensate for CAD approximation. Mesh crossings are not proof of physical interference. Do not change exclusions.",
              "saved_excluded_pairs": sorted(sorted(pair) for pair in excluded),
              "skipped_adjacent_pairs": sorted(adjacent), "rows": rows,
              "transverse_crossing_tolerance_m": 1e-5,
              "dynamic_source": str(args.states) if args.states else None,
              "limitation": "Discrete triangle-surface intersections with capped contact enumeration; no continuous collision guarantee, solid containment, joint-interface validation, penetration depth, volume or load inference."}
    (out / "original_mesh_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    for mode in ("original_formula", "closed_projection"):
        selected = [r for r in rows if r["mode"] == mode]
        print(mode, "intersection poses", sum(bool(r["intersections"]) for r in selected), "/", len(selected),
              "transverse crossing poses", sum(any(i["transverse_crossing_count_capped_100"] for i in r["intersections"]) for r in selected), flush=True)
    if args.render:
        render_crossings(model, data, ik, trajectory, rows, out / "block2_block6_interference.png")


if __name__ == "__main__":
    main()
