"""Build TARS grounded MJCF with explicit 8-caster 6 mm spring suspension.

Each of the 8 casters groups the caster ball body and its directly connected
sliding carrier body into a single rigid moving assembly that translates
upward (+Z in the steering housing frame, rotating with the swerve module)
on a passive spring-loaded slide joint with range [0, 0.006] m.
At q = 0 (fully extended lower stop), every caster ball bottom is flush
(z = 0.000 m) with every tire bottom.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull


ROOT = Path(__file__).resolve().parents[2]
GROUNDED_DIR = ROOT / "data/my_robot/mujoco/tars_mjcf_grounded"
BASE_ROBOT = ROOT / "experiments/grounded_support_navigation/robot_before.xml"
BASE_SCENE = ROOT / "experiments/grounded_support_navigation/scene_before.xml"
RAW_ROBOT = ROOT / "experiments/grounded_support_fix/original/robot.xml"

# Verified CAD mapping: (name, module_idx, steering_body, ball_part, carrier_part)
CASTER_SPECS = (
    ("caster_w1a", 1, "136", "147", "143"),
    ("caster_w1b", 1, "136", "152", "156"),
    ("caster_w2a", 2, "082", "098", "102"),
    ("caster_w2b", 2, "082", "093", "089"),
    ("caster_w3a", 3, "055", "066", "062"),
    ("caster_w3b", 3, "055", "071", "075"),
    ("caster_w4a", 4, "109", "125", "129"),
    ("caster_w4b", 4, "109", "120", "116"),
)

# Verified CAD wheel mapping: (wheel_idx, wheel_part, tire_pair_name)
TIRE_SPECS = (
    (1, "140", "rubber_floor_w1"),
    (2, "086", "rubber_floor_w2"),
    (3, "059", "rubber_floor_w3"),
    (4, "113", "rubber_floor_w4"),
)

PIVOT_JOINTS = ("dof_9", "dof_12", "dof_15", "dof_18")
PART_PREFIX = "link_insert_cd6ae4a9bfcf_part_"


@dataclass
class SuspensionConfig:
    # Winning calibrated parameters (F5_crisp_k2400_ts032):
    # Spring law: F_down(x) = preload_n + stiffness * x, for x in [0, travel_m]
    # Two parallel 5 mm OD springs -> 1200 N/m each -> 2400 N/m combined (F_max = 14.6 N at 6 mm)
    stiffness: float = 2400.0        # N/m (total of 2 parallel springs per caster)
    preload_n: float = 0.20          # N downward preload at q = 0 (flush with tire)
    damping: float = 28.0            # N*s/m viscous damping along slide (zeta ~ 1.11)
    slide_frictionloss: float = 0.08 # N guide-rod Coulomb friction
    slide_armature: float = 0.005    # kg equivalent inertia for clean conditioning
    travel_m: float = 0.006          # 6 mm max upward compression
    stop_solref: str = "0.0025 1"    # crisp hard-stop at 0 and 6 mm
    stop_solimp: str = "0.95 0.99 0.0005 0.5 2"

    # Swerve passive pivot & tire/ball contact
    pivot_frictionloss: float = 0.05 # N*m low-friction passive pivot bearing
    pivot_damping: float = 0.06      # N*m*s/rad passive pivot viscous damping
    tire_solref: str = "0.032 1"     # realistic rubber tire compliance (~1.2-1.7 mm peak)
    tire_solimp: str = "0.9 0.95 0.001 0.5 2"
    tire_friction: str = "0.8 0.8 0.005 0.0001 0.0001"
    ball_solref: str = "0.006 1"     # crisp steel transfer ball contact
    ball_solimp: str = "0.92 0.97 0.0005 0.5 2"
    ball_friction: str = "0.02 0.02 0.005 0.0001 0.0001"
    housing_friction: str = "0.35 0.35 0.005 0.0001 0.0001"

    # Geometry fidelity
    smooth_tires: bool = True        # exact R=28mm, half_w=17mm cylinder collision
    smooth_balls: bool = True        # exact R=9.525mm sphere + socket housing hull
    integrator: str = "implicitfast" # stable implicit damping/springs at 2 ms
    timestep: float = 0.002          # s (500 Hz matching all controllers)

    def to_dict(self):
        return asdict(self)


def _fit_caster_ball(stl_path: Path):
    """Fit exact sphere to the transfer ball tip and extract upper housing hull."""
    raw = stl_path.read_bytes()
    nfaces = struct.unpack_from("<I", raw, 80)[0]
    tris = np.frombuffer(
        raw, offset=84, count=nfaces,
        dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")],
    )
    verts = np.unique(tris["v"].reshape(-1, 3), axis=0).astype(float)
    tip = verts[verts[:, 2] < verts[:, 2].min() + 3.0]
    w = tip - tip.mean(0)
    center = tip.mean(0) + np.linalg.lstsq(
        np.c_[2 * w, np.ones(len(w))], (w * w).sum(1), rcond=None
    )[0][:3]
    radius = float(np.linalg.norm(tip - center, axis=1).mean())
    resid = np.abs(np.linalg.norm(verts - center, axis=1) - radius)
    housing = verts[resid > 0.05]
    hull = housing[ConvexHull(housing).vertices]
    return center * 0.001, radius * 0.001, hull * 0.001


def build_model(
    out_dir: Path,
    cfg: SuspensionConfig,
    *,
    portable_asset_paths: bool = False,
) -> tuple[Path, Path]:
    """Generate robot.xml and scene.xml in out_dir according to cfg."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    robot_tree = ET.parse(BASE_ROBOT)
    robot_root = robot_tree.getroot()
    scene_tree = ET.parse(BASE_SCENE)
    scene_root = scene_tree.getroot()
    raw_root = ET.parse(RAW_ROBOT).getroot()

    # 1. Option: integrator & timestep
    for root in (robot_root, scene_root):
        opt = root.find("option")
        if opt is None:
            opt = ET.Element("option")
            root.insert(0, opt)
        opt.set("timestep", f"{cfg.timestep:.6g}")
        opt.set("integrator", cfg.integrator)

    # 2. Mesh file paths: either relative to GROUNDED_DIR (portable) or absolute
    assets_el = robot_root.find("asset")
    mesh_by_name = {}
    for mesh in assets_el.findall("mesh[@file]"):
        fname = Path(mesh.get("file")).name
        abs_file = GROUNDED_DIR / "assets" / fname
        mesh.set("file", f"assets/{fname}" if portable_asset_paths else str(abs_file))
        mesh_by_name[mesh.get("name")] = abs_file

    # 3. Undo pre-suspension wheel shims (-0.25 mm on part_140 & part_059) using exact raw CAD pos
    raw_bodies = {b.get("name"): b for b in raw_root.findall(".//body")}
    body_map = {b.get("name"): b for b in robot_root.findall(".//body")}
    parent_map = {c: p for p in robot_root.iter() for c in p}

    for _, wpart, _ in TIRE_SPECS:
        bname = f"{PART_PREFIX}{wpart}"
        body_map[bname].set("pos", raw_bodies[bname].get("pos"))
        if cfg.smooth_tires:
            gname = f"{bname}_collision_0"
            geom = body_map[bname].find(f"geom[@name='{gname}']")
            geom.attrib.pop("mesh", None)
            geom.set("type", "cylinder")
            geom.set("pos", "0 0 0")
            geom.set("quat", "1 0 0 0")
            geom.set("size", "0.028 0.017")

    # 4. Passive pivot friction & damping
    for pname in PIVOT_JOINTS:
        j = robot_root.find(f".//joint[@name='{pname}']")
        j.set("frictionloss", f"{cfg.pivot_frictionloss:.6g}")
        j.set("damping", f"{cfg.pivot_damping:.6g}")

    # 5. Build 8 explicit caster spring-suspension slide joints
    springref = -(cfg.preload_n / cfg.stiffness) if cfg.stiffness > 0 else 0.0
    housing_geoms = []

    for cname, _, _, ball_part, carr_part in CASTER_SPECS:
        ball_name = f"{PART_PREFIX}{ball_part}"
        carr_name = f"{PART_PREFIX}{carr_part}"
        ball_body = body_map[ball_name]
        carr_body = body_map[carr_name]
        raw_ball = raw_bodies[ball_name]

        # Restore exact CAD collision geom position (undo +1.25 mm pre-suspension lift)
        vis_geom = ball_body.find(f"geom[@name='{ball_name}_visual_0']")
        col_geom = ball_body.find(f"geom[@name='{ball_name}_collision_0']")
        raw_col = raw_ball.find(f"geom[@name='{ball_name}_collision_0']")
        col_geom.set("pos", raw_col.get("pos"))

        stl_path = mesh_by_name[vis_geom.get("mesh")]
        center_m, radius_m, hull_m = _fit_caster_ball(stl_path)
        gpos = np.fromstring(vis_geom.get("pos"), sep=" ")
        ball_center_in_body = gpos + center_m

        if cfg.smooth_balls:
            # Replace collision_0 with exact R=9.525mm sphere at exact CAD flush height
            col_geom.attrib.pop("mesh", None)
            col_geom.set("type", "sphere")
            col_geom.set("pos", " ".join(f"{v:.12g}" for v in ball_center_in_body))
            col_geom.attrib.pop("quat", None)
            col_geom.set("size", f"{radius_m:.12g}")

            # Add separate collision mesh for upper caster socket housing
            hmesh_name = f"{cname}_housing_mesh"
            ET.SubElement(
                assets_el, "mesh",
                name=hmesh_name,
                vertex=" ".join(f"{v:.8g}" for v in hull_m.ravel()),
            )
            hgeom_name = f"{cname}_housing_collision"
            ET.SubElement(
                ball_body, "geom",
                name=hgeom_name,
                type="mesh",
                mesh=hmesh_name,
                pos=vis_geom.get("pos"),
                quat="1 0 0 0",
                group="3",
                contype="2",
                conaffinity="1",
            )
            housing_geoms.append(hgeom_name)

        # Insert slide joint inside ball_body: +Z = upward along steering housing guide rods
        joint_el = ET.Element(
            "joint",
            name=f"{cname}_slide",
            type="slide",
            pos=" ".join(f"{v:.12g}" for v in ball_center_in_body),
            axis="0 0 1",
            limited="true",
            range=f"0 {cfg.travel_m:.6g}",
            stiffness=f"{cfg.stiffness:.6g}",
            springref=f"{springref:.12g}",
            damping=f"{cfg.damping:.6g}",
            frictionloss=f"{cfg.slide_frictionloss:.6g}",
            armature=f"{cfg.slide_armature:.6g}",
            solreflimit=cfg.stop_solref,
            solimplimit=cfg.stop_solimp,
        )
        ball_body.insert(0, joint_el)

        # Reparent directly connected carrier body inside ball_body at identical frame
        # (both originally shared the exact same parent, pos, and quat)
        old_parent = parent_map[carr_body]
        old_parent.remove(carr_body)
        carr_body.set("pos", "0 0 0")
        carr_body.set("quat", "1 0 0 0")
        ball_body.append(carr_body)

    # 6. Update scene.xml contact pairs (tires, caster balls, and optional socket housings)
    contact_el = scene_root.find("contact")
    tire_pair_names = {p for _, _, p in TIRE_SPECS}
    ball_geom_names = {f"{PART_PREFIX}{bp}_collision_0" for _, _, _, bp, _ in CASTER_SPECS}

    for pair in list(contact_el.findall("pair")):
        pname = pair.get("name", "")
        g2 = pair.get("geom2", "")
        if pname in tire_pair_names:
            pair.set("friction", cfg.tire_friction)
            pair.set("solref", cfg.tire_solref)
            pair.set("solimp", cfg.tire_solimp)
        elif g2 in ball_geom_names:
            pair.set("friction", cfg.ball_friction)
            pair.set("solref", cfg.ball_solref)
            pair.set("solimp", cfg.ball_solimp)

    for hgeom_name in housing_geoms:
        ET.SubElement(
            contact_el, "pair",
            geom1="ctm_floor",
            geom2=hgeom_name,
            condim="3",
            friction=cfg.housing_friction,
            solref="0.005 1",
        )

    robot_path = out_dir / "robot.xml"
    scene_path = out_dir / "scene.xml"
    ET.indent(robot_tree, space="  ")
    ET.indent(scene_tree, space="  ")
    robot_tree.write(robot_path, encoding="utf-8", xml_declaration=True)
    scene_tree.write(scene_path, encoding="utf-8", xml_declaration=True)
    return robot_path, scene_path


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "trial_default"
    r, s = build_model(out, SuspensionConfig())
    print("Built:", r, s)
