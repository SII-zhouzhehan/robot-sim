"""Build a separate, portable MuJoCo model from the private SolidWorks URDF."""
from pathlib import Path
import json
import struct
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial import ConvexHull


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = PROJECT_ROOT / "data/my_robot/robot1231_2.urdf"
DEFAULT_MODEL = PROJECT_ROOT / "data/my_robot/mujoco/robot.xml"
BLOCK_NAMES = tuple(f"joint{i}" for i in range(1, 8))
STEERING_NAMES = ("joint9", "joint12", "joint15", "joint18")
WHEEL_NAMES = ("joint10", "joint13", "joint16", "joint19")
PIVOT_NAMES = ("joint8", "joint11", "joint14", "joint17")
# Provisional bearing resistance; tune against physical breakaway measurements.
DEFAULT_PIVOT_FRICTIONLOSS = 0.05  # Nm at each hinge output.
DEFAULT_PIVOT_DAMPING = 0.05  # Nm*s/rad.


def read_stl(path):
    raw = Path(path).read_bytes()
    count = struct.unpack_from("<I", raw, 80)[0] if len(raw) >= 84 else 0
    if len(raw) != 84 + 50 * count:
        raise ValueError(f"Expected binary SolidWorks STL: {path}")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    return np.frombuffer(raw, dtype=dtype, count=count, offset=84)["vertices"].copy()


def write_stl(path, triangles):
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    records = np.zeros(len(triangles), dtype=dtype)
    records["normal"], records["vertices"] = normals, triangles
    Path(path).write_bytes(b"MuJoCo derived mesh; original STL preserved".ljust(80, b" ")
                           + struct.pack("<I", len(records)) + records.tobytes())


def simplify_visual(triangles, voxel_size=0.004):
    """Vertex clustering for display only; collision geometry is built separately."""
    if len(triangles) < 3000:
        return triangles
    vertices = triangles.reshape(-1, 3).astype(np.float64)
    _, inverse = np.unique(np.floor(vertices / voxel_size).astype(np.int64), axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    averaged = np.column_stack([np.bincount(inverse, weights=vertices[:, i]) / counts for i in range(3)])
    faces = inverse.reshape(-1, 3)
    valid = (faces[:, 0] != faces[:, 1]) & (faces[:, 0] != faces[:, 2]) & (faces[:, 1] != faces[:, 2])
    faces = faces[valid]
    _, first = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    result = averaged[faces[np.sort(first)]]
    area = np.linalg.norm(np.cross(result[:, 1] - result[:, 0], result[:, 2] - result[:, 0]), axis=1)
    return result[area > 1e-12].astype(np.float32)


def convex_collision(triangles):
    vertices = np.unique(triangles.reshape(-1, 3), axis=0)
    hull = ConvexHull(vertices)
    result = vertices[hull.simplices].copy()
    normals = np.cross(result[:, 1] - result[:, 0], result[:, 2] - result[:, 0])
    reverse = np.einsum("ij,ij->i", normals, hull.equations[:, :3]) < 0
    result[reverse] = result[reverse][:, [0, 2, 1]]
    return result


def build_model(urdf_path=DEFAULT_URDF, output_path=DEFAULT_MODEL, voxel_size=0.004, *,
                pivot_frictionloss=DEFAULT_PIVOT_FRICTIONLOSS, pivot_damping=DEFAULT_PIVOT_DAMPING):
    if any(not np.isfinite(v) or v < 0 for v in (pivot_frictionloss, pivot_damping)):
        raise ValueError("Pivot resistance must be finite and nonnegative")
    urdf_path, output_path = Path(urdf_path).resolve(), Path(output_path).resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"Private robot assets are missing: {urdf_path}")
    if output_path == urdf_path or output_path.parent == urdf_path.parent / "meshes":
        raise ValueError("MuJoCo output must be separate from the original robot assets")
    source = ET.parse(urdf_path).getroot()
    extension = source.find("mujoco")
    if extension is None:
        extension = ET.SubElement(source, "mujoco")
    compiler = extension.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(extension, "compiler")
    compiler.attrib.update(meshdir=str(urdf_path.parent), discardvisual="false", fusestatic="false")
    spec = mujoco.MjSpec.from_string(ET.tostring(source, encoding="unicode"))
    spec.compile()
    root = ET.fromstring(spec.to_xml())
    root.set("model", "morphing_robot_mujoco")
    root.find("compiler").attrib.update(meshdir="meshes", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep=str(1 / 240), gravity="0 0 -9.8",
                  integrator="implicitfast", cone="elliptic", iterations="50")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="720")
    ET.SubElement(visual, "quality", shadowsize="1024")
    assets = root.find("asset")
    assets.clear()
    texture = ET.SubElement(assets, "texture", name="floor_texture", type="2d", builtin="checker",
                            rgb1="0.22 0.25 0.28", rgb2="0.32 0.35 0.38", width="512", height="512")
    ET.SubElement(assets, "material", name="floor_material", texture=texture.get("name"),
                  texrepeat="20 20", texuniform="true", reflectance="0")
    ET.SubElement(assets, "texture", name="sky", type="skybox", builtin="gradient",
                  rgb1="0.25 0.32 0.42", rgb2="0.07 0.1 0.15", width="512", height="3072")
    mesh_dir = output_path.parent / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for link in source.findall("link"):
        name = link.get("name")
        mesh = link.find("visual/geometry/mesh")
        if mesh is None:
            continue
        triangles = read_stl(urdf_path.parent / mesh.get("filename"))
        lightweight = simplify_visual(triangles, voxel_size)
        write_stl(mesh_dir / f"{name}_visual.stl", lightweight)
        ET.SubElement(assets, "mesh", name=f"{name}_visual", file=f"{name}_visual.stl", inertia="shell")
        if not name.startswith("wheel"):
            collision = convex_collision(triangles)
            write_stl(mesh_dir / f"{name}_collision.stl", collision)
            ET.SubElement(assets, "mesh", name=f"{name}_collision", file=f"{name}_collision.stl")
        report[name] = {"original_faces": len(triangles), "visual_faces": len(lightweight)}

    world = root.find("worldbody")
    base = world.find("body[@name='base']")
    base.set("pos", "0 0 0.35")
    ET.SubElement(base, "freejoint", name="floating_base")
    ET.SubElement(world, "light", name="key_light", pos="0 -2 4", dir="0 0 -1", directional="true")
    ET.SubElement(world, "geom", name="ground", type="plane", size="30 30 0.1", material="floor_material",
                  contype="1", conaffinity="2", friction="1 0.005 0.0001", group="0")
    collision_bodies = {"base", *(f"block{i}" for i in range(2, 9))}
    for body in base.iter("body"):
        name = body.get("name")
        for geom in list(body.findall("geom")):
            body.remove(geom)
        color = "0.75 0.8 0.85 1" if name.startswith("block") else "0.35 0.42 0.5 1"
        if name.startswith("wheel"):
            color = "0.12 0.13 0.14 1"
        ET.SubElement(body, "geom", name=f"{name}_visual", type="mesh", mesh=f"{name}_visual",
                      group="1", contype="0", conaffinity="0", density="0", rgba=color)
        attrs = dict(name=f"{name}_collision", group="2", density="0", rgba="0.9 0.4 0.2 0.3",
                     contype="6" if name in collision_bodies else "2",
                     conaffinity="5" if name in collision_bodies else "1",
                     friction="1 0.005 0.0001", solref="0.015 1", solimp="0.9 0.95 0.002")
        if name.startswith("wheel"):
            # Wheel meshes are axial in local Z; use a smooth contact cylinder.
            z_center = 0.008 if name in ("wheel1", "wheel2") else -0.008
            attrs.update(type="cylinder", size="0.0255 0.020", pos=f"0 0 {z_center}", condim="4")
        else:
            attrs.update(type="mesh", mesh=f"{name}_collision")
        ET.SubElement(body, "geom", **attrs)

    contact = ET.SubElement(root, "contact")
    for i in range(1, 5):
        # The CAD steering assembly includes auxiliary ball casters but exports
        # them as one rigid link. A low tangential resistance is a rolling proxy
        # for these supports, rather than treating them as rubber skids.
        ET.SubElement(contact, "pair", geom1="ground", geom2=f"steering{i}_collision", condim="3",
                      friction="0.02 0.02 0.001 0.0001 0.0001", solref="0.015 1")
    disabled = [(2, 6), (2, 7), (3, 6), (3, 7), (4, 1), (4, 5), (8, 1), (8, 5), (4, 8)]
    for left, right in disabled:
        ET.SubElement(contact, "exclude", body1=f"block{left}", body2=f"block{right}")
    for other in ("block4", "block8"):
        ET.SubElement(contact, "exclude", body1="base", body2=other)

    actuators = ET.SubElement(root, "actuator")
    joints = {j.get("name"): j for j in root.iter("joint")}
    for names, force, kp, kv in ((BLOCK_NAMES, 10, 35, 2), (STEERING_NAMES, 2, 4, 0.15), (PIVOT_NAMES, 5, 40, 1)):
        for name in names:
            joints[name].set("actuatorfrcrange", f"{-force} {force}")
            joints[name].set("damping", "0.01")
            if name in PIVOT_NAMES:
                joints[name].set("frictionloss", str(pivot_frictionloss))
                joints[name].set("damping", str(pivot_damping))
            joints[name].set("armature", "0.0001")
            ET.SubElement(actuators, "position", name=f"{name}_position", joint=name,
                          kp=str(kp), kv=str(kv), forcerange=f"{-force} {force}")
    for name in WHEEL_NAMES:
        joints[name].set("actuatorfrcrange", "-0.35 0.35")
        joints[name].set("damping", "0.0001")
        joints[name].set("armature", "0.00001")
        ET.SubElement(actuators, "velocity", name=f"{name}_velocity", joint=name, kv="0.025",
                      ctrlrange="-16.75516082 16.75516082", forcerange="-0.35 0.35")
    ET.indent(root, space="  ")
    output_path.write_text(ET.tostring(root, encoding="unicode") + "\n", encoding="utf-8")
    # Compile the final portable file, including derived assets, before reporting success.
    model = mujoco.MjModel.from_xml_path(str(output_path))
    report["total"] = {k: sum(v[k] for v in report.values()) for k in ("original_faces", "visual_faces")}
    report["model"] = {"nq": model.nq, "nv": model.nv, "actuators": model.nu, "timestep": model.opt.timestep}
    (output_path.parent / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    return output_path, report
