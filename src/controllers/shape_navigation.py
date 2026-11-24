"""Encoder-driven rigid-group geometry for a grounded robot on a mapped floor."""
import numpy as np
from scipy.spatial.transform import Rotation


class ShapeGeometry:
    """Each packed rigid group has one parent hinge, except the localized root."""

    def __init__(self, asset):
        with np.load(asset, allow_pickle=False) as data:
            self.names = tuple(data["names"].tolist())
            self.parents = data["parents"]
            self.offsets = data["offsets"]
            self.rest_rot = data["rest_rot"]
            self.axes = data["axes"]
            self.anchors = data["anchors"]
            self.reference = data["reference"]
            self.vertices = data["vertices"]
            self.group = data["group"]

    def world_vertices(self, encoders, root_xyz, root_rotation):
        """encoders maps CAD joint names to radians; rotation maps root axes to world."""
        q = np.asarray([encoders[name] for name in self.names[1:]]) - self.reference[1:]
        if not np.isfinite(q).all():
            raise ValueError("Non-finite encoder reading")
        hinge = Rotation.from_rotvec(self.axes[1:] * q[:, None]).as_matrix()
        rotations = np.empty_like(self.rest_rot)
        positions = np.empty_like(self.offsets)
        rotations[0] = root_rotation
        positions[0] = root_xyz
        for i, rot in enumerate(hinge, 1):
            parent = self.parents[i]
            rest = rotations[parent] @ self.rest_rot[i]
            rotations[i] = rest @ rot
            positions[i] = (positions[parent] + rotations[parent] @ self.offsets[i]
                            + rest @ (self.anchors[i] - rot @ self.anchors[i]))
        return (np.einsum("nij,nj->ni", rotations[self.group], self.vertices)
                + positions[self.group])

    def lateral_bounds(self, encoders, root_xyz, root_rotation, wall_normal):
        """Signed support coordinates along the world's unit horizontal wall normal."""
        normal = np.asarray(wall_normal, dtype=float)
        if normal.shape != (2,) or not np.isclose(np.linalg.norm(normal), 1.):
            raise ValueError("Expected a unit XY wall normal")
        projected = self.world_vertices(encoders, root_xyz, root_rotation)[:, :2] @ normal
        return np.array([np.min(projected), np.max(projected)])


def corridor_clearance(bounds, walls, margin=0.):
    """Worst signed clearance over supplied edge samples and both parallel walls."""
    samples = np.asarray(bounds, dtype=float)
    left, right = walls
    if (samples.ndim != 2 or samples.shape[1] != 2 or len(samples) == 0
            or not np.isfinite(samples).all() or np.any(samples[:, 0] > samples[:, 1])
            or not np.isfinite([left, right, margin]).all() or right <= left or margin < 0):
        raise ValueError("Invalid corridor or footprint samples")
    return float(min(np.min(samples[:, 0]) - left, right - np.max(samples[:, 1])) - margin)


def assess_phase_edge(bounds, wheel_loads, walls, *, margin_m, min_load_n):
    """Report independent geometric and support evidence for one sampled motion."""
    loads = np.asarray(wheel_loads, dtype=float)
    if (loads.shape != (len(bounds), 4) or not np.isfinite(loads).all()
            or not np.isfinite(min_load_n) or min_load_n < 0):
        raise ValueError("Expected one finite four-wheel load sample per geometry sample")
    clearance = corridor_clearance(bounds, walls, margin_m)
    missing = loads <= min_load_n
    return dict(clearance_m=clearance, intersects_walls=clearance < 0,
                four_supported_fraction=float(np.mean(~missing.any(axis=1))),
                wheel_unloaded_fraction=missing.mean(axis=0).tolist())
