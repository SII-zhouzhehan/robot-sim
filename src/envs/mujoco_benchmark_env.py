"""Independent MuJoCo versions of the existing A/B/C benchmark scenes."""
from pathlib import Path

import mujoco
import numpy as np

from .mujoco_model import DEFAULT_MODEL, DEFAULT_URDF, build_model
from .mujoco_robot_env import MujocoRobotEnv
from .task_manager import BenchmarkTask


class MujocoBenchmarkEnv:
    def __init__(self, model_path=None, gui=False, scene_id="A", add_boundaries=True,
                 control_steps=12, seed=42):
        source_urdf = DEFAULT_URDF
        if model_path is not None and Path(model_path).suffix.lower() == ".urdf":
            # Legacy callers passed the private URDF as the first positional
            # argument.  Build the MuJoCo model in its canonical location.
            source_urdf, model_path = Path(model_path).resolve(), None
        if model_path is None:
            self.model_path = (DEFAULT_MODEL if source_urdf.resolve() == DEFAULT_URDF.resolve()
                               else source_urdf.parent / "mujoco" / f"{source_urdf.stem}.xml").resolve()
        else:
            self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            build_model(source_urdf, self.model_path)
        self.gui, self.scene_id, self.add_boundaries = gui, scene_id, add_boundaries
        self.control_steps, self.seed = control_steps, seed
        self.core_env = None
        self._build_scene(seed)

    def _build_scene(self, seed):
        if self.core_env is not None:
            self.core_env.close()
        # The legacy task generator uses NumPy's global RNG; isolate its state.
        rng_state = np.random.get_state()
        try:
            np.random.seed(seed)
            self.task = BenchmarkTask(self.scene_id, dt=0.05)
        finally:
            np.random.set_state(rng_state)
        if self.scene_id == "B":
            # The legacy cubic spline can overshoot the +/-3 m physical walls
            # despite every waypoint being inside them. Preserve the waypoints
            # with a shape-preserving curve in this independent backend.
            from scipy.interpolate import PchipInterpolator
            nodes = np.vstack([[0.0, 0.0], self.task.waypoints])
            xs = self.task.global_traj[:, 0]
            ys = PchipInterpolator(nodes[:, 0], nodes[:, 1])(xs)
            yaw = np.arctan2(np.gradient(ys), np.gradient(xs))
            self.task.global_traj = np.column_stack([xs, ys, yaw])
        self.obstacles = [dict(o) for o in self.task.boss_obstacles]
        if self.task.door_config is not None:
            door = self.task.door_config
            self.obstacles = [{"type": "narrow_gate", "x": door["x"], "y": door.get("y_center", 0),
                               "width": door["width"], "thickness": door["thickness"]}]
        spec = mujoco.MjSpec.from_file(str(self.model_path))
        obstacle_names = []

        def box(name, position, halfsize, color):
            spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=position, size=halfsize,
                                    rgba=color, contype=1, conaffinity=2, friction=[1, 0.005, 0.0001])
            obstacle_names.append(name)

        for i, obstacle in enumerate(self.obstacles):
            x, y, kind = obstacle["x"], obstacle.get("y", 0), obstacle["type"]
            prefix = f"obstacle_{i}"
            if kind == "narrow_gate":
                for sign, side in ((-1, "left"), (1, "right")):
                    box(f"{prefix}_{side}", [x, y + sign * (2 + obstacle["width"] / 2), 0.5],
                        [obstacle.get("thickness", 0.2) / 2, 2, 0.5], [0.55, 0.58, 0.65, 1])
            elif kind == "low_platform":
                box(prefix, [x, y, obstacle["height"] / 2],
                    [0.25, obstacle["width"] / 2, obstacle["height"] / 2], [0.9, 0.65, 0.2, 1])
            elif kind == "bridge":
                box(f"{prefix}_roof", [x, y, obstacle["height"] + 0.1], [0.2, 1.5, 0.1], [0.2, 0.7, 0.7, 1])
                for sign, side in ((-1, "left"), (1, "right")):
                    box(f"{prefix}_{side}", [x, y + sign * (obstacle["width"] / 2 + 0.6), obstacle["height"] / 2],
                        [0.2, 0.6, obstacle["height"] / 2], [0.2, 0.7, 0.7, 1])
            elif kind == "cylinders":
                for sign, side in ((-1, "left"), (1, "right")):
                    name = f"{prefix}_{side}"
                    spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                            pos=[x, y + sign * 0.7, 0.3], size=[obstacle["radius"], 0.3, 0],
                                            rgba=[0.85, 0.2, 0.2, 1], contype=1, conaffinity=2)
                    obstacle_names.append(name)
        if self.add_boundaries:
            for sign, side in ((-1, "left"), (1, "right")):
                box(f"boundary_{side}", [13, sign * 3, 1], [15, 0.1, 1], [0.6, 0.7, 0.85, 0.12])
        for i, waypoint in enumerate(getattr(self.task, "waypoints", [])):
            spec.worldbody.add_geom(name=f"waypoint_{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                    pos=[*waypoint, 0.3], size=[0.08, 0, 0], rgba=[0.2, 0.8, 0.4, 0.5],
                                    contype=0, conaffinity=0)
        self.core_env = MujocoRobotEnv(self.model_path, gui=self.gui, control_steps=self.control_steps, spec=spec)
        self.obstacle_geom_ids = [mujoco.mj_name2id(self.core_env.model, mujoco.mjtObj.mjOBJ_GEOM, n)
                                  for n in obstacle_names]
        self.walls = self.obstacle_geom_ids

    def reset(self, *, seed=None):
        if seed is not None:
            self.seed = seed
            self._build_scene(seed)
        self.core_env.reset(pose=self.task.start_pose)
        return self.get_obs()

    def get_obs(self):
        pose = self.core_env.get_robot_pose()
        candidates = []
        for obstacle in self.obstacles:
            half = {"narrow_gate": obstacle.get("thickness", 0.2) / 2, "low_platform": 0.25,
                    "bridge": 0.2, "cylinders": obstacle.get("radius", 0.2)}[obstacle["type"]]
            if pose[0] <= obstacle["x"] + half + 1.0:
                candidates.append((obstacle["x"] - half - pose[0], obstacle["type"]))
        distance, kind = min(candidates, key=lambda item: abs(item[0])) if candidates else (999.0, "none")
        return {"pose": pose, "closest_dist": distance, "closest_type": kind}

    def get_info(self):
        info = self.task.evaluate_state(self.core_env.get_robot_pose())
        info["success"] = bool(info["success"])
        info["dist_to_goal"] = float(info["dist_to_goal"])
        info.update(self.core_env.get_contact_summary(self.obstacle_geom_ids))
        info["engine"] = "mujoco"
        return info

    def step(self, action):
        self.core_env.step(action)
        info = self.get_info()
        return self.get_obs(), info["success"], info

    def close(self):
        self.core_env.close()
