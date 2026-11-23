"""Baseline measurements on the imported TARS model. Run from any directory."""
import json
from datetime import datetime
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from envs.tars_grounded import MORPH_NAMES, SCENE, TarsGroundedEnv
from exploration.active_motion import ActiveMotionLearner
from exploration.closed_loop_probe import closed_physical_edges


def floor_load(env):
    m, d = env.model, env.data
    floor = m.geom("ctm_floor").id
    wheels = set(m.body_weldid[m.jnt_bodyid[env.wheel_ids]])
    normal = np.zeros(2)
    wrench = np.zeros(6)
    for idx, contact in enumerate(d.contact):
        a, b = contact.geom
        if floor not in (a, b):
            continue
        other = b if a == floor else a
        mujoco.mj_contactForce(m, d, idx, wrench)
        normal[int(m.body_weldid[m.geom_bodyid[other]] in wheels)] += wrench[0]
    return normal


def measure(env, twist):
    env.reset()
    start = env.get_robot_pose()
    loads = np.zeros(2)
    for _ in range(round(2 / env.control_dt)):
        env.drive(*twist)
        loads += floor_load(env)
    return {"command": twist, "delta_xy_yaw": (env.get_robot_pose() - start).tolist(),
            "rolling_wheel_load_fraction": float(loads[1] / loads.sum())}


def pose_trial(env, displacement):
    """Independent 16 s trial with capped world-frame pose feedback."""
    env.reset()
    goal = env.get_robot_pose() + displacement
    consecutive = 0
    status = "timeout"
    for _ in range(round(16 / env.control_dt)):
        error = goal - env.get_robot_pose()
        error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))
        inside = np.linalg.norm(error[:2]) < .005 and abs(error[2]) < .02
        consecutive = consecutive + 1 if inside else 0
        if consecutive >= 10:
            status = "reached"
            break
        twist = np.clip(error * [4, 4, 3], [-.10, -.10, -.35], [.10, .10, .35])
        try:
            env.drive(*twist)
        except ValueError as exc:
            if str(exc) != "Planar drive requires upright steering modules":
                raise
            status = "steering_axis_tilt"
            break
    return {"displacement": displacement, "status": status,
            "final_error_xy_yaw": (goal - env.get_robot_pose()).tolist(),
            "elapsed_s": round(env.data.time - .6, 3)}


def local_response(env):
    edges = closed_physical_edges(env.model, env.block_ids)
    learner = ActiveMotionLearner(MORPH_NAMES, edges)

    def probe(command):
        env.reset()
        for _ in range(round(.3 / env.control_dt)):
            env.step({"blocks": command})
        return env.observe()["blocks"]

    drift = probe(np.zeros(8))
    records = []
    for x in np.concatenate([.06 * np.eye(8), -.06 * np.eye(8)]):
        y = probe(x) - drift
        learner.observe(x, y)
        records.append({"command_rad": x.tolist(), "response_rad": y.tolist()})
    errors, baseline = [], []
    for x in np.random.default_rng(42).uniform(-.035, .035, (8, 8)):
        y = probe(x) - drift
        errors.append(y - learner.predict(x))
        baseline.append(y)
    return {"edges_zero_based": edges, "rank_at_threshold_0.08": learner.observed_rank,
            "records": records, "singular_values": learner.singular_values.tolist(),
            "heldout_rmse_rad": float(np.sqrt(np.mean(np.square(errors)))),
            "zero_prediction_rmse_rad": float(np.sqrt(np.mean(np.square(baseline))))}


def main():
    env = TarsGroundedEnv()
    motions = {"rest": (0., 0., 0.), "x+": (.06, 0., 0.), "x-": (-.06, 0., 0.),
               "y+": (0., .06, 0.), "y-": (0., -.06, 0.), "yaw+": (0., 0., .25),
               "yaw-": (0., 0., -.25)}
    result = {"scene": str(SCENE.relative_to(ROOT)), "control_dt": env.control_dt,
              "mujoco": mujoco.__version__,
              "drive": {name: measure(env, cmd) for name, cmd in motions.items()}}
    env.reset()
    steering = np.array([.25, -.25, .25, -.25])
    for _ in range(round(.5 / env.control_dt)):
        env.step({"steering": steering})
    result["steer_target_rad"] = steering.tolist()
    result["steer_actual_rad"] = env.observe()["steering"].tolist()
    result["pose_trials"] = [pose_trial(env, goal) for goal in (
        [.05, 0, 0], [-.05, 0, 0], [0, .05, 0], [0, -.05, 0], [0, 0, .18], [0, 0, -.18])]
    result["local_response"] = local_response(env)
    out = ROOT / "experiments" / ("tars_grounded_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True)
    (out / "baseline.json").write_text(json.dumps(result, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
