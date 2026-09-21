"""Headless, fixed-budget traversal evaluation; private assets stay local.

Examples:
    python scripts/eval_four_obstacles.py --trials 1 --workers 4 --nominal --output experiments/four_smoke
    python scripts/eval_four_obstacles.py --trials 20 --workers 4 --output experiments/four_batch
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import csv
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
from evaluation.obstacle_protocol import TraversalCriteria, TraversalMonitor, wilson_interval

KINDS = ("narrow_gate", "low_platform", "bridge", "cylinders")
TARGET_PHI = dict(zip(KINDS, (0.0, 2.0, 3.0, 1.5)))
DT = 0.05
PROTOCOL_VERSION = "isolated-traversal-v1"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


class StableHeading:
    """Use the existing GUI evaluator's continuous geometric Phi=3 heading."""
    def __init__(self):
        self.value = None

    def update(self, core, phi, raw_yaw):
        import pybullet as p
        if self.value is None:
            self.value = raw_yaw
        if phi > 2.8:
            points = np.array([p.getLinkState(core.robotId, j)[0][:2] for j in core.pivot_ids])
            pairs = [(i, j) for i in range(4) for j in range(i+1, 4)]
            i, j = min(pairs, key=lambda pair: np.linalg.norm(points[pair[0]]-points[pair[1]]))
            vector = points[j]-points[i]
            candidate = np.arctan2(vector[0], -vector[1])
            if np.cos(candidate-self.value) < 0:
                candidate += np.pi
            alpha = .3
        else:
            candidate = raw_yaw
            alpha = np.clip(1-(phi-2)*.45, .3, 1)
        self.value += alpha * ((candidate-self.value+np.pi) % (2*np.pi)-np.pi)
        return self.value


def simulate(job):
    import pybullet as p
    from brains.scripted_brain import ScriptedBrain
    from controllers.mpc_controller import KinematicMPC
    from controllers.swerve_kinematics import SwerveDriveController
    from envs.benchmark_env import BenchmarkEnv
    from scripts.test_mpc_vs_pid_topology import ZeroYawSmoother, compute_angles_from_phi

    kind, seed = job["kind"], job["seed"]
    required_phi = job.get("platform_phi", 2.) if kind == "low_platform" else TARGET_PHI[kind]
    rng = np.random.default_rng(seed)
    perturb = {"x_m": float(rng.uniform(-.03, .03)),
               "y_m": float(rng.uniform(-.03, .03)),
               "yaw_rad": float(rng.uniform(-np.pi/60, np.pi/60)),
               "friction": float(rng.uniform(.9, 1.1))}
    if job["nominal"]:
        perturb = {"x_m": 0., "y_m": 0., "yaw_rad": 0., "friction": 1.}
    env = BenchmarkEnv(job["urdf"], gui=False, scene_id="C", add_boundaries=False)
    trace = []
    collisions = {"force_n": 0., "penetration_m": 0., "failed": False}
    contact_links = set()
    result = {"kind": kind, "seed": seed, "success": False, "reason": "timeout",
              "perturbation": perturb, "required_phi": required_phi,
              "platform_heading_deg": job.get("platform_heading_deg", 90.),
              "align_platform": job.get("align_platform", False)}
    try:
        obstacle = dict(next(o for o in env.task.boss_obstacles if o["type"] == kind))
        obstacle.update(x=2.0, y=0.0)
        env.task.boss_obstacles = [obstacle]
        env.task.goal_pose = (4., 0., 0.)
        # Preserve the course's required sideways heading for the platform.
        xs = np.arange(0., 4.001, .02)
        yaw_ref = np.deg2rad(job.get("platform_heading_deg", 90.)) * np.clip((xs-.3)/.5, 0, 1) if kind == "low_platform" else np.zeros_like(xs)
        if kind == "low_platform" and job.get("align_platform", False):
            yaw_ref[:] = np.deg2rad(job.get("platform_heading_deg", 90.))
        env.task.global_traj = np.column_stack((xs, np.zeros_like(xs), yaw_ref))
        env.reset()
        core = env.core_env
        p.setTimeStep(1/240)
        pos, orn = p.getBasePositionAndOrientation(core.robotId)
        _, new_orn = p.multiplyTransforms([0, 0, 0], p.getQuaternionFromEuler([0, 0, perturb["yaw_rad"]]), [0, 0, 0], orn)
        p.resetBasePositionAndOrientation(core.robotId, [pos[0]+perturb["x_m"], pos[1]+perturb["y_m"], pos[2]], new_orn)
        p.changeDynamics(core.planeId, -1, lateralFriction=perturb["friction"])
        for idx in core.wheel_ids:
            p.changeDynamics(core.robotId, idx, lateralFriction=perturb["friction"])

        criteria = TraversalCriteria()
        half_length = {"narrow_gate": .1, "low_platform": .25, "bridge": .2, "cylinders": .2}[kind]
        half_lane = {"narrow_gate": .21, "low_platform": .08, "bridge": .2, "cylinders": .5}[kind]
        monitor = TraversalMonitor(2-half_length, 2+half_length, 0., half_lane, criteria)
        original_hold = core.update_pivot_hold

        def hold_and_check_contacts():
            original_hold()
            # This hook executes after EVERY 240 Hz physics tick, not only at 20 Hz.
            for obstacle_id in env.walls:
                for contact in p.getContactPoints(bodyA=core.robotId, bodyB=obstacle_id):
                    force, depth = float(contact[9]), max(0., -float(contact[8]))
                    collisions["force_n"] = max(collisions["force_n"], force)
                    collisions["penetration_m"] = max(collisions["penetration_m"], depth)
                    if force > criteria.contact_force_tolerance_n or depth > criteria.penetration_tolerance_m:
                        collisions["failed"] = True
                        contact_links.add(int(contact[3]))

        core.update_pivot_hold = hold_and_check_contacts
        for _ in range(20):
            core.step({"blocks": compute_angles_from_phi(1.5), "velocity": [0]*4, "steering": [0]*4})
        brain = ScriptedBrain(dt=DT, mpc_horizon=10)
        mpc = KinematicMPC(dt=DT, N=10)
        swerve = SwerveDriveController(wheel_radius=core.WHEEL_RADIUS)
        smoother = ZeroYawSmoother(alpha=.9)
        heading = StableHeading()
        phi, last_cmd = 1.5, np.zeros(3)
        aligned = kind != "low_platform" or not job.get("align_platform", False)
        alignment_steps = 0
        required = compute_angles_from_phi(required_phi)
        for step in range(int(job["max_time"] / DT)):
            obs = env.get_obs()
            ref, target_phi = brain.get_action(obs, env.task.global_traj)
            if kind == "low_platform" and brain.target_phi == 2.:
                target_phi = required_phi
            if not aligned:
                target_phi = 1.5
            x, y, raw_yaw = obs["pose"]
            yaw = heading.update(core, phi, raw_yaw)
            cmd = mpc.solve(np.array([x, y, yaw]), ref, last_cmd)
            if not aligned:
                # Complete the sideways approach while the ring still supports all wheels.
                error = (np.deg2rad(job.get("platform_heading_deg", 90.))-yaw+np.pi) % (2*np.pi)-np.pi
                cmd = np.array([0., 0., np.clip(1.8*error, -.8, .8)])
                alignment_steps = alignment_steps+1 if abs(error) < .02 else 0
                aligned = alignment_steps >= 10
            # Stop during transitions; do not push an unfinished shape into a passage.
            q = np.array([p.getJointState(core.robotId, j)[0] for j in core.block_ids])
            target_q = np.array(compute_angles_from_phi(target_phi))
            morphing = abs(target_phi-phi) > .005 or np.max(np.abs(q-target_q)) > criteria.joint_tolerance_rad
            if morphing:
                cmd *= 0
            last_cmd = cmd.copy()
            vx, vy, omega = cmd
            zero_yaws = smoother.update(core.get_steering_zero_yaws_imu())
            for corner, idx in zip(core.STEER_CORNER_ORDER, core.steering_ids):
                swerve.last_angles[corner] = p.getJointState(core.robotId, idx)[0]
            commands = swerve.compute_swerve_kinematics(vx*np.cos(yaw)-vy*np.sin(yaw),
                vx*np.sin(yaw)+vy*np.cos(yaw), omega, core.get_wheel_positions_for_kinematics()[0], zero_yaws)
            phi += float(np.clip(target_phi-phi, -.5*DT, .5*DT))
            action = {"blocks": compute_angles_from_phi(phi),
                      "steering": [commands[c]["angle"] for c in core.STEER_CORNER_ORDER],
                      "velocity": [commands[c]["speed"] for c in core.STEER_CORNER_ORDER]}
            obs, _, info = env.step(action)
            bounds = np.array([p.getAABB(core.robotId, j) for j in range(-1, p.getNumJoints(core.robotId))])
            q = np.array([p.getJointState(core.robotId, j)[0] for j in core.block_ids])
            joint_error = float(np.max(np.abs(q-required)))
            wheel_contacts = {c[3] for c in p.getContactPoints(bodyA=core.robotId, bodyB=core.planeId)
                              if c[3] in core.wheel_ids and c[9] > .01}
            result["success"] = monitor.observe(min_x=float(bounds[:, 0, 0].min()), max_x=float(bounds[:, 1, 0].max()),
                center_y=obs["pose"][1], goal_distance=info["dist_to_goal"], joint_error=joint_error,
                morphology_reached=abs(phi-required_phi) < .005, supported=len(wheel_contacts) >= 2,
                collision=collisions["failed"], invalid_state=not np.isfinite(bounds).all() or not np.isfinite(q).all())
            trace.append([(step+1)*DT, *obs["pose"], phi, target_phi, joint_error,
                          info["dist_to_goal"], int(monitor.crossed), len(wheel_contacts)])
            if result["success"] or monitor.failure:
                result["reason"] = "success" if result["success"] else monitor.failure
                break
        result.update(duration_s=trace[-1][0], final_goal_distance_m=trace[-1][7],
                      final_joint_error_rad=trace[-1][6], final_phi=phi,
                      crossed=bool(monitor.crossed), **collisions)
        result["contact_links"] = [
            {"index": j, "name": p.getJointInfo(core.robotId, j)[12].decode() if j >= 0 else "base",
             "aabb": p.getAABB(core.robotId, j)} for j in sorted(contact_links)]
        result["pivot_angles_rad"] = [p.getJointState(core.robotId, j)[0] for j in core.pivot_ids]
        np.savetxt(Path(job["output"]) / f"{kind}_{seed}_trace.csv", trace, delimiter=",",
                   header="t_s,x,y,yaw,phi,target_phi,joint_error_rad,goal_distance_m,crossed,supported_wheels", comments="")
        return result
    finally:
        env.core_env.close()


def run_job(job):
    log = io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            result = simulate(job)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            result = {"kind": job["kind"], "seed": job["seed"], "success": False,
                      "reason": "execution_error", "error": type(exc).__name__}
    result["wall_time_s"] = time.perf_counter()-started
    output = Path(job["output"])
    (output / f"{job['kind']}_{job['seed']}.log").write_text(log.getvalue(), encoding="utf-8")
    write_json(output / f"{job['kind']}_{job['seed']}_result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026092100)
    parser.add_argument("--max-time", type=float, default=40.)
    parser.add_argument("--nominal", action="store_true")
    parser.add_argument("--kinds", choices=KINDS, nargs="+", default=list(KINDS))
    parser.add_argument("--platform-phi", type=float, default=2.)
    parser.add_argument("--platform-heading-deg", type=float, default=90.)
    parser.add_argument("--align-platform", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=ROOT / "data/my_robot/robot1231_2.urdf")
    args = parser.parse_args()
    if args.trials < 1 or args.workers < 1 or args.max_time <= 0:
        parser.error("trials, workers and max-time must be positive")
    if not 0 <= args.platform_phi <= 3 or not -180 <= args.platform_heading_deg <= 180:
        parser.error("platform Phi must be in [0,3], heading in [-180,180]")
    if not args.urdf.is_file():
        parser.error("Private robot assets are required; restore the local URDF first")
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = [dict(kind=kind, seed=args.seed+index*10000+i, nominal=args.nominal,
                 max_time=args.max_time, output=str(args.output.resolve()), urdf=str(args.urdf.resolve()),
                 platform_phi=args.platform_phi, platform_heading_deg=args.platform_heading_deg,
                 align_platform=args.align_platform)
            for index, kind in enumerate(KINDS) if kind in args.kinds for i in range(args.trials)]
    sources = [p for folder in ("src", "scripts") for p in (ROOT/folder).rglob("*.py")]
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}
    protocol = {"version": PROTOCOL_VERSION, "criteria": asdict(TraversalCriteria()),
                "trials_per_kind": args.trials, "kinds": args.kinds, "nominal": args.nominal, "max_time_s": args.max_time,
                "control_dt_s": DT, "physics_dt_s": 1/240, "decoder": "analytic",
                "platform_phi": args.platform_phi, "platform_heading_deg": args.platform_heading_deg,
                "align_platform": args.align_platform,
                "policy": "scripted brain + MPC + transition pause + geometric heading; no joint teleportation",
                "scope": "Isolated obstacles copied from Scene C; NOT the entire compound course",
                "perturbation": "independent uniform x/y +-0.03m, yaw +-3deg, ground/wheel friction [0.9,1.1]",
                "seeds": [{"kind": j["kind"], "seed": j["seed"]} for j in jobs],
                "python": platform.python_version(), "source_hashes": hashes,
                "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    write_json(args.output / "protocol.json", protocol)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"[{len(results)}/{len(jobs)}] {result['kind']} seed={result['seed']} "
                  f"{result['reason']} sim={result.get('duration_s', 0):.2f}s wall={result['wall_time_s']:.1f}s", flush=True)
    results.sort(key=lambda r: (KINDS.index(r["kind"]), r["seed"]))
    summaries = {}
    for kind in KINDS:
        selected = [r for r in results if r["kind"] == kind]
        if not selected:
            continue
        count = sum(r["success"] for r in selected)
        summaries[kind] = dict(successes=count, trials=len(selected), rate=count/len(selected),
                               wilson_95_interval=wilson_interval(count, len(selected)))
    write_json(args.output / "summary.json", {"by_kind": summaries,
               "all_kinds_at_least_95_percent": len(summaries) == 4 and all(v["rate"] >= .95 for v in summaries.values()),
               "trials": results})
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
