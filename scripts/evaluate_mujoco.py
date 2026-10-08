"""Run scripted MPC or a PPO policy in independent MuJoCo A/B/C scenes."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from brains.scripted_brain import ScriptedBrain
from controllers.mpc_controller import KinematicMPC
from controllers.mujoco_control import MujocoDriveController
from envs.mujoco_benchmark_env import MujocoBenchmarkEnv
from envs.mujoco_brain_gym_env import MujocoBrainGymEnv
from evaluation.obstacle_protocol import TraversalMonitor


def run(args):
    gym_env = None
    if args.mode == "ppo":
        from stable_baselines3 import PPO
        gym_env = MujocoBrainGymEnv(gui=args.gui, scene_id=args.scene, source=args.source,
                                    checkpoint=args.gnn_checkpoint, imitation_weight=0,
                                    max_steps=max(1, int(np.ceil(args.seconds))), seed=args.seed)
        observation, _ = gym_env.reset(seed=args.seed)
        policy = PPO.load(args.policy, device="cpu")
        env, controller = gym_env.env, gym_env.controller
    else:
        env = MujocoBenchmarkEnv(gui=args.gui, scene_id=args.scene, control_steps=4, seed=args.seed)
        env.reset()
        controller = MujocoDriveController(env.core_env, args.source, args.gnn_checkpoint)
        brain, mpc = ScriptedBrain(), KinematicMPC()
        if args.gui:
            env.core_env.open_viewer()
    monitors = []
    for obstacle in env.obstacles:
        half_depth = {"narrow_gate": .1, "low_platform": .25, "bridge": .2,
                      "cylinders": obstacle.get("radius", .2)}[obstacle["type"]]
        half_lane = obstacle.get("width", 1.0) / 2
        monitors.append(TraversalMonitor(obstacle["x"] - half_depth, obstacle["x"] + half_depth,
                                          obstacle.get("y", 0), half_lane))
    rows, collision_seen = [], False
    previous_command, world_command, target_phi = np.zeros(3), np.zeros(3), 1.5
    wall_start, sim_start = time.perf_counter(), float(env.core_env.data.time)
    next_tick = wall_start
    tick = 0
    try:
        while float(env.core_env.data.time) - sim_start < args.seconds - 1e-9:
            terminated = False
            if gym_env:
                action, _ = policy.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = gym_env.step(action)
                terminated |= truncated
            else:
                if tick % 3 == 0:
                    reference, target_phi = brain.get_action(env.get_obs(), env.task.global_traj)
                    previous_command = mpc.solve(np.array(env.core_env.get_robot_pose()), reference, previous_command)
                yaw = env.core_env.get_robot_yaw()
                vx, vy, omega = previous_command
                world_command[:] = [vx * np.cos(yaw) - vy * np.sin(yaw), vx * np.sin(yaw) + vy * np.cos(yaw), omega]
                joint_error_to_target = np.max(np.abs(env.core_env.joint_positions(env.core_env.block_ids)
                                                      - controller.predictor.predict(target_phi)))
                if abs(target_phi - controller.phi) > 0.02 or joint_error_to_target > 0.08:
                    world_command[:] = 0
                controller.advance(*world_command, phi_target=target_phi)
                info = env.get_info()
            collision_seen |= info["obstacle_collision"]
            bounds = env.core_env.get_robot_bounds()
            pose = env.core_env.get_robot_pose()
            joint_error = float(np.max(np.abs(env.core_env.joint_positions(env.core_env.block_ids)
                                              - controller.predictor.predict(controller.phi))))
            for monitor in monitors:
                # A course report tracks crossing/corridor failures; the isolated
                # obstacle protocol's final settle criterion is not a course score.
                monitor.observe(min_x=bounds[0, 0], max_x=bounds[1, 0], center_y=pose[1],
                                goal_distance=info["dist_to_goal"], joint_error=joint_error,
                                morphology_reached=abs(controller.phi - target_phi) < .02,
                                supported=info["supported"])
            if tick % 3 == 0 or gym_env:
                rows.append({"time": float(env.core_env.data.time) - sim_start, "pose": pose,
                             "phi": controller.phi, "joint_error_rad": joint_error, **info})
            if args.gui and not gym_env:
                env.core_env.render(force=True)
                next_tick += env.core_env.control_dt
                remaining = next_tick - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                elif remaining < -.25:
                    next_tick = time.perf_counter()
                if not env.core_env.viewer.is_running():
                    break
            if info["success"] or terminated:
                break
            tick += 1
        traversal = [{"type": obstacle["type"], "approached": m.approached, "crossed": m.crossed,
                      "failure": m.failure} for obstacle, m in zip(env.obstacles, monitors)]
        final = env.get_info()
        return {"engine": "mujoco", "scene": args.scene, "mode": args.mode, "source": args.source,
                "seed": args.seed, "simulated_seconds": float(env.core_env.data.time) - sim_start,
                "wall_seconds": time.perf_counter() - wall_start, "goal_reached": bool(final["success"]),
                "collision_seen": bool(collision_seen), "success": bool(final["success"] and not collision_seen
                    and final["supported"] and all(m.crossed and not m.failure for m in monitors)),
                "traversal": traversal, "samples": rows}
    finally:
        (gym_env or env).close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=list("ABC"), default="B")
    parser.add_argument("--mode", choices=("scripted", "ppo"), default="scripted")
    parser.add_argument("--policy", type=Path, help="PPO .zip, required for --mode ppo")
    parser.add_argument("--source", choices=("gnn", "formula"), default="gnn")
    parser.add_argument("--gnn-checkpoint", type=Path)
    parser.add_argument("--seconds", type=float, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gui", action="store_true", help="On macOS launch with mjpython")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.mode == "ppo" and args.policy is None:
        parser.error("--mode ppo requires --policy")
    report = run(args)
    output = args.output or PROJECT_ROOT / "experiments" / "mujoco_evaluations" / f"{args.scene}_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
