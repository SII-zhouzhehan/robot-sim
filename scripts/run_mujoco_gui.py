"""MuJoCo manual control: MorphGNN/formula + PID + four-wheel swerve."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from controllers.mujoco_control import MujocoDriveController
from envs.mujoco_model import DEFAULT_PIVOT_DAMPING, DEFAULT_PIVOT_FRICTIONLOSS
from envs.mujoco_robot_env import MujocoRobotEnv
from utils.mujoco_control_panel import ControlPanel, ControlState


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--source", choices=("gnn", "formula"), default="gnn")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--phi", type=float, default=1.5)
    parser.add_argument("--vx", type=float, default=0)
    parser.add_argument("--vy", type=float, default=0)
    parser.add_argument("--omega", type=float, default=0)
    parser.add_argument("--no-pid", action="store_true")
    parser.add_argument("--no-panel", action="store_true", help="Use the native viewer's keyboard controls")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--passive-pivots", action=argparse.BooleanOptionalAction, default=True,
                        help="Gravity-driven hinges with friction; --no-passive-pivots restores the legacy upright servo")
    parser.add_argument("--pivot-frictionloss", type=float, default=DEFAULT_PIVOT_FRICTIONLOSS,
                        help="Dry pivot friction in Nm (provisional default: .05)")
    parser.add_argument("--pivot-damping", type=float, default=DEFAULT_PIVOT_DAMPING,
                        help="Viscous pivot damping in Nm*s/rad (provisional default: .05)")
    parser.add_argument("--seconds", type=float, help="Stop after this many simulated seconds")
    args = parser.parse_args()
    if args.seconds is not None and args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.headless and args.seconds is None:
        parser.error("--headless requires --seconds")
    if any(not math.isfinite(v) or v < 0 for v in (args.pivot_frictionloss, args.pivot_damping)):
        parser.error("Pivot resistance must be finite and nonnegative")
    env = MujocoRobotEnv(args.model, gui=not args.headless, control_steps=4,
                        passive_pivots=args.passive_pivots, pivot_frictionloss=args.pivot_frictionloss,
                        pivot_damping=args.pivot_damping)
    panel = None
    try:
        controller = MujocoDriveController(env, args.source, args.checkpoint, not args.no_pid)
        env.reset()
        controller.reset()
        state = ControlState(source=args.source, pid=not args.no_pid)
        state.command(dict(phi=args.phi, vx=args.vx, vy=args.vy, omega=args.omega))
        if not args.headless:
            env.open_viewer(state.key)
            if not args.no_panel:
                panel = ControlPanel(state)
                print(f"Slider panel: {panel.url}", flush=True)
            print("W/S: X speed | A/D: Y speed | Q/E: rotation | Z/X: Phi | Space: stop | R: reset | G/P: GNN/PID", flush=True)
        if controller.predictor.checkpoint:
            print(f"GNN: {controller.predictor.checkpoint}", flush=True)
        wall_start, sim_start = time.perf_counter(), float(env.data.time)
        next_tick = wall_start
        while args.headless or env.viewer.is_running():
            if state.consume_reset():
                env.reset()
                controller.reset()
                state.update(phi=1.5, vx=0, vy=0, omega=0)
                wall_start, sim_start = time.perf_counter(), float(env.data.time)
                next_tick = wall_start
            command = state.snapshot()
            controller.advance(command["vx"], command["vy"], command["omega"], command["phi"],
                               source=command["source"], use_pid=command["pid"])
            elapsed = float(env.data.time) - sim_start
            state.update(actual_phi=controller.phi, pose=env.get_robot_pose(), velocity=env.get_world_velocity(),
                         realtime=elapsed / max(time.perf_counter() - wall_start, 1e-6),
                         wheel_contacts=env.get_contact_summary()["wheel_contacts"])
            if not args.headless:
                env.render(force=True)
                next_tick += env.control_dt
                remaining = next_tick - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                elif remaining < -0.25:
                    # Recover after dragging the window or a suspended process.
                    next_tick = time.perf_counter()
            if args.seconds is not None and elapsed >= args.seconds - 1e-9:
                break
        simulated = float(env.data.time) - sim_start
        wall_seconds = time.perf_counter() - wall_start
        print(json.dumps({"engine": "mujoco", "phi": controller.phi,
                          "pivot_actuation": "passive" if env.passive_pivots else "legacy_active_hold",
                          "pivot_frictionloss_nm": args.pivot_frictionloss,
                          "pivot_damping_nms_rad": args.pivot_damping,
                          "simulated_seconds": simulated, "wall_seconds": wall_seconds,
                          "realtime": simulated / max(wall_seconds, 1e-6), "pose": env.get_robot_pose(),
                          "velocity": env.get_world_velocity(), **env.get_contact_summary()}, indent=2))
    except KeyboardInterrupt:
        pass
    finally:
        if panel is not None:
            panel.close()
        env.close()


if __name__ == "__main__":
    main()
