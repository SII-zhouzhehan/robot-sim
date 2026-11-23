"""Play the independent block4--block8 eight-motor loop in MuJoCo.

Linux GUI: python scripts/run_mujoco_closed_loop.py
macOS GUI: mjpython scripts/run_mujoco_closed_loop.py
Headless: python scripts/run_mujoco_closed_loop.py --headless --cycles 1
"""
import argparse
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from controllers.closed_loop_control import MorphTorqueController
from envs.mujoco_closed_loop import ClosedLoopKinematics, build_closed_loop_model
from analyze_closed_loop import make_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--cycles", type=int, default=0, help="0 repeats until the viewer closes")
    parser.add_argument("--hz", type=int, default=1920)
    parser.add_argument("--added-mass", type=float, default=0.15, help="Assumed extra motor/bracket kg")
    args = parser.parse_args()
    if args.cycles < 0:
        parser.error("--cycles must be nonnegative")
    if args.headless and not args.cycles:
        args.cycles = 1
    print("闭环试验：铰点、轴向仍为推算，旧选型待复核；手动碰撞排除项保持不变。\n"
          "详见 CLOSED_LOOP_TRIAL.md。控制目标现已按物理步插补。", flush=True)
    if not args.headless and sys.platform == "darwin":
        # The native viewer dereferences a null primary-monitor pointer in a
        # macOS session with no active displays. Detect the actual condition
        # before entering native code, rather than allowing a SIGSEGV.
        import ctypes
        cg = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
        count = ctypes.c_uint32()
        status = cg.CGGetActiveDisplayList(0, None, ctypes.byref(count))
        if status == 0 and count.value == 0:
            parser.exit(2, "当前 macOS 会话没有活动显示器，无法打开原生 MuJoCo 窗口。\n"
                        "请在有显示器的桌面会话中用 mjpython 运行，或加 --headless --cycles 1。\n")
    print("Building provisional block4--block8 closure and solving a compatible trajectory ...", flush=True)
    path, geometry = build_closed_loop_model(added_mass_kg=args.added_mass)
    ik = ClosedLoopKinematics(mujoco.MjModel.from_xml_path(str(path)))
    trajectory = ik.trajectory(601)
    env = make_env(path, args.hz)
    env.gui = not args.headless
    controller = MorphTorqueController(env, trajectory)
    phi = 1.5
    try:
        if env.gui:
            env.open_viewer()
        print(f"8 morph motors; pivots passive; extra mass {geometry['added_mass_kg']} kg; physics {args.hz} Hz.\n"
              "Orange disk = inferred eighth motor. Close the viewer to stop.", flush=True)
        targets = [0, 3, 0]
        cycles = 0
        max_gap = max_angle = max_tracking = 0.0
        peak = np.zeros(8)
        while not args.cycles or cycles < args.cycles:
            for target in targets:
                ticks = int(round(abs(target - phi) / .5 / env.control_dt)) + 480
                for _ in range(ticks):
                    if env.gui and not env.viewer.is_running():
                        return
                    start = time.monotonic()
                    phi += float(np.clip(target - phi, -.5 * env.control_dt, .5 * env.control_dt))
                    controller.step(phi)
                    error = ik.errors(env.data)
                    max_gap = max(max_gap, float(np.linalg.norm(error[:3]) * .2 * 1000))
                    max_angle = max(max_angle, float(np.linalg.norm(error[3:]) * 180 / np.pi))
                    max_tracking = max(max_tracking, float(np.max(abs(env.data.qpos[controller.qadr] - controller.target)) * 180 / np.pi))
                    peak = np.maximum(peak, abs(env.data.qfrc_actuator[controller.dofs]))
                    if env.gui:
                        env.render()
                        delay = env.control_dt - (time.monotonic() - start)
                        if delay > 0:
                            time.sleep(delay)
                print(f"Phi={phi:.3f}; max gap={max_gap:.4f} mm, angle={max_angle:.4f} deg, "
                      f"tracking={max_tracking:.3f} deg; sampled peak Nm (240 Hz)={np.round(peak, 3)}", flush=True)
            targets = [3, 0]
            cycles += 1
    finally:
        env.close()


if __name__ == "__main__":
    main()
