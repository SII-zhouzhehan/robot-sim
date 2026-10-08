"""Run a short Phi=3 stability check in MuJoCo."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from controllers.mujoco_control import MujocoDriveController
from envs.mujoco_robot_env import MujocoRobotEnv

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    env = MujocoRobotEnv(gui=args.gui, control_steps=4)
    try:
        controller = MujocoDriveController(env, source="formula")
        env.reset()
        controller.reset()
        if args.gui:
            env.open_viewer()
        while env.data.time < args.seconds:
            controller.advance(0.0, 0.0, 0.0, 3.0)
            if args.gui:
                env.render(force=True)
                if not env.viewer.is_running():
                    break
        print({"engine": "mujoco", "phi": controller.phi,
               "pose": env.get_robot_pose(), "contacts": env.get_contact_summary()})
    finally:
        env.close()

if __name__ == "__main__":
    main()
