"""Train a separate PPO policy on MuJoCo; write to a separate MuJoCo checkpoint directory."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from envs.mujoco_brain_gym_env import MujocoBrainGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=list("ABC"), default="B")
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--source", choices=("gnn", "formula"), default="gnn")
    parser.add_argument("--gnn-checkpoint", type=Path)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--imitation-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.timesteps < 1 or args.rollout_steps < 2 or args.max_steps < 1:
        parser.error("timesteps/max-steps must be positive; rollout-steps must be at least 2")
    output = args.output or PROJECT_ROOT / "experiments" / f"mujoco_ppo_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(parents=True, exist_ok=False)
    env = Monitor(MujocoBrainGymEnv(scene_id=args.scene, source=args.source, checkpoint=args.gnn_checkpoint,
                                    max_steps=args.max_steps, imitation_weight=args.imitation_weight, seed=args.seed),
                  str(output / "train"))
    try:
        policy = PPO("MlpPolicy", env, device="cpu", n_steps=args.rollout_steps,
                     batch_size=min(64, args.rollout_steps), n_epochs=4, seed=args.seed,
                     verbose=1, tensorboard_log=str(output / "tensorboard"))
        policy.learn(total_timesteps=args.timesteps)
        policy.save(output / "ppo_mujoco")
        (output / "run.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
        print(f"MuJoCo PPO policy: {output / 'ppo_mujoco.zip'}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
