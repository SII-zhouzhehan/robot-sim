"""Generate MuJoCo assets without modifying the source URDF or meshes."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from envs.mujoco_model import DEFAULT_MODEL, DEFAULT_URDF, build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--visual-voxel", type=float, default=0.004, help="Display mesh voxel size in metres")
    args = parser.parse_args()
    if args.visual_voxel <= 0:
        parser.error("--visual-voxel must be positive")
    path, report = build_model(args.urdf, args.output, args.visual_voxel)
    print(f"MuJoCo model: {path}")
    print(json.dumps({"meshes": report["total"], **report["model"]}, indent=2))


if __name__ == "__main__":
    main()
