"""Generate MuJoCo assets without modifying the source URDF or meshes."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from envs.mujoco_model import (DEFAULT_MODEL, DEFAULT_URDF, DEFAULT_PIVOT_DAMPING,
                              DEFAULT_PIVOT_FRICTIONLOSS, build_model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--visual-voxel", type=float, default=0.004, help="Display mesh voxel size in metres")
    parser.add_argument("--pivot-frictionloss", type=float, default=DEFAULT_PIVOT_FRICTIONLOSS,
                        help="Dry friction per pivot in Nm (provisional default: .05)")
    parser.add_argument("--pivot-damping", type=float, default=DEFAULT_PIVOT_DAMPING,
                        help="Viscous damping per pivot in Nm*s/rad (provisional default: .05)")
    args = parser.parse_args()
    if args.visual_voxel <= 0:
        parser.error("--visual-voxel must be positive")
    if any(not math.isfinite(v) or v < 0 for v in (args.pivot_frictionloss, args.pivot_damping)):
        parser.error("Pivot resistance must be finite and nonnegative")
    path, report = build_model(args.urdf, args.output, args.visual_voxel,
                               pivot_frictionloss=args.pivot_frictionloss, pivot_damping=args.pivot_damping)
    print(f"MuJoCo model: {path}")
    print(json.dumps({"meshes": report["total"], **report["model"]}, indent=2))


if __name__ == "__main__":
    main()
