"""Passive hinges have no tracking target and must serialize as JSON null."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("matplotlib")


def test_passive_holding_report_has_no_tracking_error_or_motor_torque(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_motor_torque.py"
    spec = importlib.util.spec_from_file_location("motor_torque_report", script)
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    args = SimpleNamespace(
        model=study.DEFAULT_MODEL, urdf=study.DEFAULT_URDF, output=tmp_path,
        control_steps=2, timestep_divisor=2, block_gain_scale=30.,
        passive_pivots=True, static_points=2, settle_seconds=.02,
        static_sample_seconds=.02, pivot_frictionloss=.05, pivot_damping=.05,
    )
    rows = study.run_holds(args)
    # Strict JSON is the report writer's contract; NaN breaks the whole export.
    exported = json.loads(json.dumps(rows, allow_nan=False))
    passive = [row for row in exported if row["group"] == "pivot"]
    assert len(passive) == 8 * args.static_points
    assert all(row["tracking_error_p95_rad"] is None for row in passive)
    assert all(row["peak_abs_nm"] == 0 for row in passive)
    assert all(np.isfinite(row["tracking_error_p95_rad"])
               for row in exported if row["group"] == "block")
