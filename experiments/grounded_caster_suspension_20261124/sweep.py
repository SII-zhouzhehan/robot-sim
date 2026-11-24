"""Systematic multi-round parameter sweep for TARS 8-caster 6 mm suspension."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import json
import sys
import tempfile
import time

import mujoco as mj
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(OUT),
    str(ROOT / "experiments/grounded_phi0_3"),
    str(ROOT / "experiments/grounded_phi_buffer"),
    str(ROOT / "experiments/grounded_heading_audit"),
]

from build_suspension import CASTER_SPECS, PART_PREFIX, SuspensionConfig, TIRE_SPECS, build_model
from envs.tars_grounded import TarsGroundedEnv
import control
import study as audit_study


def _contact_breakdown(m, d, floor, tire_g, ball_g, housing_g):
    tire_f = np.zeros(4)
    ball_f = np.zeros(8)
    house_f = 0.0
    other_f = 0.0
    tire_pen = 0.0
    ball_pen = 0.0
    for i, c in enumerate(d.contact):
        if floor not in c.geom:
            continue
        other = c.geom2 if c.geom1 == floor else c.geom1
        w = np.empty(6)
        mj.mj_contactForce(m, d, i, w)
        fn = max(0.0, float(w[0]))
        pen = max(0.0, float(-c.dist * 1000.0))
        if other in tire_g:
            tire_f[tire_g[other]] += fn
            tire_pen = max(tire_pen, pen)
        elif other in ball_g:
            ball_f[ball_g[other]] += fn
            ball_pen = max(ball_pen, pen)
        elif other in housing_g:
            house_f += fn
        else:
            other_f += fn
    return tire_f, ball_f, house_f, other_f, tire_pen, ball_pen


def eval_planar_drive(env: TarsGroundedEnv, twists=(("y+", (0.0, 0.08, 0.0)), ("x+", (0.08, 0.0, 0.0)), ("yaw+", (0.0, 0.0, 0.30)))):
    """Evaluate planar swerve driving + braking stop across multiple directions."""
    m, d = env.model, env.data
    floor = m.geom("ctm_floor").id
    tire_g = {m.pair_geom2[m.pair(p).id]: idx for idx, (_, _, p) in enumerate(TIRE_SPECS)}
    ball_g = {m.geom(f"{PART_PREFIX}{bp}_collision_0").id: idx for idx, (_, _, _, bp, _) in enumerate(CASTER_SPECS)}
    housing_g = {
        m.geom(f"{c}_housing_collision").id
        for c, *_ in CASTER_SPECS
        if mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, f"{c}_housing_collision") >= 0
    }
    cids = np.array([m.joint(f"{c}_slide").id for c, *_ in CASTER_SPECS])
    cqadr = m.jnt_qposadr[cids]

    env.control_steps = 5  # 100 Hz planar drive
    results = {}
    for label, cmd in twists:
        env.reset()
        start = env.get_robot_pose()
        tilts, loads_t, loads_b, strokes, yaw_errs = [], [], [], [], []
        house_max, other_max, pen_max = 0.0, 0.0, 0.0
        n_drive = round(2.0 / env.control_dt)
        n_stop = round(0.6 / env.control_dt)
        for k in range(n_drive + n_stop):
            tw = cmd if k < n_drive else (0.0, 0.0, 0.0)
            env.drive(*tw)
            tilt = np.degrees(np.arcsin(np.clip(np.abs(d.xaxis[env.wheel_ids, 2]), 0.0, 1.0)))
            tf, bf, hf, of, tpen, _ = _contact_breakdown(m, d, floor, tire_g, ball_g, housing_g)
            tilts.append(tilt.max())
            loads_t.append(tf)
            loads_b.append(bf.sum())
            strokes.append(d.qpos[cqadr] * 1000.0)
            pose = env.get_robot_pose() - start
            if abs(cmd[2]) < 1e-6:
                yaw_errs.append(abs(np.degrees(np.arctan2(np.sin(pose[2]), np.cos(pose[2])))))
            house_max = max(house_max, hf)
            other_max = max(other_max, of)
            pen_max = max(pen_max, tpen)
        end_delta = env.get_robot_pose() - start
        loads_t = np.asarray(loads_t)
        loads_b = np.asarray(loads_b)
        strokes = np.asarray(strokes)
        tilts = np.asarray(tilts)
        results[label] = {
            "four_pct": round(float(100.0 * (loads_t > 1.0).all(1).mean()), 2),
            "tire_frac": round(float(loads_t.sum() / max(loads_t.sum() + loads_b.sum(), 1e-6)), 4),
            "peak_tilt_deg": round(float(tilts.max()), 2),
            "stop_tilt_deg": round(float(tilts[-10:].mean()), 2),
            "max_stroke_mm": round(float(strokes.max()), 3),
            "p95_stroke_mm": round(float(np.percentile(strokes.max(1), 95)), 3),
            "yaw_drift_deg": round(float(max(yaw_errs)) if yaw_errs else 0.0, 2),
            "delta_xy_cm": [round(float(end_delta[0] * 100), 2), round(float(end_delta[1] * 100), 2)],
            "delta_yaw_deg": round(float(np.degrees(end_delta[2])), 2),
            "max_tire_pen_mm": round(float(pen_max), 3),
            "house_max_N": round(float(house_max), 2),
            "other_max_N": round(float(other_max), 2),
        }
    return results


def eval_morph_cases(env: TarsGroundedEnv, grid, path, snap, cases):
    """Run grounded_heading_audit.study.run on selected morphing cases + suspension metrics."""
    m, d = env.model, env.data
    floor = m.geom("ctm_floor").id
    tire_g = {m.pair_geom2[m.pair(p).id]: idx for idx, (_, _, p) in enumerate(TIRE_SPECS)}
    ball_g = {m.geom(f"{PART_PREFIX}{bp}_collision_0").id: idx for idx, (_, _, _, bp, _) in enumerate(CASTER_SPECS)}
    housing_g = {
        m.geom(f"{c}_housing_collision").id
        for c, *_ in CASTER_SPECS
        if mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, f"{c}_housing_collision") >= 0
    }
    cids = np.array([m.joint(f"{c}_slide").id for c, *_ in CASTER_SPECS])
    cqadr = m.jnt_qposadr[cids]

    summaries = {}
    traces = {}
    for name, cmd, mode in cases:
        mj.mj_copyData(d, m, snap)
        env._wheel_heading = None
        env._morph_motion = None
        ball_loads, house_loads, other_loads, pens, strokes, tilts = [], [], [], [], [], []
        orig_dm = env.drive_morph

        def capture(*args):
            out = orig_dm(*args)
            tf, bf, hf, of, tpen, bpen = _contact_breakdown(m, d, floor, tire_g, ball_g, housing_g)
            ball_loads.append(bf)
            house_loads.append(hf)
            other_loads.append(of)
            pens.append((tpen, bpen))
            strokes.append(d.qpos[cqadr] * 1000.0)
            tilt = np.degrees(np.arcsin(np.clip(np.abs(d.xaxis[env.wheel_ids, 2]), 0.0, 1.0)))
            tilts.append(tilt)
            return out

        with patch.object(env, "drive_morph", capture):
            r, z = audit_study.run(env, grid, path, np.array(cmd, dtype=float), mode)

        ball_loads = np.asarray(ball_loads)
        house_loads = np.asarray(house_loads)
        other_loads = np.asarray(other_loads)
        pens = np.asarray(pens)
        strokes = np.asarray(strokes)
        tilts = np.asarray(tilts)

        L = z["load"] > 0.1
        miss = np.where(~L.all(1))[0]
        if len(miss):
            spans = np.split(miss, np.where(np.diff(miss) != 1)[0] + 1)
            max_unloaded_s = max(len(s) for s in spans) * env.dt
        else:
            max_unloaded_s = 0.0

        L1 = z["load"] > 1.0
        tire_sum = float(z["load"].sum())
        ball_sum = float(ball_loads.sum())

        r.update(
            strong_four_pct=round(float(100.0 * L1.all(1).mean()), 2),
            min_wheel_N=round(float(z["load"].min()), 2),
            p05_min_wheel_N=round(float(np.percentile(z["load"].min(1), 5)), 2),
            max_unloaded_s=round(float(max_unloaded_s), 4),
            tire_load_frac=round(float(tire_sum / max(tire_sum + ball_sum, 1e-6)), 4),
            mean_tire_N=round(float(z["load"].sum(1).mean()), 2),
            mean_ball_N=round(float(ball_loads.sum(1).mean()), 2),
            max_tire_pen_mm=round(float(pens[:, 0].max()), 3),
            max_ball_pen_mm=round(float(pens[:, 1].max()), 3),
            max_stroke_mm=round(float(strokes.max()), 3),
            p95_stroke_mm=round(float(np.percentile(strokes.max(1), 95)), 3),
            mean_active_stroke_mm=round(float(np.maximum(strokes, 0.0).mean()), 3),
            bottomout_pct=round(float(100.0 * (strokes >= 5.8).any(1).mean()), 2),
            peak_tilt_deg=round(float(tilts.max()), 2),
            house_max_N=round(float(house_loads.max()), 2),
            other_max_N=round(float(other_loads.max()), 2),
        )
        z["ball_load"] = ball_loads
        z["strokes_mm"] = strokes
        z["tilts_deg"] = tilts
        z["pens_mm"] = pens
        summaries[name] = r
        traces[name] = z
    return summaries, traces


def evaluate_config(label: str, cfg: SuspensionConfig, full: bool = False):
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        _, scene = build_model(Path(tmp), cfg)
        env = TarsGroundedEnv(scene)
        drive = eval_planar_drive(env)
        # Prepare phi=0 start via grounded_phi_buffer.control.prepare()
        with patch.object(control, "TarsGroundedEnv", lambda: TarsGroundedEnv(scene)):
            env_m, grid, path = control.prepare()
        snap = mj.MjData(env_m.model)
        mj.mj_copyData(snap, env_m.model, env_m.data)
        cases = (
            [
                ("still", [0, 0, 0], "active"),
                ("moving", [0.04, 0.03, 0.12], "active"),
                ("lateral", [0, 0.05, 0], "active"),
                ("reverse", [-0.04, -0.03, -0.12], "active"),
                ("half", [0.02, 0.015, 0.06], "active"),
                ("coast", [0, 0, 0], "coast"),
            ]
            if full
            else [
                ("still", [0, 0, 0], "active"),
                ("moving", [0.04, 0.03, 0.12], "active"),
            ]
        )
        morph, traces = eval_morph_cases(env_m, grid, path, snap, cases)
    elapsed = round(time.monotonic() - t0, 1)
    return {"label": label, "elapsed_s": elapsed, "config": cfg.to_dict(), "drive": drive, "morph": morph}, traces


def run_round1():
    base = SuspensionConfig()
    candidates = [
        # 1. Geometry fidelity ablation (raw CAD mesh vs smooth cylinder+sphere)
        ("r1_mesh_k1200_f1", replace(base, smooth_tires=False, smooth_balls=False)),
        # 2. Spring stiffness sweep (at preload=1.0 N, damping=15, pivot_fl=0.25, tire_solref=0.010)
        ("r1_k400_f1_c10", replace(base, stiffness=400.0, preload_n=1.0, damping=10.0)),
        ("r1_k800_f1_c14", replace(base, stiffness=800.0, preload_n=1.0, damping=14.0)),
        ("r1_k1200_f1_c18", replace(base, stiffness=1200.0, preload_n=1.0, damping=18.0)),
        ("r1_k1800_f1_c22", replace(base, stiffness=1800.0, preload_n=1.0, damping=22.0)),
        ("r1_k2600_f1_c28", replace(base, stiffness=2600.0, preload_n=1.0, damping=28.0)),
        ("r1_k4000_f1_c35", replace(base, stiffness=4000.0, preload_n=1.0, damping=35.0)),
        # 3. Preload sweep (at k=1200)
        ("r1_k1200_f0_c18", replace(base, stiffness=1200.0, preload_n=0.0, damping=18.0)),
        ("r1_k1200_f0p3_c18", replace(base, stiffness=1200.0, preload_n=0.3, damping=18.0)),
        ("r1_k1200_f2p5_c18", replace(base, stiffness=1200.0, preload_n=2.5, damping=18.0)),
        # 4. Pivot friction sweep (at k=1200, f=0.5)
        ("r1_k1200_piv0p05", replace(base, stiffness=1200.0, preload_n=0.5, damping=18.0, pivot_frictionloss=0.05)),
        ("r1_k1200_piv0p15", replace(base, stiffness=1200.0, preload_n=0.5, damping=18.0, pivot_frictionloss=0.15)),
        ("r1_k1200_piv0p50", replace(base, stiffness=1200.0, preload_n=0.5, damping=18.0, pivot_frictionloss=0.50)),
        ("r1_k1200_piv0p75", replace(base, stiffness=1200.0, preload_n=0.5, damping=18.0, pivot_frictionloss=0.75)),
        # 5. Tire compliance sweep (at k=1200, f=0.5, piv=0.25)
        ("r1_k1200_ts006", replace(base, stiffness=1200.0, preload_n=0.5, damping=18.0, tire_solref="0.006 1")),
        ("r1_k1200_ts014", replace(base, stiffness=1200.0, preload_n=0.5, damping=18.0, tire_solref="0.014 1")),
        ("r1_k1200_ts020", replace(base, stiffness=1200.0, preload_n=0.5, damping=18.0, tire_solref="0.020 1")),
    ]

    all_res = []
    hdr = (
        f"{'label':20s} | {'drv4%':>5s} {'drvTf':>5s} {'drvTlt':>6s} {'drvYaw':>6s} | "
        f"{'stl4%':>5s} {'stlErr':>6s} {'stlSlp':>6s} | "
        f"{'mov4%':>5s} {'mov1N':>5s} {'movErr':>6s} {'movSlp':>6s} {'mPen':>5s} {'mStrk':>5s} {'bot%':>5s} {'mTlt':>5s} {'mTf':>5s}"
    )
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for label, cfg in candidates:
        res, _ = evaluate_config(label, cfg, full=False)
        all_res.append(res)
        d = res["drive"]
        drv4 = min(v["four_pct"] for v in d.values())
        drvTf = np.mean([v["tire_frac"] for v in d.values()])
        drvTlt = max(v["peak_tilt_deg"] for v in d.values())
        drvYaw = max(v["yaw_drift_deg"] for v in d.values())
        s = res["morph"]["still"]
        m = res["morph"]["moving"]
        print(
            f"{label:20s} | {drv4:5.1f} {drvTf:5.3f} {drvTlt:6.2f} {drvYaw:6.2f} | "
            f"{s['four_pct']:5.1f} {s['max_err_mm']:6.1f} {s['slip_rms_mm_s']:6.1f} | "
            f"{m['four_pct']:5.1f} {m['strong_four_pct']:5.1f} {m['max_err_mm']:6.1f} {m['slip_rms_mm_s']:6.1f} "
            f"{m['max_tire_pen_mm']:5.2f} {m['max_stroke_mm']:5.2f} {m['bottomout_pct']:5.1f} {m['peak_tilt_deg']:5.2f} {m['tire_load_frac']:5.3f}",
            flush=True,
        )
    (OUT / "round1_sweep.json").write_text(json.dumps(all_res, indent=2) + "\n")


if __name__ == "__main__":
    run_round1()
