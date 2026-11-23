"""Preliminary eight-motor sizing from corrected-controller full load tests.

Run --simulate to repeat both physics rates and all 26 holding cases per rate.
Without that flag, read the saved simulation_report.json and export the table.
Manual collision exclusions are preserved; hinge geometry remains inferred.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import mujoco
import numpy as np

from analyze_closed_loop import run_case
from envs.mujoco_closed_loop import build_closed_loop_model, ClosedLoopKinematics, MORPH_NAMES
from envs.mujoco_model import DEFAULT_MODEL

DEFAULT_OUTPUT = ROOT / "experiments/closed_loop_sizing_20261020_interpolated"


def workspace_path(value):
    """Resolve current and historical report paths after workspace migration."""
    path = Path(value)
    if not path.is_absolute():
        return ROOT / path
    if path.exists():
        return path
    if "robot-sim" in path.parts:
        return ROOT.joinpath(*path.parts[path.parts.index("robot-sim") + 1:])
    return path


def excluded_pairs(path):
    return {frozenset((e.get("body1"), e.get("body2")))
            for e in ET.parse(path).getroot().findall("contact/exclude")}


def simulate(out):
    out.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(DEFAULT_MODEL.read_bytes()).hexdigest()
    model_path, geometry = build_closed_loop_model(output=out / "robot_closed_loop.xml", added_mass_kg=.15)
    ik = ClosedLoopKinematics(mujoco.MjModel.from_xml_path(str(model_path)))
    trajectory = ik.trajectory(601)
    np.savez_compressed(out / "compatible_trajectory.npz", phi=trajectory[0],
                        joint_names=MORPH_NAMES, position_rad=trajectory[1])
    report = {"geometry": geometry, "source_sha256": source_hash, "mujoco_version": mujoco.__version__,
              "physics_frequencies_hz": [1920, 3840], "control_hz": 240,
              "target_interpolation": True, "motor_sizing_status": "preliminary_for_inferred_geometry",
              "margin_factor": 1.5, "manual_collision_exclusions_preserved": True, "cases": {}}
    for hz in report["physics_frequencies_hz"]:
        print(f"Testing {hz} Hz, full motion and 26 holds ...", flush=True)
        report["cases"][f"closed_{hz}_mass_0.150"] = run_case(
            model_path, trajectory, hz, out / f"closed_{hz}_mass_0.150.npz", holds=True)
        (out / "simulation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if hashlib.sha256(DEFAULT_MODEL.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("The original seven-joint model changed")
    return report


def verify_and_size(report):
    """Recompute loads from raw samples before applying the declared margin."""
    if report["physics_frequencies_hz"] != [1920, 3840]:
        raise ValueError("Both declared physics rates are required")
    source = workspace_path(report["geometry"]["source_model"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != report["source_sha256"]:
        raise ValueError("Saved data no longer matches the original model")
    closed_path = workspace_path(next(iter(report["cases"].values()))["data"]).parent / "robot_closed_loop.xml"
    if excluded_pairs(source) != excluded_pairs(closed_path):
        raise ValueError("Manual collision exclusions changed")
    original, closed = [mujoco.MjModel.from_xml_path(str(p)) for p in (source, closed_path)]
    for g in range(original.ngeom):
        copy = closed.geom(original.geom(g).name).id
        if (original.geom_contype[g] != closed.geom_contype[copy] or
                original.geom_conaffinity[g] != closed.geom_conaffinity[copy]):
            raise ValueError("An existing collision mask changed")

    motion, holding, peak = [], [], []
    for hz in report["physics_frequencies_hz"]:
        case = report["cases"][f"closed_{hz}_mass_0.150"]
        if not (case["target_interpolation"] and case["holds_tested"] and not case["warnings"]):
            raise ValueError("Require corrected control, complete holds and no numerical warnings")
        quality = case["hold_quality_from_position_derivative"]
        if len(quality) != 26 or not all(h["settled"] for h in quality):
            raise ValueError("A required holding pose has not settled")
        with np.load(workspace_path(case["data"])) as a:
            torque, phase = a["torque_nm"], a["phase"]
            if torque.shape[1] != 8 or not np.isfinite(torque).all():
                raise ValueError("Require eight finite actuator torque channels")
            motion.append(np.sqrt(np.mean(torque[np.isin(phase, ("forward", "reverse"))] ** 2, axis=0)))
            holds = [np.sqrt(np.mean(torque[phase == p] ** 2, axis=0))
                     for p in np.unique(phase) if p.startswith("hold_")]
            if len(holds) != 26:
                raise ValueError("Raw data does not contain all 26 holds")
            holding.append(np.max(holds, axis=0))
            peak.append(np.max(abs(torque), axis=0))
        for index, field in enumerate(("motion_rms_nm", "worst_hold_rms_nm", "peak_nm")):
            recomputed = (motion, holding, peak)[index][-1]
            np.testing.assert_allclose(recomputed, [j[field] for j in case["joints"]], rtol=1e-10, atol=1e-12)

    rms, hold, maxima = [np.max(v, axis=0) for v in (motion, holding, peak)]
    margin = float(report["margin_factor"])
    rows = []
    for i, name in enumerate(MORPH_NAMES):
        continuous = float(max(rms[i], hold[i]))
        rated = math.ceil(continuous * margin * 2 - 1e-10) / 2
        burst = math.ceil(float(maxima[i]) * margin - 1e-10)
        rows.append({"joint": name, "label": f"J{i+1}", "motion_rms_nm": float(rms[i]),
                     "worst_hold_rms_nm": float(hold[i]), "continuous_proxy_nm": continuous,
                     "peak_nm": float(maxima[i]), "margin_factor": margin,
                     "preliminary_rated_nm": rated, "preliminary_peak_nm": burst,
                     "candidate_3_5_12_5_raw_pass": bool(continuous <= 3.5 and maxima[i] <= 12.5),
                     "candidate_3_5_12_5_with_50percent_margin": bool(continuous * margin <= 3.5 and maxima[i] * margin <= 12.5)})
    return rows


def export(out, report, rows):
    result = {"status": "preliminary_for_inferred_geometry", "final_motor_sizing_valid": False,
              "source_simulation_report": str((out / "simulation_report.json").resolve()),
              "source_sha256": report["source_sha256"], "target_interpolation": True,
              "manual_collision_exclusions_preserved": True,
              "margin_factor": report["margin_factor"], "physics_frequencies_hz": report["physics_frequencies_hz"],
              "holds_settled": 52, "rows": rows,
              "uniform_minimum_rated_nm": max(r["preliminary_rated_nm"] for r in rows),
              "uniform_minimum_peak_nm": max(r["preliminary_peak_nm"] for r in rows),
              "limitation": "New hinge axes/anchors are inferred. These joint-output torque values support preliminary selection only; calibrate geometry and verify actual motor duty/thermal limits."}
    (out / "motor_selection.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (out / "motor_selection.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 八关节闭环电机预选表（修正控制后）", "",
             "当前推算模型的预选值。保留全部手动碰撞排除项，新增铰点及轴向仍需装配校准。",
             "以下均为减速器后关节输出端 N·m；不是电机裸轴扭矩。", "",
             "| 关节 | 运动 RMS | 最重保持 RMS | 连续需求 | 仿真最大 | 建议额定 +50% | 建议峰值 +50% |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    if (out / "presentation/eight_joint_motor_selection.pptx").exists():
        lines[5:5] = ["[可编辑一页 PPTX](presentation/eight_joint_motor_selection.pptx)　[高清表格图片](presentation/eight_joint_motor_selection.png)", ""]
    for r in rows:
        label = r["label"] + (" 新增闭环" if r["joint"] == "joint_closure" else "")
        lines.append(f'| {label} | {r["motion_rms_nm"]:.3f} | {r["worst_hold_rms_nm"]:.3f} | {r["continuous_proxy_nm"]:.3f} | {r["peak_nm"]:.3f} | {r["preliminary_rated_nm"]:g} | {r["preliminary_peak_nm"]:g} |')
    fail = "、".join(r["label"] for r in rows if not r["candidate_3_5_12_5_with_50percent_margin"])
    lines += ["", "连续需求 = max(正反向运动 RMS，26 个保持工况各自 RMS)。两档物理频率取包络。",
              "建议额定 = 连续需求 ×1.5，向上取到 0.5 N·m；建议峰值 = 全程最大绝对力矩 ×1.5，向上取到 1 N·m。", "",
              f"统一电机规格按本模型至少额定 {result['uniform_minimum_rated_nm']:g} N·m、峰值 {result['uniform_minimum_peak_nm']:g} N·m；可优先按 3.5 / 15 N·m 规格预选。",
              f"原 3.5 / 12.5 N·m 候选覆盖本模型未加余量的最大值，但 {fail} 不满足 50% 峰值余量。", "",
              "## 工况与校核", "",
              "- 平地、无额外载荷，新增电机及支架假设 0.15 kg。Φ=0～3 六秒正向、六秒反向，两端各保持两秒。",
              "- 每档频率另测 13 个形态，从两个方向到达，共 26 个静态保持。先稳定五秒，再采样一秒。两档共 52 个保持均通过稳定性检查。",
              "- 物理 1920 / 3840 Hz，控制 240 Hz，目标在物理步内插补。所有输出扭矩逐物理步记录。URDF effort 限幅关闭。",
              "- 四个 pivot 被动，干摩擦 0.05 N·m、阻尼 0.05 N·m·s/rad。转向保持初始角，轮毂零速保持。",
              "- 原 XML 哈希、手动碰撞排除项和既有碰撞掩码逐项核对未变；数据重新计算与保存统计相符。",
              "- 新增关节几何仍为推算，约 14.6° 对称关节姿态差尚未校准。50% 为设计余量，不表示模型误差已被证明小于 50%。",
              "- 峰值持续时间、输出转速、减速器效率、供电和热能力仍需结合实际电机规格验证。", "",
              "## 两档频率的负载", "",
              "| 物理 Hz | 最大运动 RMS | 最大保持 RMS | 全局最大扭矩 | 最大跟踪误差° | 闭环位置误差 mm |",
              "|---:|---:|---:|---:|---:|---:|"]
    for case in report["cases"].values():
        js = case["joints"]
        lines.append(f'| {case["physics_hz"]} | {max(j["motion_rms_nm"] for j in js):.3f} | {max(j["worst_hold_rms_nm"] for j in js):.3f} | {max(j["peak_nm"] for j in js):.3f} | {max(j["max_tracking_error_deg"] for j in js):.3f} | {case["max_closure_gap_mm"]:.4f} |')
    lines += ["", "## 复现", "", "```bash", "conda activate robot-sim",
              "python scripts/report_closed_loop_sizing.py --simulate", "```", "",
              "省略 --simulate 可直接校验已有原始数据并重新导出选型表，不重复仿真。", ""]
    (out / "report.md").write_text("\n".join(lines))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    report = simulate(out) if args.simulate else json.loads((out / "simulation_report.json").read_text())
    rows = verify_and_size(report)
    result = export(out, report, rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
