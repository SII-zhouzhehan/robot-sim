"""Audit internal contacts, closure distortion, and command-update torque spikes.

Disabling self collision is a diagnostic ablation only; ground contact remains.
The original model files are never changed by these comparisons.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from analyze_closed_loop import make_env
from controllers.closed_loop_control import MorphTorqueController
from envs.mujoco_closed_loop import ClosedLoopKinematics, DEFAULT_CLOSED_MODEL


def remove_internal_collision_bit(model):
    """Remove robot--robot bit 4, retaining robot--ground bits 2/1."""
    ids = np.flatnonzero(model.geom_group == 2)
    model.geom_contype[ids] &= ~4
    model.geom_conaffinity[ids] &= ~4


def run(path, trajectory, hz, output, *, self_collision=True, interpolation=False):
    env = make_env(path, hz)
    if not self_collision:
        remove_internal_collision_bit(env.model)
    # Repeat the same clean reset in every case, after changing only the
    # collision masks; neither floor contact nor joint/actuator gains change.
    env.reset()
    controller = MorphTorqueController(env, trajectory, interpolate_targets=interpolation)
    kinematics = ClosedLoopKinematics(env.model)
    rows, pair_stats = [], {}
    phi, phase, tick = 1.5, "prepare", 0
    start = float(env.data.time)
    force = np.zeros(6)

    def observe(e):
        nonlocal tick
        mujoco.mj_forward(e.model, e.data)
        jac = e.data.efc_J.reshape(e.data.nefc, e.model.nv)
        internal_qfrc = np.zeros(e.model.nv)
        max_depth, max_force, internal_count, ground_count = 0.0, 0.0, 0, 0
        for i, contact in enumerate(e.data.contact[:e.data.ncon]):
            geoms = (int(contact.geom1), int(contact.geom2))
            if e.ground_geom_id in geoms:
                ground_count += 1
                continue
            mujoco.mj_contactForce(e.model, e.data, i, force)
            internal_count += 1
            depth, normal = max(0.0, -float(contact.dist)), float(force[0])
            max_depth, max_force = max(max_depth, depth), max(max_force, normal)
            key = " / ".join(sorted(e.model.geom(g).name for g in geoms))
            stat = pair_stats.setdefault(key, {"contact_samples": 0, "max_depth_mm": 0.0, "max_normal_force_n": 0.0})
            stat["contact_samples"] += 1
            stat["max_depth_mm"] = max(stat["max_depth_mm"], depth * 1000)
            stat["max_normal_force_n"] = max(stat["max_normal_force_n"], normal)
            address = int(contact.efc_address)
            if address >= 0:
                # Both source models use an elliptic contact cone.
                count = int(contact.dim)
                internal_qfrc += jac[address:address + count].T @ e.data.efc_force[address:address + count]
        eq = (e.data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
        closure_qfrc = jac[eq].T @ e.data.efc_force[eq]
        errors = kinematics.errors(e.data)
        q = e.data.qpos[controller.qadr]
        asymmetry = np.degrees(q[[0, 1, 2]] - q[[4, 5, 6]])
        rows.append((float(e.data.time) - start, phi, phase, tick % env.control_steps,
                     e.data.qfrc_actuator[controller.dofs].copy(), q.copy(), controller.target.copy(),
                     e.data.qpos.copy(), e.data.qvel.copy(), e.data.ctrl.copy(),
                     internal_count, ground_count, max_depth * 1000, max_force,
                     internal_qfrc[controller.dofs].copy(), closure_qfrc[controller.dofs].copy(),
                     np.linalg.norm(errors[:3]) * .2 * 1000,
                     np.linalg.norm(errors[3:]) * 180 / np.pi, asymmetry))
        tick += 1

    def advance(target, seconds, record=False):
        nonlocal phi
        for _ in range(round(seconds / env.control_dt)):
            phi += float(np.clip(target - phi, -.5 * env.control_dt, .5 * env.control_dt))
            controller.step(phi, observe if record else None)

    try:
        advance(0, 3)
        advance(0, 5)
        start = float(env.data.time)
        for phase, target, seconds in (("forward", 3, 6), ("end_hold_3", 3, 2),
                                        ("reverse", 0, 6), ("end_hold_0", 0, 2)):
            advance(target, seconds, True)
        fields = ("time_s", "phi", "phase", "tick_in_control", "torque_nm", "position_rad", "target_rad",
                  "full_qpos", "full_qvel", "ctrl", "internal_contacts", "ground_contacts", "internal_depth_mm",
                  "internal_normal_force_n", "internal_contact_generalized_nm", "closure_generalized_nm",
                  "closure_gap_mm", "closure_angle_deg", "paired_joint_asymmetry_deg")
        arrays = {name: np.asarray([r[i] for r in rows]) for i, name in enumerate(fields)}
        np.savez_compressed(output, **arrays)
        motion = np.isin(arrays["phase"], ("forward", "reverse"))
        peak = np.max(abs(arrays["torque_nm"]), axis=0)
        index, joint = np.unravel_index(np.argmax(abs(arrays["torque_nm"])), arrays["torque_nm"].shape)
        return {"physics_hz": hz, "control_hz": 240, "self_collision_enabled": self_collision,
                "target_interpolation": interpolation,
                "motion_rms_nm": np.sqrt(np.mean(arrays["torque_nm"][motion] ** 2, axis=0)).tolist(),
                "peak_nm": peak.tolist(), "overall_peak_nm": float(peak.max()),
                "overall_peak_joint": int(joint + 1), "overall_peak_time_s": float(arrays["time_s"][index]),
                "overall_peak_phi": float(arrays["phi"][index]),
                "overall_peak_phase": str(arrays["phase"][index]),
                "overall_peak_tick_in_control": int(arrays["tick_in_control"][index]),
                "internal_contact_sample_count": int(np.count_nonzero(arrays["internal_contacts"])),
                "max_internal_penetration_mm": float(arrays["internal_depth_mm"].max()),
                "max_internal_normal_force_n": float(arrays["internal_normal_force_n"].max()),
                "max_internal_contact_generalized_nm": float(abs(arrays["internal_contact_generalized_nm"]).max()),
                "max_paired_angle_asymmetry_deg": float(abs(arrays["paired_joint_asymmetry_deg"]).max()),
                "max_tracking_error_deg": float(abs(np.degrees(arrays["position_rad"] - arrays["target_rad"])).max()),
                "max_closure_gap_mm": float(arrays["closure_gap_mm"].max()),
                "max_closure_angle_deg": float(arrays["closure_angle_deg"].max()),
                "min_ground_contact_count": int(arrays["ground_contacts"].min()),
                "pair_stats": pair_stats, "data": str(output),
                "warnings": {mujoco.mjtWarning(i).name: int(w.number) for i, w in enumerate(env.data.warning) if w.number}}
    finally:
        env.close()


def plot_report(out, report):
    font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    if font.exists():
        from matplotlib.font_manager import FontProperties
        plt.rcParams["font.family"] = FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    cases = report["cases"]
    first_hz = report["frequencies"][0]
    labels = [(f"self_{first_hz}", "原控制 / 原碰撞设置"),
              (f"no_self_{first_hz}", "原控制 / 关闭内部碰撞（对照）"),
              (f"interpolated_{first_hz}", "目标插补 / 原碰撞设置")]
    fig, axs = plt.subplots(2, 2, figsize=(16, 9), layout="constrained")
    x = np.arange(8)
    for i, (key, label) in enumerate(labels):
        case = cases[key]
        axs[0, 0].bar(x + (i - 1) * .25, case["peak_nm"], .25, label=label)
        axs[0, 1].bar(x + (i - 1) * .25, case["motion_rms_nm"], .25, label=label)
        a = np.load(case["data"])
        mask = (a["phase"] == "forward") & (a["time_s"] < .22)
        axs[1, 0].plot(a["time_s"][mask], a["torque_nm"][mask, 4], lw=1, label=label)
    for ax, title in zip(axs[0], ("各关节峰值", "运动 RMS")):
        ax.set(title=title, ylabel="扭矩 / N·m", xticks=x, xticklabels=[f"J{i}" for i in range(1, 9)])
        ax.legend(fontsize=9)
    axs[1, 0].set(title="正向起步 J5：控制目标更新的影响", xlabel="时间 / s", ylabel="扭矩 / N·m")
    axs[1, 0].legend(fontsize=9)
    a = np.load(cases[f"self_{first_hz}"]["data"])
    for i, name in enumerate(("J1−J5", "J2−J6", "J3−J7")):
        axs[1, 1].plot(a["time_s"], a["paired_joint_asymmetry_deg"][:, i], lw=1, label=name)
    axs[1, 1].set(title="对称关节角度差：闭合轨迹本身的偏斜", xlabel="时间 / s", ylabel="角度差 / °")
    axs[1, 1].legend(fontsize=9)
    for ax in axs.ravel():
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.savefig(out / "contact_diagnosis.png", dpi=220)
    fig.savefig(out / "contact_diagnosis.svg")
    plt.close(fig)


def write_report(out, report):
    report["user_constraint"] = ("Preserve intentional collision exclusions and masks. "
                                 "Excluded CAD mesh crossings are not physical interference evidence.")
    report["motor_sizing_valid"] = False
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 第八关节的接触、轨迹与控制复核", "",
             "**保留用户手动关闭的碰撞对。用户说明这些排除用于补偿 CAD 精度，实物不会碰撞。不能用这些网格交叉证明实物干涉。**", "",
             "## 本次可以确认的结果", "",
             "1. 原设置下模块间求解器接触为零，内部碰撞反力为零。独立运行中关闭内部碰撞后，电机扭矩数组逐项相同。地面碰撞设置保持启用，原 XML 和手动排除列表未改动。",
             "2. 推算闭环让原本对称的关节对出现约 14.63° 的角度差；这是刚体关节姿态偏差，不是材料弹性扭曲。需复核新关节的铰点、轴向及参考装配。",
             "3. 控制目标每 1/240 秒整步跳变会放大瞬时峰值。相同端点在物理步内线性插补后，峰值显著降低。插补已成为闭环试验控制器默认设置。", "",
             "## 六秒正向、六秒反向及两端各两秒保持", "",
             "新增电机及支架假设 0.15 kg；平地；240 Hz 控制；pivot 被动，保留原有摩擦。所有 effort 限幅关闭。", "",
             "| 物理 Hz | 原设置最大峰值 N·m | 关闭内部碰撞 N·m | 两组扭矩最大差 N·m | 插补后最大峰值 N·m | 原设置内部接触采样数 |",
             "|---:|---:|---:|---:|---:|---:|"]
    for hz in report["frequencies"]:
        a, b, c = [report["cases"][f"{mode}_{hz}"] for mode in ("self", "no_self", "interpolated")]
        lines.append(f'| {hz} | {a["overall_peak_nm"]:.3f} | {b["overall_peak_nm"]:.3f} | {b["max_torque_difference_from_self_nm"]:.6f} | {c["overall_peak_nm"]:.3f} | {a["internal_contact_sample_count"]} |')
    lines += ["", "插补试验只改变目标到达各物理步的方式，目标端点、用时、增益、模型质量和碰撞设置相同。",
              "最大峰值的下降不能直接解释为实物电机的需求下降，装配几何仍未经校准。", "",
              "![接触与控制对照](contact_diagnosis.png)", "",
              "## 网格检查的适用范围", "",
              "使用原始 STL 约 81 万个三角面、61 个 Φ 采样点，独立检查非相邻 block。",
              "新闭环轨迹检测到 block2 / block6（Φ=2.45～3.00，共 12 个采样姿态）、block1 / block4（Φ=0～0.25，共 6 个采样姿态）网格交叉。",
              "原七关节公式也有已排除模块对的网格交叉。所有检出的对都属于原有手动排除项，不能认定为实物干涉，更不能自动启用这些碰撞。",
              "同时要求相交三角面跨过对方平面超过 10 μm，排除了仅共面贴合；该量不是穿透深度。未检查相邻关节接口、实体包容或连续运动，不能推导碰撞力。",
              "最大扭矩动态姿态也检测到已排除的 block2 / block6 网格交叉，仅作为建模记录。", "",
              "![已排除模块对的网格示意，不代表实物干涉](block2_block6_interference.png)", "",
              "## 修改与保留", "",
              "- 原模型碰撞掩码和手动排除列表保持原样；关闭内部碰撞仅为独立运行内存中的消融对照。",
              "- 闭环试验控制器增加物理步内目标插补；用 interpolate_targets=False 可重现旧阶梯行为。原七关节入口控制器不变。",
              "- 旧闭环电机选型建议撤回。旧动态 NPZ 保留，图表和报告标为待装配与控制复核；插补数据不是新的采购建议。",
              "- 新增关节仍按现有公式拟合，未擅自调整任何原 CAD 关节的位置、轴向或网格。", "",
              "## 复现", "", "```bash",
              "conda activate robot-sim",
              "python scripts/diagnose_closed_loop_contacts.py",
              "python scripts/audit_closed_loop_meshes.py --deps-dir experiments/closed_loop_contact_audit_20261020/mesh_audit_deps --states experiments/closed_loop_contact_audit_20261020/self_1920.npz --render",
              "```", "",
              "python-fcl 为独立几何检查的可选依赖，本机放在 experiments 下，未改变 robot-sim 环境中的包。",
              "NPZ 包含每个物理步的完整位姿、速度、输出扭矩、目标、接触与约束诊断；JSON 包含统计及数值警告。", ""]
    (out / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_CLOSED_MODEL)
    parser.add_argument("--trajectory", type=Path, default=ROOT / "experiments/closed_loop_20261020/compatible_trajectory.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / f"closed_loop_contact_audit_{datetime.now():%Y%m%d}")
    parser.add_argument("--frequencies", nargs="+", type=int, default=[1920, 3840])
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    tr = np.load(args.trajectory)
    trajectory = tr["phi"], tr["position_rad"]
    report = {"frequencies": args.frequencies, "cases": {}}
    for hz in args.frequencies:
        for label, collision, interpolation in (("self", True, False), ("no_self", False, False),
                                                ("interpolated", True, True)):
            key = f"{label}_{hz}"
            print(f"Testing {key} ...", flush=True)
            case = run(args.model, trajectory, hz, out / f"{key}.npz",
                       self_collision=collision, interpolation=interpolation)
            report["cases"][key] = case
            print(json.dumps(case, ensure_ascii=False), flush=True)
            (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for hz in args.frequencies:
        a, b = [np.load(report["cases"][f"{mode}_{hz}"]["data"]) for mode in ("self", "no_self")]
        report["cases"][f"no_self_{hz}"]["max_torque_difference_from_self_nm"] = float(abs(a["torque_nm"] - b["torque_nm"]).max())
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    plot_report(out, report)
    write_report(out, report)
    print(f"Saved audit to {out}", flush=True)


if __name__ == "__main__":
    main()
