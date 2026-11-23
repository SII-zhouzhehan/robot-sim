"""Compare the original seven motors with a provisional eight-motor loop.

Run with the robot-sim Python; GUI playback is in run_mujoco_closed_loop.py.
All torques are joint-output Nm, without URDF effort limits.
"""
import argparse
import csv
from datetime import datetime
import hashlib
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
from controllers.closed_loop_control import MorphTorqueController
from controllers.mujoco_control import angles_from_phi
from envs.mujoco_closed_loop import (ClosedLoopKinematics, MORPH_NAMES, build_closed_loop_model)
from envs.mujoco_model import DEFAULT_MODEL
from envs.mujoco_robot_env import MujocoRobotEnv


def make_env(path, hz):
    if hz < 240 or hz % 240:
        raise ValueError("Physics frequency must be an integer multiple of 240 Hz")
    env = MujocoRobotEnv(path, passive_pivots=True, control_steps=hz // 240)
    env.model.opt.timestep = env.dt = 1 / hz
    env.control_dt = 1 / 240
    env.model.actuator_forcelimited[:] = 0
    env.model.jnt_actfrclimited[:] = 0
    ids = [i for i in range(env.model.nu) if env.model.actuator(i).name in
           {f"{n}_position" for n in MORPH_NAMES}]
    env.model.actuator_gainprm[ids, 0] *= 30
    env.model.actuator_biasprm[ids, 1] *= 30
    env.model.actuator_biasprm[ids, 2] *= np.sqrt(30)
    env.reset()
    return env


def run_case(path, trajectory, hz, output, *, holds=True):
    env = make_env(path, hz)
    controller = MorphTorqueController(env, trajectory)
    kinematics = ClosedLoopKinematics(env.model) if trajectory is not None else None
    rows = []
    phi = 1.5
    phase = "prepare"
    start = float(env.data.time)

    def observe(e):
        mujoco.mj_forward(e.model, e.data)
        error = np.zeros(6) if kinematics is None else kinematics.errors(e.data)
        ground_contacts = set()
        for contact in e.data.contact[:e.data.ncon]:
            pair = {int(contact.geom1), int(contact.geom2)}
            if e.ground_geom_id in pair:
                ground_contacts.update(pair - {e.ground_geom_id})
        wheels = len(ground_contacts & e._wheel_geom_set)
        modules = sum(e.model.geom(f"wheel{i}_collision").id in ground_contacts or
                      e.model.geom(f"steering{i}_collision").id in ground_contacts for i in range(1, 5))
        rows.append((float(e.data.time) - start, phi, phase,
                     e.data.qfrc_actuator[controller.dofs].copy(),
                     e.data.qpos[controller.qadr].copy(), controller.target.copy(),
                     np.linalg.norm(error[:3]) * 0.2 * 1000,
                     np.linalg.norm(error[3:]) * 180 / np.pi, wheels, modules))

    def advance(target, seconds=None, record=False):
        nonlocal phi
        if seconds is None:
            seconds = abs(target - phi) / 0.5
        for _ in range(int(np.ceil(seconds / env.control_dt - 1e-9))):
            phi += float(np.clip(target - phi, -0.5 * env.control_dt, 0.5 * env.control_dt))
            controller.step(phi, observe if record else None)

    try:
        advance(0)
        advance(0, 5)
        start = float(env.data.time)
        for phase, target, seconds in (("forward", 3, 6), ("end_hold_3", 3, 2),
                                        ("reverse", 0, 6), ("end_hold_0", 0, 2)):
            advance(target, seconds, True)
        if holds:
            for direction, phis in (("ascending", np.linspace(0, 3, 13)),
                                    ("descending", np.linspace(3, 0, 13))):
                advance(float(phis[0]))
                advance(float(phis[0]), 5)
                for p in phis:
                    advance(float(p))
                    advance(float(p), 5)
                    phase = f"hold_{direction}_{p:.2f}"
                    advance(float(p), 1, True)
        fields = ("time_s", "phi", "phase", "torque_nm", "position_rad", "target_rad",
                  "closure_gap_mm", "closure_angle_deg", "wheel_contacts", "module_contacts")
        arrays = {name: np.asarray([row[i] for row in rows]) for i, name in enumerate(fields)}
        np.savez_compressed(output, **arrays)
        motion = np.isin(arrays["phase"], ["forward", "reverse"])
        hold_quality = []
        for hold_phase in np.unique(arrays["phase"]):
            if not hold_phase.startswith("hold_"):
                continue
            mask = arrays["phase"] == hold_phase
            speed = np.diff(arrays["position_rad"][mask], axis=0) / np.diff(arrays["time_s"][mask])[:, None]
            p95 = float(np.percentile(np.max(abs(speed), axis=1), 95))
            hold_quality.append({"phase": str(hold_phase), "motor_speed_p95_rad_s": p95,
                                 "settled": p95 < .03})
        summary = []
        for i, name in enumerate(MORPH_NAMES[:len(controller.joints)]):
            torque = arrays["torque_nm"][:, i]
            static = [float(np.sqrt(np.mean(torque[arrays["phase"] == phase] ** 2)))
                      for phase in np.unique(arrays["phase"]) if phase.startswith("hold_")]
            peak_index = int(np.argmax(abs(torque)))
            summary.append({"joint": name, "motion_rms_nm": float(np.sqrt(np.mean(torque[motion] ** 2))),
                            "worst_hold_rms_nm": max(static, default=0),
                            "peak_nm": float(abs(torque[peak_index])),
                            "peak_phase": str(arrays["phase"][peak_index]),
                            "peak_phi": float(arrays["phi"][peak_index]),
                            "max_tracking_error_deg": float(np.max(abs(arrays["position_rad"][:, i] -
                                                                       arrays["target_rad"][:, i])) * 180 / np.pi)})
        try:
            data_reference = output.relative_to(ROOT).as_posix()
        except ValueError:
            data_reference = str(output)
        return {"physics_hz": hz, "control_hz": 240, "closed": trajectory is not None,
                "target_interpolation": controller.interpolate_targets,
                "holds_tested": holds, "joints": summary,
                "max_closure_gap_mm": float(arrays["closure_gap_mm"].max()),
                "max_closure_angle_deg": float(arrays["closure_angle_deg"].max()),
                "hold_quality_from_position_derivative": hold_quality,
                "motion_four_modules_percent": float(np.mean(arrays["module_contacts"][motion] == 4) * 100),
                "motion_four_tires_percent": float(np.mean(arrays["wheel_contacts"][motion] == 4) * 100),
                "warnings": {mujoco.mjtWarning(i).name: int(w.number) for i, w in enumerate(env.data.warning)
                             if w.number}, "data": data_reference}
    finally:
        env.close()


def envelope(cases):
    count = len(cases[0]["joints"])
    result = []
    for i in range(count):
        records = [c["joints"][i] for c in cases]
        rms = max(r["motion_rms_nm"] for r in records)
        hold = max(r["worst_hold_rms_nm"] for r in records)
        peak = max(r["peak_nm"] for r in records)
        continuous = max(rms, hold)
        result.append({"joint": records[0]["joint"], "motion_rms_nm": rms, "worst_hold_rms_nm": hold,
                       "continuous_proxy_nm": continuous, "peak_nm": peak,
                       # Closure axes/anchors are inferred, and the historical
                       # command-update peaks require revalidation. Intentional
                       # CAD collision exclusions are preserved, not defects.
                       "motor_sizing_valid": False,
                       "suggested_rated_nm_50percent": None,
                       "suggested_peak_nm_50percent": None,
                       "candidate_3_5_12_5_with_margin": None})
    return result


def render_poses(path, trajectory, output):
    """Render the actual simulated mechanism at four shapes, including J8."""
    env = make_env(path, 1920)
    controller = MorphTorqueController(env, trajectory)
    renderer = mujoco.Renderer(env.model, width=960, height=640)
    option = mujoco.MjvOption()
    option.geomgroup[2] = 0
    phi = 1.5
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), layout="constrained")
    try:
        for target, ax in zip((0, 1, 2, 3), axes.ravel()):
            ticks = int(round((abs(target - phi) / .5 + 3) / env.control_dt))
            for _ in range(ticks):
                phi += float(np.clip(target - phi, -.5 * env.control_dt, .5 * env.control_dt))
                controller.step(phi)
            camera = mujoco.MjvCamera()
            camera.distance, camera.azimuth, camera.elevation = 1.25, 130, -35
            camera.lookat[:] = np.mean([env.data.xpos[env.model.body(f"block{i}").id] for i in range(1, 9)], axis=0)
            renderer.update_scene(env.data, camera=camera, scene_option=option)
            ax.imshow(renderer.render())
            ax.set_title(f"Phi = {target} | orange: motor J8")
            ax.axis("off")
        fig.savefig(output, dpi=180)
    finally:
        plt.close(fig)
        renderer.close()
        env.close()


def render_animation(path, trajectory, output, fps=10):
    """Export a full six-second forward + six-second reverse physics replay."""
    from PIL import Image, ImageDraw
    if 240 % fps:
        raise ValueError("Animation fps must divide 240")
    env = make_env(path, 1920)
    controller = MorphTorqueController(env, trajectory)
    renderer = mujoco.Renderer(env.model, width=640, height=416)
    option = mujoco.MjvOption()
    option.geomgroup[2] = 0
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 1.3, 130, -30
    frames = []
    phi = 1.5
    try:
        for _ in range(3 * 240):
            phi = max(0, phi - .5 / 240)
            controller.step(phi)
        for _ in range(5 * 240):
            controller.step(0)
        phi = 0.0
        for target in (3, 0):
            for _ in range(6 * fps):
                for _ in range(240 // fps):
                    phi += float(np.clip(target - phi, -.5 / 240, .5 / 240))
                    controller.step(phi)
                camera.lookat[:] = np.mean([env.data.xpos[env.model.body(f"block{i}").id] for i in range(1, 9)], axis=0)
                renderer.update_scene(env.data, camera=camera, scene_option=option)
                frame = Image.fromarray(renderer.render())
                ImageDraw.Draw(frame).text((12, 12), f"8-motor closed loop | Phi={phi:.2f} | orange = J8", fill="white", stroke_width=1, stroke_fill="black")
                frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
        frames[0].save(output, save_all=True, append_images=frames[1:], duration=1000 // fps,
                       loop=0, optimize=True, disposal=2)
    finally:
        renderer.close()
        env.close()


def figures(out, report):
    old, new = report["seven_joint_envelope"], report["eight_joint_envelope"]
    font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    if font.exists():
        from matplotlib.font_manager import FontProperties
        plt.rcParams["font.family"] = FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11})
    fig, axs = plt.subplots(2, 2, figsize=(16, 9), layout="constrained")
    fig.suptitle("推算闭环尚待装配校准 · 以下为仿真数值对照，选型待复核", color="#A12D22", fontsize=17)
    x = np.arange(8)
    for ax, field, title in ((axs[0, 0], "continuous_proxy_nm", "RMS 数值对照：运动与最重保持工况取大值"),
                              (axs[0, 1], "peak_nm", "仿真峰值扭矩（未加余量）")):
        ax.bar(x[:7] - .18, [r[field] for r in old], .36, label="原七关节", color="#9AA6B2")
        ax.bar(x + .18, [r[field] for r in new], .36, label="八关节闭环", color="#167BA6")
        ax.set(xticks=x, xticklabels=[f"J{i}" for i in range(1, 9)], ylabel="关节输出扭矩 / N·m", title=title)
        ax.legend(fontsize=9)
    for index, hz in enumerate(report["physics_frequencies_hz"]):
        ls = ("--", "-", ":")[index % 3]
        key = f"closed_{hz}_mass_0.150"
        if key not in report["cases"]:
            continue
        data = np.load(report["cases"][key]["data"])
        use = data["phase"] == "forward"
        t = data["time_s"][use]
        for i, color in ((0, "#167BA6"), (4, "#E58528"), (7, "#8553A6")):
            axs[1, 0].plot(t, data["torque_nm"][use, i], ls=ls, lw=1,
                           color=color, label=f"J{i+1} · {hz} Hz")
        axs[1, 1].plot(t, data["closure_gap_mm"][use], ls=ls, lw=1.2, label=f"{hz} Hz")
    axs[1, 0].set(title="6 秒正向变形：J1、J5 与新增 J8", xlabel="时间 / s", ylabel="扭矩 / N·m")
    axs[1, 0].legend(fontsize=9, ncol=2)
    axs[1, 1].set(title="闭环端部位置误差", xlabel="时间 / s", ylabel="误差 / mm")
    axs[1, 1].legend()
    for ax in axs.ravel():
        ax.grid(alpha=.2, axis="y")
        ax.set_axisbelow(True)
    fig.savefig(out / "closed_loop_comparison.png", dpi=240)
    fig.savefig(out / "closed_loop_comparison.svg")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(16, 9), layout="constrained")
    ax.axis("off")
    rows = [[f"J{i+1}" + (" 新增" if i == 7 else ""), f'{r["motion_rms_nm"]:.2f}',
             f'{r["worst_hold_rms_nm"]:.2f}', f'{r["peak_nm"]:.2f}',
             "待装配校核", "待装配校核"]
            for i, r in enumerate(new)]
    table = ax.table(cellText=rows, colLabels=["关节", "运动 RMS", "最重保持", "仿真峰值", "建议额定 +50%", "建议峰值 +50%"],
                     cellLoc="center", bbox=[.02, .18, .96, .66])
    table.auto_set_font_size(False)
    table.set_fontsize(18)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor("#15384B" if row == 0 else ("#EAF3F8" if row % 2 else "#F6F8FA"))
        if row == 0:
            cell.set_text_props(color="white")
    ax.text(.02, .93, "八电机闭环 · 轴点待校准，选型待复核", transform=ax.transAxes,
            fontsize=27, weight="bold", color="#A12D22")
    hz_text = " / ".join(str(hz) for hz in report["physics_frequencies_hz"])
    ax.text(.02, .08, f"单位 N·m；新增质量假设 0.15 kg；平地；6 秒单程；{hz_text} Hz 包络。\n保留手动碰撞排除。闭环轴点为推算，旧控制峰值受目标跳变影响；历史数值仅供诊断。", transform=ax.transAxes, fontsize=13, color="#4F6573")
    fig.savefig(out / "eight_motor_selection.png", dpi=240)
    fig.savefig(out / "eight_motor_selection.svg")
    plt.close(fig)


def write_report(out, report):
    report["motor_sizing_valid"] = False
    report["validation_rejection"] = "Closure axes/anchors inferred and command-update peaks require revalidation; intentional collision exclusions preserved."
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    table = report["eight_joint_envelope"]
    with (out / "motor_selection.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    hz_text = " / ".join(str(hz) for hz in report["physics_frequencies_hz"])
    lines = ["# 八电机闭环试验（装配待校准，旧选型待复核）", "",
             "**闭环铰点与轴向仍为推算，旧峰值明显受目标跳变影响。下列力矩仅为数值诊断，不能用于电机采购；此前 +50% 选型建议撤回。手动碰撞排除项保留，网格相交不视为实物干涉证据。**", "",
             "详见 `CLOSED_LOOP_TRIAL.md` 和 `experiments/closed_loop_contact_audit_20261020/report.md`。",
             f"控制目标步内插补：{report.get('target_interpolation', '见各工况 target_interpolation；旧数据未插补')}。旧阶梯目标数据不会因代码更新而变成插补数据。", "",
             "新增关节：block4 ↔ block8。原始 URDF、七关节 XML 与原控制入口不变。",
             "新关节名 joint_closure，图表记为 J8；原 joint8 仍是无电机驱动的被动 pivot。", "",
             "## 条件与有效范围", "", "- 平地、无外加负载、Φ=0→3 六秒及反向六秒；两端各保持两秒。",
             "- 13 个姿态，升序和降序到达，各稳定五秒后采样一秒，共 26 个保持工况。",
             f"- 控制 240 Hz，物理 {hz_text} Hz；pivot 干摩擦 0.05 N·m、阻尼 0.05 N·m·s/rad，pivot 无主动电机。",
             "- 四个转向电机保持初始角度，轮毂电机零速保持。七关节和八关节对照使用相同条件；与以往转向跟随/不同步长的报告不可直接混用。",
             "- 所有电机与关节 effort 限幅关闭；原 CAD 的角度限位保留；地面与轮胎摩擦延用原模型。",
             "- 新增电机/支架暂按 0.15 kg，分布在关节两侧；质量敏感性另测 0 / 0.30 kg。",
             "- 真实铰链 + 转子与 block8 的 weld 约束。八路输出采用约束切空间最小二乘分配，避免冗余位置伺服互相顶力；结果依赖该控制策略。",
             "- 八关节轨迹由原公式投影到可闭合轨迹，所以同一 Φ 的七个原关节角度会改变；表格比较的是同一变形任务，并非完全相同角度轨迹。",
             "- 连续扭矩代理值 = max(正反向运动 RMS, 26 个保持工况各自 RMS)。当前装配与控制复核未完成，暂不给出电机选型判断。", "",
             "## 几何与数值检查", "", f'- 原公式端部最大闭合偏差：{report["geometry"]["original_formula_max_gap_mm"]:.3f} mm。',
             f'- 投影后原七关节最大角度调整：{report["trajectory"]["max_original_angle_change_deg"]:.2f}°。',
             f'- 采样点最大闭合位置误差：{report["trajectory"]["max_gap_mm"]:.6f} mm。',
             f'- 全部闭环动力学工况最大位置误差：{max(c["max_closure_gap_mm"] for c in report["cases"].values() if c["closed"]):.4f} mm。',
             f'- 全部闭环动力学工况最大角度误差：{max(c["max_closure_angle_deg"] for c in report["cases"].values() if c["closed"]):.4f}°。', "",
             f"## 0.15 kg 新增质量，{hz_text} Hz 包络（N·m）", "",
             "| 关节 | 运动 RMS | 最重保持 RMS | 仿真峰值 | 建议额定 +50% | 建议峰值 +50% |",
             "|---|---:|---:|---:|---:|---:|"]
    for i, r in enumerate(table):
        lines.append(f'| J{i+1} | {r["motion_rms_nm"]:.3f} | {r["worst_hold_rms_nm"]:.3f} | {r["peak_nm"]:.3f} | 待装配校核 | 待装配校核 |')
    lines += ["", "## 完整工况检查", "", "| 工况 | 最大跟踪误差° | 闭环位置 mm | 闭环角度° | 运动四模块接地% |",
              "|---|---:|---:|---:|---:|"]
    for name, c in report["cases"].items():
        lines.append(f'| {name} | {max(j["max_tracking_error_deg"] for j in c["joints"]):.3f} | {c["max_closure_gap_mm"]:.4f} | {c["max_closure_angle_deg"]:.4f} | {c["motion_four_modules_percent"]:.1f} |')
    checks = [v for c in report["cases"].values()
              for v in c.get("hold_quality_from_position_derivative", [])]
    if checks:
        lines += ["", f'保持稳定性：{sum(v["settled"] for v in checks)} / {len(checks)} 个工况通过速度检查。'
                  '按采样位置差分计算，各电机速度绝对值最大值的 95 分位须低于 0.03 rad/s。']
    lines += ["", "## 步长敏感性与解释", "", "| 物理频率 Hz | J1 运动 RMS | J5 运动 RMS | J8 运动 RMS | 最大峰值 |",
              "|---|---:|---:|---:|---:|"]
    for hz in report["physics_frequencies_hz"]:
        c = report["cases"][f"closed_{hz}_mass_0.150"]
        lines.append(f'| {hz} | {c["joints"][0]["motion_rms_nm"]:.3f} | {c["joints"][4]["motion_rms_nm"]:.3f} | {c["joints"][7]["motion_rms_nm"]:.3f} | {max(j["peak_nm"] for j in c["joints"]):.3f} |')
    lines += ["", "步长减半后仍会改变接触切换及部分运动 RMS，结果不能称为已完全数值收敛。表格取两档包络，50% 余量不能替代模型和接触校准。",
              "手动排除的网格交叉源于 CAD 精度，不据此判断实物干涉。当前闭环轴点未经正式装配校准，不能据此判断实物扭矩能降低多少。", "",
              "## 新增质量敏感性（最高物理频率）", "", "| 新增质量 kg | 最大运动 RMS | 最大保持 RMS | 最大峰值 |", "|---|---:|---:|---:|"]
    for mass in (0.0, .15, .30):
        key = f'closed_{max(report["physics_frequencies_hz"])}_mass_{mass:.3f}'
        if key in report["cases"]:
            c = report["cases"][key]
            lines.append(f'| {mass:.2f} | {max(j["motion_rms_nm"] for j in c["joints"]):.3f} | {max(j["worst_hold_rms_nm"] for j in c["joints"]):.3f} | {max(j["peak_nm"] for j in c["joints"]):.3f} |')
    lines += ["", "模块接地包含轮胎或 CAD 中的辅助支承件；不能将‘三只轮胎接触’直接等同于‘一个模块悬空’。",
              "不同步长仍可能改变接触切换，需结合各工况原始 NPZ 的闭环误差、跟踪误差及扭矩共同判断。",
              "瞬时峰值包含离散目标更新及闭环轨迹曲率变化带来的控制瞬态；轨迹平滑或改变控制策略后需重新测试。",
              "本模型未包含真实支架、轴承、装配弹性及电机热模型；正式电机采购应在正式闭环 CAD 和实测摩擦下复核。", "",
              "![扭矩对比](closed_loop_comparison.png)", "", "![选型表](eight_motor_selection.png)", "",
              "![闭环形态](closed_loop_poses.png)", ""]
    if (out / "closed_loop_motion.gif").exists():
        lines += ["![六秒正向与六秒反向播放](closed_loop_motion.gif)", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / f"closed_loop_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--frequencies", nargs="+", type=int, default=[1920, 3840])
    parser.add_argument("--skip-mass-sensitivity", action="store_true")
    parser.add_argument("--skip-holds", action="store_true", help="Smoke test only; excludes holding loads")
    parser.add_argument("--animation", action="store_true", help="Also export a 12-second physics replay GIF")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    print("闭环轴点待装配校准：本次只生成数值诊断，选型待复核；保留手动碰撞排除。", flush=True)
    source_hash = hashlib.sha256(DEFAULT_MODEL.read_bytes()).hexdigest() if DEFAULT_MODEL.exists() else None
    path, geometry = build_closed_loop_model()
    model = mujoco.MjModel.from_xml_path(str(path))
    ik = ClosedLoopKinematics(model)
    trajectory = ik.trajectory(601)
    phis, q = trajectory
    residual = np.array([ik.residual(row) for row in q])
    midpoint_residual = np.array([ik.residual(row) for row in (q[1:] + q[:-1]) / 2])
    np.savez_compressed(out / "compatible_trajectory.npz", phi=phis, joint_names=MORPH_NAMES, position_rad=q)
    report = {"geometry": geometry, "mujoco_version": mujoco.__version__, "source_sha256": source_hash,
              "target_interpolation": True, "motor_sizing_valid": False,
              "physics_frequencies_hz": args.frequencies,
              "trajectory": {"samples": len(phis),
                             "max_original_angle_change_deg": float(np.max(abs(q[:, :7] - np.array([angles_from_phi(p) for p in phis]))) * 180 / np.pi),
                             "max_gap_mm": float(np.linalg.norm(residual[:, :3], axis=1).max() * .2 * 1000),
                             "interpolated_midpoint_max_gap_mm": float(np.linalg.norm(midpoint_residual[:, :3], axis=1).max() * .2 * 1000),
                             "interpolated_midpoint_max_orientation_deg": float(np.linalg.norm(midpoint_residual[:, 3:], axis=1).max() * 180 / np.pi),
                             "max_orientation_error_deg": float(np.linalg.norm(residual[:, 3:], axis=1).max() * 180 / np.pi)},
              "cases": {}}
    for hz in args.frequencies:
        for closed in (False, True):
            label = f"closed_{hz}_mass_0.150" if closed else f"seven_{hz}"
            print(f"Testing {label} ...", flush=True)
            report["cases"][label] = run_case(path if closed else DEFAULT_MODEL,
                                               trajectory if closed else None, hz,
                                               out / f"{label}.npz", holds=not args.skip_holds)
            print(json.dumps(report["cases"][label], ensure_ascii=False), flush=True)
            (out / "progress.json").write_text(json.dumps(report, indent=2))
    if not args.skip_mass_sensitivity:
        for mass in (0.0, 0.30):
            mass_path, _ = build_closed_loop_model(output=out / f"model_mass_{mass:.3f}.xml", added_mass_kg=mass)
            label = f"closed_{max(args.frequencies)}_mass_{mass:.3f}"
            print(f"Testing {label} ...", flush=True)
            report["cases"][label] = run_case(mass_path, trajectory, max(args.frequencies),
                                               out / f"{label}.npz", holds=not args.skip_holds)
            print(json.dumps(report["cases"][label], ensure_ascii=False), flush=True)
    report["seven_joint_envelope"] = envelope([c for c in report["cases"].values() if not c["closed"]])
    report["eight_joint_envelope"] = envelope([c for name, c in report["cases"].items() if name.endswith("mass_0.150")])
    if source_hash and hashlib.sha256(DEFAULT_MODEL.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("Original seven-joint model changed during the experiment")
    figures(out, report)
    render_poses(path, trajectory, out / "closed_loop_poses.png")
    if args.animation:
        render_animation(path, trajectory, out / "closed_loop_motion.gif")
    write_report(out, report)
    print(f"Report: {out / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
