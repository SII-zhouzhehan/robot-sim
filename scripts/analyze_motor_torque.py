"""Headless CAD mass and joint-load study along the existing Phi formula."""
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from controllers.mujoco_control import MujocoDriveController
from envs.mujoco_model import DEFAULT_MODEL, DEFAULT_URDF
from envs.mujoco_robot_env import MujocoRobotEnv
from evaluation.mujoco_loads import JOINT_GROUPS, JOINT_NAMES, JointLoadSampler, audit_masses


GROUP_BY_JOINT = {n: g for g, names in JOINT_GROUPS.items() for n in names}


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Recorder:
    def __init__(self, env, controller):
        self.env, self.controller = env, controller
        self.sampler = JointLoadSampler(env.model)
        self.start = float(env.data.time)
        self.rows = []
        self.phase = "forward"

    def __call__(self, env):
        row = self.sampler.sample(env.data)
        d = self.sampler.data
        wheels = set()
        for contact in d.contact[:d.ncon]:
            pair = {int(contact.geom1), int(contact.geom2)}
            if env.ground_geom_id in pair:
                wheels.update(pair & env._wheel_geom_set)
        row.update(time_s=float(env.data.time) - self.start, phi=self.controller.phi,
                   phase=self.phase, wheel_contacts=len(wheels),
                   base_linear_speed_m_s=float(np.linalg.norm(d.qvel[:3])),
                   base_angular_speed_rad_s=float(np.linalg.norm(d.qvel[3:6])))
        self.rows.append(row)

    def arrays(self):
        return {key: np.asarray([r[key] for r in self.rows]) for key in self.rows[0]}


def advance_for(env, controller, target, seconds, callback=None):
    for _ in range(int(np.ceil(seconds / env.control_dt - 1e-9))):
        controller.advance(phi_target=target, physics_callback=callback)


def move_to(env, controller, target):
    advance_for(env, controller, target, abs(target - controller.phi) / controller.phi_rate)


def make_env(args):
    env = MujocoRobotEnv(args.model, urdf_path=args.urdf, control_steps=args.control_steps)
    env.model.opt.timestep /= args.timestep_divisor
    env.dt = float(env.model.opt.timestep)
    env.control_dt = env.dt * env.control_steps
    # Motor capacity is an unknown to estimate, never an input clamp.
    env.model.actuator_forcelimited[:] = 0
    env.model.jnt_actfrclimited[:] = 0
    env.configure_pivot_resistance(frictionloss=getattr(args, "pivot_frictionloss", None),
                                   damping=getattr(args, "pivot_damping", None))
    if getattr(args, "passive_pivots", True):
        env.set_passive_pivots()
    # Optional tracking sensitivity study, confined to this in-memory model.
    ids = env.block_actuators
    env.model.actuator_gainprm[ids, 0] *= args.block_gain_scale
    env.model.actuator_biasprm[ids, 1] *= args.block_gain_scale
    env.model.actuator_biasprm[ids, 2] *= np.sqrt(args.block_gain_scale)
    env.reset()
    controller = MujocoDriveController(env, source="formula", phi_rate=.5)
    controller.reset()
    return env, controller


def run_cycle(args, rate, label):
    env, controller = make_env(args)
    try:
        move_to(env, controller, 0)
        advance_for(env, controller, 0, args.settle_seconds)
        controller.phi_rate = rate
        recorder = Recorder(env, controller)
        for phase, target in (("forward", 3), ("hold_3", 3), ("reverse", 0), ("hold_0", 0)):
            recorder.phase = phase
            seconds = args.end_hold_seconds if phase.startswith("hold") else 3 / rate
            advance_for(env, controller, target, seconds, recorder)
        arrays = recorder.arrays()
        export_run(args.output, label, arrays)
        return {"label": label, "rate_phi_s": rate, "one_way_seconds": 3 / rate,
                "data": arrays, "summary": summarize(arrays, label),
                "solver_balance_max": float(arrays["full_balance_residual"].max())}
    finally:
        env.close()


def summarize(data, label):
    rows = []
    for i, name in enumerate(JOINT_NAMES):
        torque, speed = data["torque_nm"][:, i], data["speed_rad_s"][:, i]
        peak = int(np.argmax(np.abs(torque)))
        error = np.abs(data["position_rad"][:, i] - data["target_rad"][:, i])
        rows.append({
            "run": label, "joint": name, "group": GROUP_BY_JOINT[name],
            "peak_abs_nm": float(np.max(np.abs(torque))), "rms_nm": float(np.sqrt(np.mean(torque ** 2))),
            "p95_abs_nm": float(np.percentile(np.abs(torque), 95)),
            "min_nm": float(torque.min()), "max_nm": float(torque.max()),
            "peak_time_s": float(data["time_s"][peak]), "peak_phi": float(data["phi"][peak]),
            "peak_phase": str(data["phase"][peak]), "peak_speed_rpm": float(np.max(np.abs(speed)) * 60 / (2 * np.pi)),
            "peak_abs_mechanical_power_w": float(np.max(np.abs(torque * speed))),
            "peak_tracking_error_rad": None if np.isnan(error).all() else float(np.nanmax(error)),
            "p95_tracking_error_rad": None if np.isnan(error).all() else float(np.nanpercentile(error, 95)),
            "peak_joint_bending_nm": float(data["reaction_bending_nm"][:, i].max()),
            "peak_joint_radial_n": float(data["reaction_radial_n"][:, i].max()),
            "peak_joint_axial_n": float(np.max(np.abs(data["reaction_axial_n"][:, i]))),
        })
    return rows


def export_run(output, label, data):
    np.savez_compressed(output / f"{label}.npz", joint_names=JOINT_NAMES, **data)
    fields = [key for key, value in data.items() if value.ndim == 1]
    joint_fields = [key for key, value in data.items() if value.ndim == 2]
    rows = []
    for tick in range(len(data["time_s"])):
        row = {key: data[key][tick].item() for key in fields}
        for key in joint_fields:
            for i, name in enumerate(JOINT_NAMES):
                row[f"{name}_{key}"] = float(data[key][tick, i])
        rows.append(row)
    write_csv(output / f"{label}.csv", rows)


def run_holds(args):
    rows, raw = [], {}
    for direction in ("ascending", "descending"):
        env, controller = make_env(args)
        try:
            grid = np.linspace(0, 3, args.static_points)
            if direction == "descending":
                grid = grid[::-1]
            move_to(env, controller, float(grid[0]))
            advance_for(env, controller, float(grid[0]), args.settle_seconds)
            recorder = Recorder(env, controller)
            for phi in grid:
                move_to(env, controller, float(phi))
                advance_for(env, controller, float(phi), args.settle_seconds)
                recorder.rows = []
                recorder.phase = direction
                advance_for(env, controller, float(phi), args.static_sample_seconds, recorder)
                data = recorder.arrays()
                raw[f"{direction}_phi_{phi:.3f}"] = data
                block_error = np.abs(data["position_rad"][:, :7] - data["target_rad"][:, :7])
                block_speed = np.abs(data["speed_rad_s"][:, :7])
                quasi_static = (np.max(np.percentile(block_speed, 95, axis=0)) < .05
                                and np.percentile(data["base_linear_speed_m_s"], 95) < .02
                                and np.percentile(data["base_angular_speed_rad_s"], 95) < .05)
                tracking_ok = np.max(np.percentile(block_error, 95, axis=0)) < .05
                for i, name in enumerate(JOINT_NAMES):
                    torque = data["torque_nm"][:, i]
                    tracking_error = np.abs(data["position_rad"][:, i] - data["target_rad"][:, i])
                    rows.append({
                        "direction": direction, "phi": float(phi), "joint": name, "group": GROUP_BY_JOINT[name],
                        "mean_nm": float(torque.mean()), "rms_nm": float(np.sqrt(np.mean(torque ** 2))),
                        "peak_abs_nm": float(np.max(np.abs(torque))),
                        "gravity_mean_nm": float(data["gravity_nm"][:, i].mean()),
                        "speed_p95_rad_s": float(np.percentile(np.abs(data["speed_rad_s"][:, i]), 95)),
                        "tracking_error_p95_rad": (None if np.isnan(tracking_error).all() else
                                                   float(np.nanpercentile(tracking_error, 95))),
                        "robot_quasi_static": bool(quasi_static), "block_tracking_within_0_05_rad": bool(tracking_ok),
                        "min_wheel_contacts": int(data["wheel_contacts"].min()),
                        "peak_bending_nm": float(data["reaction_bending_nm"][:, i].max()),
                        "peak_radial_n": float(data["reaction_radial_n"][:, i].max()),
                        "solver_balance_max": float(data["full_balance_residual"].max()),
                    })
                print(f"  Hold {direction}: Phi={phi:.2f}, quasi-static={quasi_static}, "
                      f"shape error p95={np.max(np.percentile(block_error, 95, axis=0)):.3f} rad", flush=True)
            # Keep every holding-window sample, including vector reaction wrenches.
            combined = {key: np.concatenate([d[key] for d in raw.values() if d["phase"][0] == direction])
                        for key in recorder.rows[0]}
            export_run(args.output, f"holds_{direction}", combined)
        finally:
            env.close()
    write_csv(args.output / "holding_summary.csv", rows)
    return rows


def save_figure(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=170)
    fig.savefig(output / f"{name}.svg")
    plt.close(fig)


def make_plots(args, runs, holds, masses):
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .22})
    baseline = min(runs, key=lambda r: abs(r["rate_phi_s"] - .5))
    d = baseline["data"]
    fig, axes = plt.subplots(8, 1, figsize=(12, 14), sharex=True, layout="constrained")
    axes[0].plot(d["time_s"], d["phi"], color="black")
    axes[0].set(ylabel="Phi", ylim=(-.1, 3.1))
    for i, ax in enumerate(axes[1:]):
        ax.plot(d["time_s"], d["torque_nm"][:, i], lw=1)
        ax.set(ylabel=f"{JOINT_NAMES[i]}\nN m")
    axes[-1].set_xlabel("Simulation time (s)")
    fig.suptitle(f"Morphing motor output torque | {baseline['one_way_seconds']:g} s each direction\n"
                 f"Zero driving command; torque limits disabled; gain scale {args.block_gain_scale:g}")
    save_figure(fig, args.output, "torque_time")

    fig, axes = plt.subplots(4, 2, figsize=(13, 13), sharex=True, layout="constrained")
    for i, ax in enumerate(axes.flat):
        if i == 7:
            ax.axis("off")
            continue
        for run in runs:
            data = run["data"]
            color = {0.25: "tab:blue", .5: "tab:orange", 1.: "tab:green"}.get(run["rate_phi_s"])
            for phase, style in (("forward", "-"), ("reverse", "--")):
                mask = data["phase"] == phase
                ax.plot(data["phi"][mask], data["torque_nm"][mask, i], style, color=color, lw=.8,
                        label=f"{run['one_way_seconds']:g} s {phase}")
        ax.set(title=JOINT_NAMES[i], ylabel="Motor torque (N m)", xlabel="Commanded Phi")
    axes.flat[7].legend(*axes.flat[0].get_legend_handles_labels(), loc="center")
    fig.suptitle("Motor torque vs Phi | different speeds and folding directions")
    save_figure(fig, args.output, "torque_vs_phi")

    fig, axes = plt.subplots(4, 2, figsize=(13, 13), sharex=True, layout="constrained")
    for i, ax in enumerate(axes.flat):
        if i == 7:
            ax.axis("off")
            continue
        for direction, color in (("ascending", "tab:blue"), ("descending", "tab:orange")):
            rows = sorted([r for r in holds if r["joint"] == JOINT_NAMES[i] and r["direction"] == direction],
                          key=lambda r: r["phi"])
            ax.plot([r["phi"] for r in rows], [r["mean_nm"] for r in rows], "o-", color=color,
                    label=f"Holding mean ({direction})")
            bad = [r for r in rows if not r["robot_quasi_static"]]
            ax.scatter([r["phi"] for r in bad], [r["mean_nm"] for r in bad], marker="x", s=80,
                       color="red", label="Not settled" if direction == "ascending" else None)
        ax.set(title=JOINT_NAMES[i], ylabel="Mean holding torque (N m)", xlabel="Commanded Phi")
    axes.flat[7].legend(*axes.flat[0].get_legend_handles_labels(), loc="center")
    fig.suptitle("Holding load after settling | ground-supported robot\n"
                 f"Mean of {args.static_sample_seconds:g} s window; finite servo position error is retained")
    save_figure(fig, args.output, "holding_torque")

    groups = ("steering", "wheel") if getattr(args, "passive_pivots", True) else ("pivot", "steering", "wheel")
    fig, axes = plt.subplots(len(groups), 2, figsize=(13, 4 * len(groups)), layout="constrained")
    for row, group in enumerate(groups):
        for name in JOINT_GROUPS[group]:
            index = JOINT_NAMES.index(name)
            line, = axes[row, 0].plot(d["time_s"], d["torque_nm"][:, index], lw=.8, label=name)
            for direction, style in (("ascending", "-"), ("descending", "--")):
                points = sorted([r for r in holds if r["joint"] == name and r["direction"] == direction],
                                key=lambda r: r["phi"])
                axes[row, 1].plot([r["phi"] for r in points], [r["mean_nm"] for r in points], style,
                                  color=line.get_color(), label=f"{name} {direction}")
        axes[row, 0].set(title=f"{group}: dynamic cycle", ylabel="Torque (N m)", xlabel="Time (s)")
        axes[row, 1].set(title=f"{group}: holding mean", ylabel="Torque (N m)", xlabel="Phi")
        axes[row, 0].legend(ncol=4, fontsize=8)
        axes[row, 1].legend(ncol=2, fontsize=8)
    fig.suptitle(f"Other {4 * len(groups)} motor joints | stationary folding / holding only")
    save_figure(fig, args.output, "auxiliary_joint_torques")

    worst = max(range(7), key=lambda i: np.max(np.abs(d["torque_nm"][:, i])))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, layout="constrained")
    for key, label, sign in (("gravity_nm", "Gravity", 1), ("inertia_nm", "M qacc", 1),
                             ("velocity_bias_nm", "Coriolis / centrifugal", 1),
                             ("constraint_nm", "Minus contact / limit constraint", -1),
                             ("passive_nm", "Minus passive", -1)):
        axes[0].plot(d["time_s"], sign * d[key][:, worst], lw=.8, label=label)
    axes[0].plot(d["time_s"], d["torque_nm"][:, worst], "k", lw=1.2, label="Motor output")
    axes[0].set(ylabel="Generalized torque (N m)", title=f"{JOINT_NAMES[worst]}: dynamic force balance")
    axes[0].legend(ncol=2)
    axes[1].plot(d["time_s"], d["torque_nm"][:, worst], label="Motor-axis torque")
    axes[1].plot(d["time_s"], d["reaction_bending_nm"][:, worst], label="Joint bending moment")
    axes[1].set(ylabel="N m", xlabel="Time (s)", title="Bending loads size bearings / structure")
    axes[1].legend()
    save_figure(fig, args.output, "load_components")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, layout="constrained")
    for run in runs:
        data = run["data"]
        mask = data["phase"] == "forward"
        axes[0].plot(data["phi"][mask], np.max(np.abs(data["position_rad"][mask, :7]
                                                   - data["target_rad"][mask, :7]), axis=1) * 180 / np.pi,
                     label=f"{run['one_way_seconds']:g} s")
        axes[1].plot(data["phi"][mask], np.max(np.abs(data["torque_nm"][mask, :7]), axis=1),
                     label=f"{run['one_way_seconds']:g} s")
    axes[0].set(ylabel="Max block error (deg)", title="Tracking error during forward folding")
    axes[1].set(ylabel="Max motor torque (N m)", xlabel="Commanded Phi")
    axes[0].legend()
    save_figure(fig, args.output, "tracking_error")

    labels = [r["link"] for r in masses["links"]]
    values = [r["mujoco_mass_kg"] for r in masses["links"]]
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    ax.bar(labels, values)
    ax.tick_params(axis="x", rotation=60)
    ax.set(ylabel="Mass (kg)", title=f"Imported CAD link masses | total {masses['mujoco_total_mass_kg']:.3f} kg")
    save_figure(fig, args.output, "link_masses")


def write_report(args, runs, holds, masses, env):
    summary = [row for run in runs for row in run["summary"]]
    write_csv(args.output / "joint_summary.csv", summary)
    mass_csv = []
    for row in masses["links"]:
        flat = {k: v for k, v in row.items() if not isinstance(v, (dict, list))}
        flat.update({f"com_{axis}_m": row["urdf_com_xyz_m"][i] for i, axis in enumerate("xyz")})
        flat.update(row["urdf_inertia_kg_m2"])
        mass_csv.append(flat)
    write_csv(args.output / "link_masses.csv", mass_csv)
    (args.output / "link_masses.json").write_text(json.dumps(masses, indent=2), encoding="utf-8")
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    m = env.model
    report = {
        "created_at": datetime.now().isoformat(), "mujoco_version": mujoco.__version__,
        "settings": settings, "model_sha256": hashlib.sha256(env.model_path.read_bytes()).hexdigest(),
        "urdf_sha256": hashlib.sha256(args.urdf.read_bytes()).hexdigest(),
        "physics_timestep_s": env.dt, "controller_timestep_s": env.control_dt,
        "block_position_kp": m.actuator_gainprm[env.block_actuators, 0].tolist(),
        "block_velocity_kv": (-m.actuator_biasprm[env.block_actuators, 2]).tolist(),
        "motor_torque_limits_enabled": False,
        "pivot_actuation": "passive" if getattr(args, "passive_pivots", True) else "active_hold_legacy_proxy",
        "gravity_m_s2": m.opt.gravity.tolist(), "model_mass_kg": masses["mujoco_total_mass_kg"],
        "collision_exclusion_count": int(m.nexclude),
        "joint_mapping": [{"joint": n, "parent_body": m.body(int(m.body_parentid[int(m.jnt_bodyid[jid])])).name,
                           "child_body": m.body(int(m.jnt_bodyid[jid])).name,
                           "axis_local": m.jnt_axis[jid].tolist(), "damping_nms_rad": float(m.dof_damping[m.jnt_dofadr[jid]]),
                           "frictionloss_nm": float(m.dof_frictionloss[m.jnt_dofadr[jid]]),
                           "armature_kg_m2": float(m.dof_armature[m.jnt_dofadr[jid]])}
                          for n in JOINT_NAMES for jid in [env.joint_name_to_id[n]]],
        "cycles": [{**{k: v for k, v in run.items() if k != "data"},
                    "minimum_wheel_contacts": int(run["data"]["wheel_contacts"].min()),
                    "fraction_below_two_wheel_contacts": float(np.mean(run["data"]["wheel_contacts"] < 2))}
                   for run in runs],
        "holding": holds,
        "limitations": [
            "Joint-output torque, not motor-shaft torque before a gearbox; gear ratios are currently 1.",
            "Actual closed-loop trajectories with finite tracking error; not exact-path inverse dynamics or a minimum torque solution.",
            "Existing CAD mass/inertia, flat ground, zero commanded wheel velocity, no additional payload.",
            "Bias gravity alone is not required motor torque: ground contact and joint limits carry loads.",
            "Reaction moments/forces include loads for bearings and structure, distinct from motor-axis output.",
            "RMS is for this recorded cycle including end holds; continuous ratings require the intended duty cycle and cooling.",
            "CAD material density, omitted hardware, friction, gearbox efficiency, backlash, wiring and real collision geometry need validation.",
            "Finite-timestep compliant contact peaks are approximate; timestep/contact sensitivity must be checked before final motor sizing.",
            "Piecewise-linear Phi formula has velocity corners at Phi 1 and 2; peaks depend on trajectory and servo tuning.",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# 当前建模的电机载荷初步测算", "", "这是现有公式、控制器及地面接触条件下的关节输出端载荷估算。",
             "所有执行器扭矩限幅已关闭；未添加载荷或改动模型质量；驱动速度指令为零，原 PID 的零速死区返回零轮速目标。", "",
             ("pivot（joint8/11/14/17）为重力被动关节，执行器驱动力矩为零，不参与电机选型；保留模型关节限位和轴承阻尼。"
              if getattr(args, "passive_pivots", True) else "使用历史主动 pivot 保持代理，与重力被动结构不同。"), "",
             f"pivot 干摩擦（N·m）{m.dof_frictionloss[m.jnt_dofadr[env.pivot_ids]].tolist()}，"
             f"黏性阻尼（N·m·s/rad）{m.dof_damping[m.jnt_dofadr[env.pivot_ids]].tolist()}。这些是未校准的试验参数。", "",
             f"变形关节增益倍率 {args.block_gain_scale:g}，物理采样 {1 / env.dt:g} Hz，控制更新 {1 / env.control_dt:g} Hz。所有变动仅作用于测算时的内存模型。", "",
             f"模型总质量 **{masses['mujoco_total_mass_kg']:.6f} kg**，8 个 block 合计 **{masses['group_mass_kg']['block']:.6f} kg**。",
             "URDF 的 `<inertial>` 包含质量（kg）、质心（m）和完整惯量（kg·m²）；STL 本身不保存质量或材料。",
             "导入时采用 URDF 惯量，不按碰撞网格重新计算质量。实际材料和遗漏的电机/线束需在 CAD 中复核。", "",
             "## 正向、反向完整变形", "", "| 单向时长 | 最大变形扭矩 | 关节 | Φ / 阶段 | 最大变形角误差 |", "|---|---:|---|---|---:|"]
    for run in runs:
        blocks = run["summary"][:7]
        top = max(blocks, key=lambda r: r["peak_abs_nm"])
        error = max(r["peak_tracking_error_rad"] for r in blocks) * 180 / np.pi
        lines.append(f"| {run['one_way_seconds']:g} s "
                     f"| {top['peak_abs_nm']:.3f} N·m | {top['joint']} | {top['peak_phi']:.3f} / {top['peak_phase']} | {error:.2f}° |")
    baseline = min(runs, key=lambda r: abs(r["rate_phi_s"] - .5))
    lines += ["", f"## {baseline['one_way_seconds']:g} 秒工况的 7 个变形关节", "",
              "| 关节 | 峰值 N·m | 周期 RMS N·m | 95% 分位 N·m | 最大转速 rpm |", "|---|---:|---:|---:|---:|"]
    for r in baseline["summary"][:7]:
        lines.append(f"| {r['joint']} | {r['peak_abs_nm']:.3f} | {r['rms_nm']:.3f} | {r['p95_abs_nm']:.3f} | {r['peak_speed_rpm']:.2f} |")
    lines += ["", "最大误差包含运动跟踪滞后；扭矩是实际执行轨迹的估算。仅有 Φ 公式不能唯一确定支撑、受力或动态扭矩。",
              "若最大跟踪误差较大，曲线不能代表严格按公式几何运动的全部需求。应对轨迹、增益和控制频率做敏感性检查。", ""]
    for run in runs:
        below = int(np.sum(run["data"]["wheel_contacts"] < 2))
        if below:
            lines.append(f"{run['one_way_seconds']:g} 秒工况有 {below * env.dt:.4f} 秒少于两个轮子接触地面；需结合其他部件接触与支撑稳定性评估。")
    configurations = [r for r in holds if r["joint"] == "joint1"]
    unsettled = [f"{r['direction']} Φ={r['phi']:g}" for r in configurations if not r["robot_quasi_static"]]
    inaccurate = [f"{r['direction']} Φ={r['phi']:g}" for r in configurations if not r["block_tracking_within_0_05_rad"]]
    lines += ["", "## 保持载荷与结构载荷", "",
              f"每个形态等待 {args.settle_seconds:g} 秒，再记录 {args.static_sample_seconds:g} 秒；分别从两端接近。",
              "保持曲线是实际姿态的平均驱动力矩，不是理论最小保持力矩。接触、限位及角度误差均保留。",
              "未充分稳定的采样点：" + ("；".join(unsettled) if unsettled else "无（按本脚本速度阈值）") + "。",
              "角度误差 95% 分位超过 0.05 rad 的采样点：" + ("；".join(inaccurate) if inaccurate else "无") + "。",
              "`reaction_*` 是父子刚体传递载荷，力矩已移到关节轴心。弯矩和径向/轴向力用于轴承、连接件和结构评估；不能全部计为电机扭矩。",
              "`constraint_nm` 同时包括地面/自碰撞接触与关节限位的广义力，不能全部解读为单个关节反力。", "",
              "## 怎样用于选型", "", "先用 `joint_summary.csv` 看关节峰值、RMS 和转速，再结合 `holding_summary.csv` 的长时间保持负载。",
              "若减速比为 N、传动效率为 η，粗估电机轴扭矩 τ_motor≈τ_joint/(Nη)，电机转速 ω_motor=Nω_joint。还需计入减速器与电机的惯量、摩擦和安全余量。",
              "周期 RMS 只对应这里的变形/保持时间；长期保持、刹车/自锁、温升与真实工作循环要单独校核。",
              "当前控制器有有限刚度；若角度误差过大，应先改善轨迹与控制再复测。Φ=1、2 的分段公式改变角速度，平滑加减速后峰值可能变化。",
              ("记录 19 个活动关节：15 个电机轴与 4 个被动 pivot；base 与 block1 之间 joint0 为固定连接。"
               if getattr(args, "passive_pivots", True) else "记录 19 个驱动关节，含 4 个历史主动 pivot 保持代理；joint0 为固定连接。"), "",
              "## 可复现数据", "", "- `torque_time.png`、`torque_vs_phi.png`：7 个变形关节的全过程扭矩。",
              "- `holding_torque.png`：地面支撑下的保持曲线。",
              "- `auxiliary_joint_torques.png`：转向与轮毂电机扭矩；默认排除被动 pivot。",
              "- `tracking_error.png`：不同变形速度的跟踪误差。",
              "- `load_components.png`：重力、惯性、约束及结构弯矩对照。",
              "- `link_masses.csv/json`：每个 link 的质量、质心和惯量。",
              "- 每个工况 `.csv`：每个物理步、19 个关节的标量载荷；`.npz` 另含完整世界坐标力/力矩向量。",
              "- `report.json`：设置、模型哈希、关节映射、所有摘要与局限。", "",
              "所有输出保存在 Git 忽略的 experiments 中。", "",
              "参考：[MuJoCo 动力学](https://mujoco.readthedocs.io/en/stable/computation/index.html)", ""]
    (args.output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--rates", type=float, nargs="+", default=[.25, .5, 1.])
    parser.add_argument("--settle-seconds", type=float, default=5.)
    parser.add_argument("--end-hold-seconds", type=float, default=2.)
    parser.add_argument("--static-points", type=int, default=13)
    parser.add_argument("--static-sample-seconds", type=float, default=1.)
    parser.add_argument("--control-steps", type=int, default=4,
                        help="Physics ticks per controller update (default: original 60 Hz controller)")
    parser.add_argument("--timestep-divisor", type=int, default=1,
                        help="Divide the model physics timestep for a numerical sensitivity check")
    parser.add_argument("--block-gain-scale", type=float, default=1.,
                        help="Optional position-servo gain study; kp scaled by this, kv by its square root")
    parser.add_argument("--passive-pivots", action=argparse.BooleanOptionalAction, default=True,
                        help="Gravity-driven pivot hinges (default); --no-passive-pivots reproduces the legacy active-hold proxy")
    parser.add_argument("--pivot-frictionloss", type=float,
                        help="Dry friction torque per passive pivot in Nm; omitted: use environment default")
    parser.add_argument("--pivot-damping", type=float,
                        help="Viscous resistance per pivot in Nm*s/rad; omitted: use environment default")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "experiments" / f"motor_sizing_{datetime.now():%Y%m%d_%H%M%S}")
    args = parser.parse_args()
    if any(v is not None and (not np.isfinite(v) or v < 0)
           for v in (args.pivot_frictionloss, args.pivot_damping)):
        parser.error("Pivot frictionloss/damping must be finite and nonnegative")
    if (any(r <= 0 or not np.isfinite(r) for r in args.rates) or args.static_points < 2
            or args.control_steps < 1 or args.timestep_divisor < 1
            or not all(np.isfinite(v) and v > 0 for v in (args.settle_seconds, args.end_hold_seconds,
                                                         args.static_sample_seconds, args.block_gain_scale))):
        parser.error("Rates and times must be finite/positive; static-points must be >= 2")
    args.model, args.urdf, args.output = args.model.resolve(), args.urdf.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    env, _ = make_env(args)
    try:
        masses = audit_masses(env.model, args.urdf)
        print(f"Model mass {masses['mujoco_total_mass_kg']:.6f} kg; output {args.output}", flush=True)
        runs = []
        for rate in args.rates:
            label = f"cycle_{3 / rate:g}s".replace(".", "p")
            print(f"Cycle: {3 / rate:g} s each direction", flush=True)
            runs.append(run_cycle(args, rate, label))
        holds = run_holds(args)
        make_plots(args, runs, holds, masses)
        write_report(args, runs, holds, masses, env)
        print(f"Finished: {args.output / 'report.md'}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
