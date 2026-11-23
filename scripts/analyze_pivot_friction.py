"""Compare passive pivot resistance, driven-tire support and morphing loads.

All comparisons start from the same reset procedure, with resistance applied
before the initial fall. No pivot motor, position lock or imposed support is
used. Friction values are assumptions for sensitivity, not measured hardware.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

import analyze_motor_torque as study
from evaluation.mujoco_loads import JointLoadSampler


class SupportRecorder:
    def __init__(self, env, controller, sample_stride=1):
        self.env, self.controller = env, controller
        self.sampler = JointLoadSampler(env.model)
        self.stride, self.tick = sample_stride, 0
        self.rows, self.phase = [], "forward"
        self.start = float(env.data.time)
        self.module_geoms = [set(env.model.geom(f"{part}{i}_collision").id
                                 for part in ("wheel", "steering", "pivot")) for i in range(1, 5)]

    def __call__(self, env):
        self.tick += 1
        if (self.tick - 1) % self.stride:
            return
        r = self.sampler.sample(env.data)
        m, d = env.model, self.sampler.data
        tire_force, auxiliary_force = np.zeros(4), np.zeros(4)
        tire_contact, module_contact = np.zeros(4, dtype=bool), np.zeros(4, dtype=bool)
        wrench = np.zeros(6)
        for ci, c in enumerate(d.contact[:d.ncon]):
            if env.ground_geom_id not in (c.geom1, c.geom2):
                continue
            other = int(c.geom2 if c.geom1 == env.ground_geom_id else c.geom1)
            mujoco.mj_contactForce(m, d, ci, wrench)
            for i, geoms in enumerate(self.module_geoms):
                if other in geoms:
                    module_contact[i] = True
                    if other == env.wheel_geom_ids[i]:
                        tire_contact[i] = True
                        tire_force[i] += wrench[0]
                    else:
                        auxiliary_force[i] += wrench[0]
        gap = []
        for gid in env.wheel_geom_ids:
            axis = d.geom_xmat[gid].reshape(3, 3)[:, 2]
            bottom = d.geom_xpos[gid, 2] - m.geom_size[gid, 0] * np.sqrt(max(0., 1 - axis[2]**2))
            bottom -= m.geom_size[gid, 1] * abs(axis[2])
            gap.append(bottom * 1000)
        r.update(time_s=float(d.time) - self.start, phi=float(self.controller.phi), phase=self.phase,
                 tire_contacts=tire_contact, module_contacts=module_contact,
                 tire_normal_force_n=tire_force, auxiliary_normal_force_n=auxiliary_force,
                 tire_gap_mm=np.asarray(gap),
                 steering_tilt_deg=np.rad2deg(np.arccos(np.clip(d.xaxis[env.steering_ids, 2], -1, 1))))
        self.rows.append(r)

    def arrays(self):
        return {k: np.asarray([r[k] for r in self.rows]) for k in self.rows[0]}


def summarize(data):
    torque = data["torque_nm"][:, :7]
    contacts = data["tire_contacts"].sum(axis=1)
    # Contact points with nearly zero force can exist within the soft margin.
    loaded = (data["tire_normal_force_n"] > .1).sum(axis=1)
    return {"block_rms_nm": np.sqrt(np.mean(torque**2, axis=0)).tolist(),
            "block_peak_nm": np.max(np.abs(torque), axis=0).tolist(),
            "block_error_p95_deg": np.rad2deg(np.percentile(np.abs(
                data["position_rad"][:, :7] - data["target_rad"][:, :7]), 95, axis=0)).tolist(),
            "pivot_speed_p95_deg_s": np.rad2deg(np.percentile(np.abs(data["speed_rad_s"][:, 7:11]), 95, axis=0)).tolist(),
            "pivot_motor_peak_nm": float(np.max(np.abs(data["torque_nm"][:, 7:11]))),
            "pivot_friction_peak_nm": float(np.max(np.abs(data["joint_friction_nm"][:, 7:11]))),
            "four_tires_contact_fraction": float(np.mean(contacts == 4)),
            "four_tires_loaded_fraction": float(np.mean(loaded == 4)),
            "minimum_tire_contacts": int(contacts.min()),
            "all_modules_contact_fraction": float(np.mean(data["module_contacts"].sum(axis=1) == 4)),
            "max_tire_gap_mm": float(data["tire_gap_mm"].max()),
            "max_steering_tilt_deg": float(data["steering_tilt_deg"].max()),
            "balance_residual_max_nm": float(data["full_balance_residual"].max())}


def run_case(args, friction, damping):
    name = f"f{friction:g}_d{damping:g}".replace(".", "p")
    config = SimpleNamespace(model=args.model, urdf=args.urdf, control_steps=args.control_steps,
        timestep_divisor=args.timestep_divisor, block_gain_scale=30., passive_pivots=True,
        pivot_frictionloss=friction, pivot_damping=damping)
    result = {"name": name, "frictionloss_nm": friction, "damping_nms_rad": damping,
              "holds": [], "initial_conditions": "O_straight reset; 1 s free fall; Phi 1.5 controller reset"}
    for direction in ("ascending", "descending"):
        env, controller = study.make_env(config)
        try:
            grid = np.linspace(0, 3, args.static_points)
            if direction == "descending":
                grid = grid[::-1]
            study.move_to(env, controller, float(grid[0]))
            study.advance_for(env, controller, float(grid[0]), args.settle_seconds)
            recorder = SupportRecorder(env, controller, args.sample_stride)
            recorder.phase = direction
            for phi in grid:
                study.move_to(env, controller, float(phi))
                study.advance_for(env, controller, float(phi), args.settle_seconds)
                recorder.rows = []
                study.advance_for(env, controller, float(phi), args.sample_seconds, recorder)
                data = recorder.arrays()
                summary = summarize(data)
                summary.update(direction=direction, phi=float(phi),
                    pivot_angle_mean_deg=np.rad2deg(data["position_rad"][:, 7:11].mean(axis=0)).tolist(),
                    tire_normal_mean_n=data["tire_normal_force_n"].mean(axis=0).tolist(),
                    auxiliary_normal_mean_n=data["auxiliary_normal_force_n"].mean(axis=0).tolist(),
                    tire_gap_mean_mm=data["tire_gap_mm"].mean(axis=0).tolist(),
                    pivot_drift_deg=np.rad2deg(np.abs(data["position_rad"][-1, 7:11] - data["position_rad"][0, 7:11])).tolist())
                result["holds"].append(summary)
        finally:
            env.close()
    env, controller = study.make_env(config)
    try:
        study.move_to(env, controller, 0)
        study.advance_for(env, controller, 0, args.settle_seconds)
        recorder = SupportRecorder(env, controller, args.sample_stride)
        for phase, target, seconds in (("forward", 3, 6), ("hold_3", 3, 2),
                                       ("reverse", 0, 6), ("hold_0", 0, 2)):
            recorder.phase = phase
            study.advance_for(env, controller, target, seconds, recorder)
        data = recorder.arrays()
        np.savez_compressed(args.output / f"{name}_cycle.npz", **data)
        result["cycle"] = summarize(data)
        for phase in ("forward", "reverse"):
            result[phase] = summarize({k: v[data["phase"] == phase] for k, v in data.items()})
        result["physics_hz"], result["control_hz"] = 1 / env.dt, 1 / env.control_dt
        holds = result["holds"]
        result["holding_four_tires_fraction"] = float(np.mean([h["four_tires_contact_fraction"] for h in holds]))
        result["holding_four_loaded_fraction"] = float(np.mean([h["four_tires_loaded_fraction"] for h in holds]))
        result["holding_max_rms_nm"] = np.max([h["block_rms_nm"] for h in holds], axis=0).tolist()
        result["holding_max_peak_nm"] = np.max([h["block_peak_nm"] for h in holds], axis=0).tolist()
        result["holding_max_pivot_drift_deg"] = float(np.max([h["pivot_drift_deg"] for h in holds]))
        return result
    finally:
        env.close()


def plot_results(args, results):
    labels = [f"{r['frictionloss_nm']:g} / {r['damping_nms_rad']:g}" for r in results]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), layout="constrained")
    x = np.arange(len(results))
    ax = axes[0, 0]
    ax.bar(x-.2, [100*r["holding_four_tires_fraction"] for r in results], .4, label="Holding")
    ax.bar(x+.2, [100*r["cycle"]["four_tires_contact_fraction"] for r in results], .4, label="6 s cycle")
    ax.set(ylabel="Four driven tires in contact (%)", ylim=(0, 105), title="Driven-tire support")
    ax.legend()
    ax = axes[0, 1]
    ax.bar(x, [max(r["cycle"]["pivot_speed_p95_deg_s"]) for r in results], color="#279e68")
    ax.set(ylabel="Pivot speed, worst hinge p95 (deg/s)", title="Motion resistance")
    for r in results:
        data = np.load(args.output / f"{r['name']}_cycle.npz")
        mask = data["phase"] == "forward"
        axes[1, 0].plot(data["phi"][mask], np.rad2deg(data["position_rad"][mask, 9]), label=r["name"])
    axes[1, 0].set(xlabel="Phi", ylabel="RL passive pivot angle (deg)", title="Same 6 s forward command")
    axes[1, 0].legend(fontsize=9, ncol=2)
    ax = axes[1, 1]
    for j, color in ((0, "#247cbb"), (4, "#ef8525")):
        rms = [max(r["cycle"]["block_rms_nm"][j], r["holding_max_rms_nm"][j]) for r in results]
        peak = [max(r["cycle"]["block_peak_nm"][j], r["holding_max_peak_nm"][j]) for r in results]
        ax.plot(x, rms, "o-", color=color, label=f"joint{j+1} RMS/holding reference")
        ax.plot(x, peak, "x--", color=color, label=f"joint{j+1} peak")
    ax.set(ylabel="Output torque (Nm)", title="Load sensitivity; assumed friction")
    ax.legend(fontsize=9)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
        ax.set(xticks=x, xticklabels=labels, xlabel="Dry friction (Nm) / damping (Nm s/rad)")
    for ax in axes.flat:
        ax.grid(alpha=.2, axis="y")
    fig.savefig(args.output / "pivot_friction_comparison.png", dpi=240)
    fig.savefig(args.output / "pivot_friction_comparison.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=study.DEFAULT_MODEL)
    parser.add_argument("--urdf", type=Path, default=study.DEFAULT_URDF)
    parser.add_argument("--frictions", type=float, nargs="+", default=[0, .02, .05, .1, .2])
    parser.add_argument("--dampings", type=float, nargs="+", default=[.01, .05])
    parser.add_argument("--static-points", type=int, default=13)
    parser.add_argument("--settle-seconds", type=float, default=5.)
    parser.add_argument("--sample-seconds", type=float, default=1.)
    parser.add_argument("--sample-stride", type=int, default=1,
                        help="Record every physics tick by default; decimation can alias controller torque pulses")
    parser.add_argument("--control-steps", type=int, default=2)
    parser.add_argument("--timestep-divisor", type=int, default=2)
    parser.add_argument("--output", type=Path, default=study.PROJECT_ROOT / "experiments" /
                        f"pivot_friction_{datetime.now():%Y%m%d_%H%M%S}")
    args = parser.parse_args()
    if (any(not np.isfinite(x) or x < 0 for x in [*args.frictions, *args.dampings])
            or args.static_points < 2 or min(args.sample_stride, args.control_steps, args.timestep_divisor) < 1
            or any(not np.isfinite(x) or x <= 0 for x in (args.settle_seconds, args.sample_seconds))):
        parser.error("Nonnegative finite resistance, positive times/steps and static-points >= 2 required")
    args.output.mkdir(parents=True, exist_ok=False)
    settings = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings.update(created_at=datetime.now().isoformat(), mujoco_version=mujoco.__version__,
                    model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                    urdf_sha256=hashlib.sha256(args.urdf.read_bytes()).hexdigest(),
                    passive_pivots=True, motor_capacity_clamps=False, block_gain_scale=30.)
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2, allow_nan=False))
    results = []
    for damping in args.dampings:
        for friction in args.frictions:
            print(f"Testing dry friction {friction:g} Nm, damping {damping:g} Nm s/rad", flush=True)
            r = run_case(args, friction, damping)
            results.append(r)
            (args.output / "report.json").write_text(json.dumps(results, indent=2, allow_nan=False))
            print(f"  Four tires: holds {r['holding_four_tires_fraction']:.1%}, cycle "
                  f"{r['cycle']['four_tires_contact_fraction']:.1%}; "
                  f"hold RMS j1/j5 {r['holding_max_rms_nm'][0]:.3f}/{r['holding_max_rms_nm'][4]:.3f} Nm", flush=True)
    plot_results(args, results)
    lines = ["# 被动 pivot 摩擦敏感性复测", "",
             "本次参数是假设的轴承阻力，未由 CAD 或实测确定。pivot 电机输出力矩为零。",
             "每组独立重置，阻力在初始下落前生效；正、反向各 6 秒，端点各保持 2 秒；另沿两个方向测试形态保持。",
             f"物理 {results[0]['physics_hz']:g} Hz，控制 {results[0]['control_hz']:g} Hz，"
             f"每 {args.sample_stride} 个物理步采样。", "",
             "| 干摩擦 N·m | 阻尼 N·m·s/rad | 保持时四驱动轮接地 | 完整往返四驱动轮接地 | pivot 最大 p95 速度 °/s |",
             "|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['frictionloss_nm']:g} | {r['damping_nms_rad']:g} | "
                     f"{r['holding_four_tires_fraction']:.1%} | {r['cycle']['four_tires_contact_fraction']:.1%} | "
                     f"{max(r['cycle']['pivot_speed_p95_deg_s']):.1f} |")
    lines += ["", "接地率是等时采样窗口内四个驱动轮同时存在地面接触的比例，包含软接触；不等于所有轮组持续承重。",
              "JSON 同时记录每轮法向载荷、轮胎间隙、整轮组接触和保持漂移，不能用辅助支撑代替驱动轮接地统计。",
              "较大的摩擦可能留下倾斜姿态；它没有固定的回正角度。部分工况仍有驱动轮离地，不能视作实际四轮持续接地机构的已验证载荷。",
              "[对比图](pivot_friction_comparison.png) · [数据](report.json) · [参数](settings.json)", "",
              "模型说明：[MuJoCo friction constraints](https://mujoco.readthedocs.io/en/stable/modeling.html#friction)"]
    (args.output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Finished: {args.output}", flush=True)


if __name__ == "__main__":
    main()
