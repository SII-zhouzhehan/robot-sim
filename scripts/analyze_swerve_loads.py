"""Headless swerve sizing on flat ground and real inclined planes.

All motor capacity and wheel-speed clamps are disabled in memory. The sizing
run uses a torque-driven wheel PI servo so a zero-speed command can hold a
slope; this is an analysis controller, not a change to the application.
The four pivot hinges are gravity-driven and have no actuator force by default.
"""
import argparse
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
from controllers.mujoco_control import MujocoDriveController, angles_from_phi
from controllers.swerve_kinematics import SwerveDriveController
from envs.mujoco_model import DEFAULT_MODEL, DEFAULT_URDF
from envs.mujoco_robot_env import MujocoRobotEnv
from evaluation.mujoco_loads import JOINT_NAMES, JointLoadSampler, incline_robot_and_ground
from analyze_motor_torque import export_run, save_figure, write_csv


STEER_SLICE, WHEEL_SLICE = slice(11, 15), slice(15, 19)


def smooth_fraction(t, duration):
    u = float(np.clip(t / duration, 0, 1))
    return u ** 3 * (10 - 15 * u + 6 * u ** 2)


class SwerveLoadRun:
    def __init__(self, args, phi, slope):
        self.args, self.phi, self.slope = args, phi, slope
        self.env = env = MujocoRobotEnv(args.model, urdf_path=args.urdf, control_steps=args.control_steps)
        env.model.opt.timestep /= args.timestep_divisor
        env.dt = float(env.model.opt.timestep)
        env.control_dt = env.dt * env.control_steps
        env.model.actuator_forcelimited[:] = 0
        env.model.jnt_actfrclimited[:] = 0
        env.configure_pivot_resistance(frictionloss=getattr(args, "pivot_frictionloss", None),
                                       damping=getattr(args, "pivot_damping", None))
        env.model.actuator_ctrllimited[env.wheel_actuators] = 0
        env.MAX_WHEEL_OMEGA = 1e6  # Remove the historical 160 rpm software clip.
        ids = env.block_actuators
        env.model.actuator_gainprm[ids, 0] *= args.block_gain_scale
        env.model.actuator_biasprm[ids, 1] *= args.block_gain_scale
        env.model.actuator_biasprm[ids, 2] *= np.sqrt(args.block_gain_scale)
        ids = env.steering_actuators
        env.model.actuator_gainprm[ids, 0] *= args.steering_gain_scale
        env.model.actuator_biasprm[ids, 1] *= args.steering_gain_scale
        env.model.actuator_biasprm[ids, 2] *= np.sqrt(args.steering_gain_scale)
        self.passive_pivots = getattr(args, "passive_pivots", True)
        if self.passive_pivots:
            # Zero both gain and bias: a zero position command alone would still
            # apply a restoring torque. Keep hinge limits and bearing damping.
            env.set_passive_pivots()
        env.reset()
        morph = MujocoDriveController(env, source="formula")
        morph.reset()
        for _ in range(int(np.ceil((abs(phi - 1.5) / .5 + 5) / env.control_dt))):
            morph.advance(phi_target=phi)
        self.rotation = np.eye(3)
        self.swerve = SwerveDriveController(env.WHEEL_RADIUS)
        self.pi_integral = np.zeros(4)
        self.speed_target = np.zeros(4)
        self.command = np.zeros(3)
        self.phase = "prepare"
        self.rows = []
        self.sampler = JointLoadSampler(env.model)
        self.target_blocks = angles_from_phi(phi)
        self.steering_target = env.get_steering_positions()
        self.steering_rate_limit = np.deg2rad(getattr(args, "steering_rate_limit_deg_s", 337.5))
        # PI wheel torque with implicit velocity feedback for numerical stability.
        # tau = kp * (omega_target - omega) + ki * integral(error).
        # Put kp in MuJoCo's velocity actuator so implicitfast integrates its
        # damping; an explicit torque callback is unstable for these tiny wheels.
        env.model.actuator_gainprm[env.wheel_actuators, :] = 0
        env.model.actuator_gainprm[env.wheel_actuators, 0] = args.wheel_kp
        env.model.actuator_biasprm[env.wheel_actuators, :] = 0
        env.model.actuator_biasprm[env.wheel_actuators, 2] = -args.wheel_kp
        env.model.actuator_biastype[env.wheel_actuators] = mujoco.mjtBias.mjBIAS_AFFINE
        self.start = float(env.data.time)

    def wheel_commands(self, command):
        e = self.env
        if np.max(np.abs(command)) < 1e-12:
            return self.steering_target.copy(), np.zeros(4)
        current = e.get_steering_positions()
        self.swerve.last_angles.update(dict(zip(e.STEER_CORNER_ORDER, current)))
        points = (e.data.xipos[e.steering_body_ids] - e.get_support_center()) @ self.rotation
        normal = self.rotation[:, 2]
        rolling = np.cross(e.data.xaxis[e.wheel_ids], normal) @ self.rotation
        zero = np.arctan2(rolling[:, 1], rolling[:, 0]) - current
        corners = e.STEER_CORNER_ORDER
        targets = self.swerve.compute_swerve_kinematics(*command,
                    dict(zip(corners, points[:, :2])), dict(zip(corners, zero)))
        return np.array([targets[c]["angle"] for c in corners]), np.array([targets[c]["speed"] for c in corners])

    def physics_tick(self, env):
        error = self.speed_target - env.joint_velocities(env.wheel_ids)
        self.pi_integral += error * env.dt
        env.data.ctrl[env.wheel_actuators] = (self.speed_target + self.args.wheel_ki / self.args.wheel_kp * self.pi_integral
                                             if self.args.wheel_kp else np.zeros(4))
        if self.phase == "prepare":
            return
        row = self.sampler.sample(env.data)
        d = self.sampler.data
        velocity = np.zeros(6)
        speeds = []
        for body in env.steering_body_ids:
            mujoco.mj_objectVelocity(env.model, d, mujoco.mjtObj.mjOBJ_BODY, int(body), velocity, 0)
            speeds.append(velocity[3:].copy())
        mean = np.mean(speeds, axis=0) @ self.rotation
        mujoco.mj_objectVelocity(env.model, d, mujoco.mjtObj.mjOBJ_BODY, env.base_body_id, velocity, 0)
        actual = np.array([mean[0], mean[1], velocity[:3] @ self.rotation[:, 2]])
        normal_force = np.zeros(4)
        wheel_contacts = set()
        contact_force = np.zeros(6)
        for i in range(d.ncon):
            contact = d.contact[i]
            pair = {int(contact.geom1), int(contact.geom2)}
            if env.ground_geom_id not in pair:
                continue
            for corner, gid in enumerate(env.wheel_geom_ids):
                if int(gid) in pair:
                    wheel_contacts.add(corner)
                    mujoco.mj_contactForce(env.model, d, i, contact_force)
                    normal_force[corner] += abs(float(contact_force[0]))
        position = env.get_support_center() @ self.rotation
        row.update(time_s=float(d.time) - self.start, phi=self.phi, slope_deg=self.slope, phase=self.phase,
                   wheel_contacts=len(wheel_contacts),
                   command_vx_m_s=float(self.command[0]), command_vy_m_s=float(self.command[1]),
                   command_omega_rad_s=float(self.command[2]),
                   actual_vx_m_s=float(actual[0]), actual_vy_m_s=float(actual[1]), actual_omega_rad_s=float(actual[2]),
                   plane_position_x_m=float(position[0]), plane_position_y_m=float(position[1]),
                   block_tracking_error_rad=float(np.max(np.abs(row["position_rad"][:7] - self.target_blocks))))
        # Keep export_run's 19-joint-column contract; these apply only to wheels.
        row["wheel_target_rad_s"] = np.r_[np.full(15, np.nan), self.speed_target]
        row["wheel_normal_force_n"] = np.r_[np.zeros(15), normal_force]
        self.rows.append(row)

    def step(self, steering, speed, command=(0, 0, 0)):
        requested = np.asarray(steering)
        if self.steering_rate_limit:
            delta = self.steering_rate_limit * self.env.control_dt
            steering = self.steering_target + np.clip(requested - self.steering_target, -delta, delta)
        self.steering_target = np.asarray(steering).copy()
        self.speed_target[:] = speed
        self.command[:] = command
        self.env.step({"blocks": self.target_blocks, "steering": steering, "velocity": np.zeros(4)},
                      physics_callback=self.physics_tick)

    def prealign(self, heading):
        e = self.env
        old = e.get_steering_positions()
        target, _ = self.wheel_commands((np.cos(heading), np.sin(heading), 0))
        for i in range(int(np.ceil(self.args.steer_seconds / e.control_dt))):
            t = (i + 1) * e.control_dt
            self.step(old + smooth_fraction(t, self.args.steer_seconds) * (target - old), np.zeros(4))
        for _ in range(round(1 / e.control_dt)):
            self.step(target, np.zeros(4))

    def prepare_slope(self, heading=0):
        self.prealign(heading)
        self.rotation = incline_robot_and_ground(self.env, self.slope)
        # Let support geometry settle physically, including gravity-driven pivots.
        for _ in range(round(3 / self.env.control_dt)):
            self.step(self.steering_target, np.zeros(4))
        self.start = float(self.env.data.time)

    def drive(self, mode):
        e = self.env
        self.prepare_slope(np.pi / 2 if mode == "lateral" else 0)
        if mode == "spin":
            target, _ = self.wheel_commands((0, 0, self.args.yaw_rate))
            old = e.get_steering_positions()
            for i in range(round(self.args.steer_seconds / e.control_dt)):
                self.step(old + smooth_fraction((i + 1) * e.control_dt, self.args.steer_seconds) * (target - old), np.zeros(4))
            self.start = float(e.data.time)
        for sign, label in ((1, "outbound"), (-1, "return")):
            if self.slope and sign < 0:
                label = "downhill"
            elif self.slope:
                label = "uphill"
            vector = {"straight": (self.args.speed, 0, 0), "lateral": (0, self.args.speed, 0),
                      "spin": (0, 0, self.args.yaw_rate)}[mode]
            vector = sign * np.asarray(vector)
            for segment, seconds in (("start", self.args.ramp_seconds), ("cruise", self.args.cruise_seconds),
                                     ("brake", self.args.ramp_seconds), ("hold", self.args.hold_seconds)):
                self.phase = f"{label}_{segment}"
                for i in range(int(np.ceil(seconds / e.control_dt))):
                    u = smooth_fraction((i + 1) * e.control_dt, seconds)
                    factor = u if segment == "start" else 1 - u if segment == "brake" else 1 if segment == "cruise" else 0
                    command = factor * vector
                    steering, speed = self.wheel_commands(command)
                    self.step(steering, speed, command)

    def steer_test(self, duration, braked):
        e = self.env
        self.prepare_slope()
        old = e.get_steering_positions()
        start_gain = self.args.wheel_kp, self.args.wheel_ki
        if not braked:
            self.args.wheel_kp, self.args.wheel_ki = 0., 0.
            e.model.actuator_gainprm[e.wheel_actuators, 0] = 0
            e.model.actuator_biasprm[e.wheel_actuators, 2] = 0
        try:
            for phase, target in (("turn_90", old + np.pi / 2), ("turn_back", old)):
                initial = e.get_steering_positions()
                self.phase = phase
                for i in range(int(np.ceil(duration / e.control_dt))):
                    self.step(initial + smooth_fraction((i + 1) * e.control_dt, duration) * (target - initial), np.zeros(4))
                self.phase = "steering_hold"
                for _ in range(round(self.args.hold_seconds / e.control_dt)):
                    self.step(target, np.zeros(4))
        finally:
            self.args.wheel_kp, self.args.wheel_ki = start_gain
            e.model.actuator_gainprm[e.wheel_actuators, 0] = start_gain[0]
            e.model.actuator_biasprm[e.wheel_actuators, 2] = -start_gain[0]

    def direction_change(self):
        e = self.env
        self.prepare_slope()
        vectors = [np.array(v) * self.args.speed for v in ((1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 0))]
        previous = np.zeros(3)
        for index, target in enumerate(vectors):
            self.phase = f"direction_change_{index}"
            for i in range(round(self.args.steer_seconds / e.control_dt)):
                u = smooth_fraction((i + 1) * e.control_dt, self.args.steer_seconds)
                command = previous * (1 - u) + target * u
                steering, speed = self.wheel_commands(command)
                self.step(steering, speed, command)
            self.phase = f"direction_cruise_{index}"
            for _ in range(round(self.args.cruise_seconds / e.control_dt)):
                steering, speed = self.wheel_commands(target)
                self.step(steering, speed, target)
            previous = target

    def finish(self, name):
        data = {k: np.asarray([r[k] for r in self.rows]) for k in self.rows[0]}
        export_run(self.args.output, name, data)
        self.env.close()
        return data


def summarize_run(name, data):
    rows = []
    cruise = np.array(["cruise" in p for p in data["phase"]])
    hold = np.array(["hold" in p for p in data["phase"]])
    active = ~hold
    speed_errors = np.hypot(data["actual_vx_m_s"] - data["command_vx_m_s"],
                            data["actual_vy_m_s"] - data["command_vy_m_s"])
    for group, indices in (("steering", range(11, 15)), ("wheel", range(15, 19))):
        for corner, i in enumerate(indices):
            torque, speed = data["torque_nm"][:, i], data["speed_rad_s"][:, i]
            peak = int(np.argmax(np.abs(torque)))
            error = data["position_rad"][:, i] - data["target_rad"][:, i]
            row = {"run": name, "phi": float(data["phi"][0]), "slope_deg": float(data["slope_deg"][0]),
                   "group": group, "corner": ("RR", "FR", "RL", "FL")[corner], "joint": JOINT_NAMES[i],
                   "peak_nm": float(np.max(np.abs(torque))), "rms_nm": float(np.sqrt(np.mean(torque ** 2))),
                   "p95_abs_nm": float(np.percentile(np.abs(torque), 95)), "peak_time_s": float(data["time_s"][peak]),
                   "peak_phase": str(data["phase"][peak]),
                   "active_rms_nm": float(np.sqrt(np.mean(torque[active] ** 2))) if active.any() else None,
                   "cruise_rms_nm": float(np.sqrt(np.mean(torque[cruise] ** 2))) if cruise.any() else None,
                   "hold_rms_nm": float(np.sqrt(np.mean(torque[hold] ** 2))) if hold.any() else None,
                   "peak_error_deg": float(np.max(np.abs(error)) * 180 / np.pi) if group == "steering" else None,
                   "actual_peak_rpm": float(np.max(np.abs(speed)) * 60 / (2 * np.pi)) if group == "wheel" else None,
                   "target_peak_rpm": float(np.nanmax(np.abs(data["wheel_target_rad_s"][:, i])) * 60 / (2 * np.pi)) if group == "wheel" else None,
                   "minimum_wheel_contacts": int(data["wheel_contacts"].min()),
                   "cruise_translation_error_p95_m_s": float(np.percentile(speed_errors[cruise], 95)) if cruise.any() else None,
                   "peak_body_translation_error_m_s": float(speed_errors.max()),
                   "hold_speed_p95_m_s": float(np.percentile(np.hypot(data["actual_vx_m_s"][hold], data["actual_vy_m_s"][hold]), 95)) if hold.any() else None,
                   "solver_balance_max": float(data["full_balance_residual"].max()),
                   "block_error_max_deg": float(data["block_tracking_error_rad"].max() * 180 / np.pi)}
            rows.append(row)
    return rows


def plots(args, summaries, runs):
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .2})
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), layout="constrained")
    for row, phi in enumerate(args.phis):
        if row >= 3:
            break
        label = f"phi_{phi:g}_slope_{max(args.slopes):g}_straight".replace(".", "p")
        if label not in runs:
            continue
        d = runs[label]
        for corner, name in enumerate(("RR", "FR", "RL", "FL")):
            axes[row, 0].plot(d["time_s"], d["torque_nm"][:, 15 + corner], lw=.8, label=name)
            axes[row, 1].plot(d["time_s"], d["speed_rad_s"][:, 15 + corner] * 60 / (2 * np.pi), lw=.8, label=name)
        axes[row, 0].set(title=f"Phi={phi:g}, slope={max(args.slopes):g} deg", ylabel="Wheel torque (N m)")
        axes[row, 1].set(title="Actual wheel speed", ylabel="rpm")
        axes[row, 0].legend(ncol=4, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    fig.suptitle(("Uphill / downhill" if max(args.slopes) else "Flat-ground forward / reverse")
                 + " start, cruise, brake and hold | torque-driven PI wheels")
    save_figure(fig, args.output, "slope_wheel_loads")

    steering_name = max((r for r in summaries if r["group"] == "steering"), key=lambda r: r["peak_nm"])["run"]
    d = runs[steering_name]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout="constrained")
    for i, corner in enumerate(("RR", "FR", "RL", "FL")):
        axes[0].plot(d["time_s"], d["torque_nm"][:, 11 + i], label=corner, lw=.8)
        axes[1].plot(d["time_s"], (d["target_rad"][:, 11 + i] - d["position_rad"][:, 11 + i]) * 180 / np.pi,
                     label=corner, lw=.8)
    axes[0].set(ylabel="Steering torque (N m)", title=f"Highest steering load: {steering_name}")
    axes[1].set(ylabel="Tracking error (deg)", xlabel="Time (s)")
    axes[0].legend(ncol=4)
    save_figure(fig, args.output, "steering_loads")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for column, group in enumerate(("steering", "wheel")):
        rows = [r for r in summaries if r["group"] == group]
        for corner in ("RR", "FR", "RL", "FL"):
            peaks = [max(r["peak_nm"] for r in rows if r["slope_deg"] == slope and r["corner"] == corner)
                     for slope in args.slopes]
            axes[column].plot(args.slopes, peaks, "o-", label=corner)
        axes[column].set(title=f"{group}: maximum across tested cases", xlabel="Slope (deg)", ylabel="Peak torque (N m)")
        axes[column].legend(ncol=2)
    speeds = np.array([.2, .4, .6, .8, 1.])
    axes[2].plot(speeds, speeds / (2 * np.pi * MujocoRobotEnv.WHEEL_RADIUS) * 60, "o-")
    axes[2].set(title="Wheel diameter 51 mm: speed requirement", xlabel="Vehicle speed (m/s)", ylabel="Wheel rpm (pure rolling)")
    save_figure(fig, args.output, "swerve_overview")


def write_report(args, summaries, runs, env):
    def top(group, key):
        return max((r for r in summaries if r["group"] == group), key=lambda r: r[key] or 0)
    rows = []
    for group in ("steering", "wheel"):
        for corner in ("RR", "FR", "RL", "FL"):
            points = [r for r in summaries if r["group"] == group and r["corner"] == corner]
            peak = max(points, key=lambda r: r["peak_nm"])
            continuous = max(max(r["rms_nm"], r["active_rms_nm"] or 0,
                                 r["cruise_rms_nm"] or 0, r["hold_rms_nm"] or 0) for r in points)
            rows.append({"group": group, "corner": corner, "joint": peak["joint"], "measured_peak_nm": peak["peak_nm"],
                         "measured_continuous_proxy_nm": continuous,
                         "preliminary_continuous_with_1p5_margin_nm": continuous * 1.5,
                         "preliminary_peak_with_1p5_margin_nm": peak["peak_nm"] * 1.5,
                         "worst_run": peak["run"],
                         "actual_peak_rpm": max(r["actual_peak_rpm"] or 0 for r in points) if group == "wheel" else None})
    write_csv(args.output / "module_summary.csv", rows)
    write_csv(args.output / "case_summary.csv", summaries)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    report = {"settings": metadata, "mujoco_version": mujoco.__version__,
              "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
              "motor_capacity_clamps": False, "historical_160rpm_clamp": False,
              "pivot_actuation": "passive" if getattr(args, "passive_pivots", True) else "active_hold_legacy_proxy",
              "pivot_frictionloss_nm": env.model.dof_frictionloss[
                  env.model.jnt_dofadr[env.pivot_ids]].tolist(),
              "pivot_damping_nms_rad": env.model.dof_damping[
                  env.model.jnt_dofadr[env.pivot_ids]].tolist(),
              "steering_target_rate_limit_deg_s": getattr(args, "steering_rate_limit_deg_s", 337.5),
              "wheel_controller": "Analysis PI torque servo; gain in Nm/(rad/s), integral gain Nm/rad",
              "summary": rows, "cases": summaries,
              "run_validation": [{"run": name,
                                  "min_contacts": int(d["wheel_contacts"].min()),
                                  "whole_record_translation_error_p95_m_s": float(np.percentile(np.hypot(
                                      d["actual_vx_m_s"] - d["command_vx_m_s"], d["actual_vy_m_s"] - d["command_vy_m_s"]), 95)),
                                  "max_block_error_deg": float(d["block_tracking_error_rad"].max() * 180 / np.pi)}
                                 for name, d in runs.items()]}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    mass = float(MujocoRobotEnv(args.model).model.body_mass.sum())
    radius = MujocoRobotEnv.WHEEL_RADIUS
    theory = []
    for slope in args.slopes:
        theta = np.deg2rad(slope)
        for driven in (4, 2):
            for resistance in (.02, .05):
                steady = mass * 9.8 * (np.sin(theta) + resistance * np.cos(theta)) * radius / driven
                accel = mass * (1.875 * args.speed / args.ramp_seconds) * radius / driven
                theory.append({"slope_deg": slope, "effective_driven_wheels": driven,
                               "assumed_rolling_resistance_coefficient": resistance,
                               "steady_torque_per_wheel_nm": steady, "peak_start_torque_per_wheel_nm": steady + accel})
    write_csv(args.output / "traction_theory.csv", theory)
    lines = ["# 四个舵轮的载荷初步测算", "",
             f"质量 {mass:.3f} kg、轮径 {radius * 2000:g} mm、坡度 {args.slopes}°、形态 Φ={args.phis}。",
             f"直行/横移目标车速 {args.speed:g} m/s；起步/制动 {args.ramp_seconds:g} s，五次平滑曲线最大加速度约 {1.875 * args.speed / args.ramp_seconds:.3f} m/s²。",
             f"原地自转 {args.yaw_rate:g} rad/s；原地转向 90° 用 {args.steer_seconds:g} s，另测更快的 0.5 s。只报告转向扭矩。", "",
             "关闭所有力矩限幅和原 160 rpm 限制，不把 effort 或旧仿真容量作为选型数据。",
             ("pivot（joint8/11/14/17）为重力驱动的被动关节：执行器增益与偏置均归零，保留关节限位及模型轴承阻尼，不参与电机选型。"
              if getattr(args, "passive_pivots", True) else "本次使用历史主动 pivot 保持代理，与重力被动结构不同。"),
             f"pivot 干摩擦 {report['pivot_frictionloss_nm']} N·m，黏性阻尼 "
             f"{report['pivot_damping_nms_rad']} N·m·s/rad；未校准的试验参数。",
             "轮毂使用测算专用、无限幅 PI 扭矩控制，减少速度静差并测试坡上零速保持；原应用控制器未改变。",
             f"变形关节增益倍率 {args.block_gain_scale:g}、转向增益倍率 {args.steering_gain_scale:g}；仅在测算模型生效。转向的速度/轨迹会影响惯性与阻尼力矩；忽略转速选型不代表扭矩与转向时间无关。", "",
             f"测算中转向目标变化速度限制为 {getattr(args, 'steering_rate_limit_deg_s', 337.5):g} °/s（0 表示关闭）。默认值对应 0.5 秒五次平滑转 90° 的最大角速度；防止不承重轮子的投影运动学目标在起步时突变，不是电机力矩限幅。", "",
             "## 8 个电机输出端载荷", "", "| 类型 | 舵轮 | 关节 | 测得峰值 N·m | 持续负载指标 N·m | 1.5×初步额定/峰值 N·m |", "|---|---|---|---:|---:|---|" ]
    for row in rows:
        lines.append(f"| {row['group']} | {row['corner']} | {row['joint']} | {row['measured_peak_nm']:.3f} | {row['measured_continuous_proxy_nm']:.3f} "
                     f"| {row['preliminary_continuous_with_1p5_margin_nm']:.3f} / {row['preliminary_peak_with_1p5_margin_nm']:.3f} |")
    lines += ["", "持续负载指标取各测试整周期 RMS、排除保持后的运动段 RMS、匀速段 RMS 和保持段 RMS 中的最大值，作为筛选线索。1.5 倍只是本次采用的初步工程余量，并非已验证的安全系数。",
              "RMS 的热负荷适用性还取决于真实工作循环、散热和厂商额定条件。峰值必须由电机、减速器和驱动器在相应转速下同时承受。", "",
              "## 速度与坡度", "", f"{args.speed:g} m/s 的纯滚动转速为 {args.speed / (2 * np.pi * radius) * 60:.1f} rpm；0.6、1.0 m/s 分别约 225、374 rpm。",
              f"测试中最大实际轮速 {top('wheel', 'actual_peak_rpm')['actual_peak_rpm']:.1f} rpm，包含瞬态和打滑；不应当作达到的车速。",
              "上坡理论载荷为 τ≈[m g(sinθ+Crr cosθ)+m a]r/n，其中 n 为有效驱动轮数；Crr=0.02/0.05 是独立假设，未伪称模型已校准的滚阻。",
              "`traction_theory.csv` 给出四轮分担与两轮分担的对照。真实车重、滚阻、负载分配和轮胎抓地力都影响需求。", "",
              "## 工况有效性", "", "| 工况 | 匀速平移误差 95% 分位 m/s | 保持速度 95% 分位 m/s | 最少接地轮数 | 最大变形误差 ° |", "|---|---:|---:|---:|---:|"]
    for name in runs:
        r = next(row for row in summaries if row["run"] == name)
        ce = "—" if r["cruise_translation_error_p95_m_s"] is None else f"{r['cruise_translation_error_p95_m_s']:.3f}"
        he = "—" if r["hold_speed_p95_m_s"] is None else f"{r['hold_speed_p95_m_s']:.3f}"
        lines.append(f"| {name} | {ce} | {he} | {r['minimum_wheel_contacts']} | {r['block_error_max_deg']:.2f} |")
    lines += ["", "接触轮数包括短暂/低法向载荷接触；实际载荷分配另存 wheel_normal_force_n。速度/支撑不达标的工况不能当作实现目标的容量验证。",
              ("原地转向分别测试轮毂制动和自由转动；本次仅平地，没有测试坡道。"
               if max(args.slopes) == 0 else "原地转向时轮子分别测了制动和自由转动（自由轮只在平地测）；上坡下坡双向、坡上制动保持均已覆盖。"),
              "## 数据", "", "- `case_summary.csv`：每个工况、8 个电机的峰值/RMS/误差。",
              "- `module_summary.csv`：每个舵轮的载荷最大值。",
              "- 每工况 `.csv/.npz`：每个物理步的全部关节载荷、转速与接触载荷。",
              "- `slope_wheel_loads.png/svg`：坡上轮毂扭矩与转速。",
              "- `steering_loads.png/svg`：最高转向负载及角度误差。",
              "- `swerve_overview.png/svg`：坡度和轮毂转速概览。", "",
              "所有数值是关节/轮毂输出端；带减速器电机轴扭矩约为输出扭矩/(减速比×效率)。轮毂直驱则直接对应电机转矩。",
              "当前模型使用光滑圆柱轮胎、低摩擦辅助万向球接触代理，未校准真实轮胎滚阻、地面材料、轴承/传动损耗。",
              "大于估算的电机扭矩无法解决轮胎打滑或重心不稳定。更高车速、额外载荷、台阶、软地面和冲击需重新测算。", "",
              "参考：[maxon 电机选型](https://www.maxongroup.com/medias/sys_master/8815460712478.pdf)、[MuJoCo 动力学](https://mujoco.readthedocs.io/en/stable/computation/index.html)。", ""]
    (args.output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    p.add_argument("--phis", type=float, nargs="+", default=[0, 1.5, 3])
    p.add_argument("--slopes", type=float, nargs="+", default=[0, 5, 10, 15])
    p.add_argument("--speed", type=float, default=.4)
    p.add_argument("--yaw-rate", type=float, default=.8)
    p.add_argument("--ramp-seconds", type=float, default=.5)
    p.add_argument("--steer-seconds", type=float, default=1.)
    p.add_argument("--cruise-seconds", type=float, default=4.)
    p.add_argument("--hold-seconds", type=float, default=2.)
    p.add_argument("--block-gain-scale", type=float, default=30.)
    p.add_argument("--steering-gain-scale", type=float, default=1.)
    p.add_argument("--steering-rate-limit-deg-s", type=float, default=337.5,
                   help="Limit analysis steering target changes; default is peak speed of a smooth 90 deg / 0.5 s turn; 0 disables")
    p.add_argument("--passive-pivots", action=argparse.BooleanOptionalAction, default=True,
                   help="Gravity-driven pivot hinges (default); --no-passive-pivots reproduces the legacy active-hold proxy")
    p.add_argument("--pivot-frictionloss", type=float,
                   help="Dry friction torque per pivot in Nm; omitted: use environment default")
    p.add_argument("--pivot-damping", type=float,
                   help="Viscous resistance per pivot in Nm*s/rad; omitted: use environment default")
    p.add_argument("--wheel-kp", type=float, default=.025)
    p.add_argument("--wheel-ki", type=float, default=.12)
    p.add_argument("--control-steps", type=int, default=2,
                   help="Physics ticks per target update (default: 240 Hz control with 480 Hz physics)")
    p.add_argument("--timestep-divisor", type=int, default=2,
                   help="Divide the source model timestep (default 2: 480 Hz physics); 240 Hz showed contact-sensitive FR oscillation")
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "experiments" / f"swerve_sizing_{datetime.now():%Y%m%d_%H%M%S}")
    args = p.parse_args()
    if any(v is not None and (not np.isfinite(v) or v < 0)
           for v in (args.pivot_frictionloss, args.pivot_damping)):
        p.error("Pivot frictionloss/damping must be finite and nonnegative")
    if (any(not np.isfinite(phi) or not 0 <= phi <= 3 for phi in args.phis)
            or any(not np.isfinite(slope) or not 0 <= slope <= 20 for slope in args.slopes)
            or any(not np.isfinite(x) or x <= 0 for x in (args.speed, args.yaw_rate, args.ramp_seconds,
                     args.steer_seconds, args.cruise_seconds, args.hold_seconds, args.block_gain_scale,
                     args.steering_gain_scale, args.wheel_kp, args.wheel_ki))
            or not np.isfinite(args.steering_rate_limit_deg_s) or args.steering_rate_limit_deg_s < 0
            or args.control_steps < 1 or args.timestep_divisor < 1):
        p.error("Positive finite motion/gain values, Phi in [0,3], and slopes in [0,20] are required")
    args.output, args.model, args.urdf = args.output.resolve(), args.model.resolve(), args.urdf.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    runs, summaries = {}, []
    cases = [(phi, slope, "straight", None, None) for phi in args.phis for slope in args.slopes]
    cases += [(phi, 0, mode, None, None) for phi in args.phis for mode in ("lateral", "spin", "direction")]
    cases += [(phi, slope, "steer", duration, braked) for phi in args.phis for slope in (0, max(args.slopes))
              for duration, braked in ((args.steer_seconds, True), (.5, True))]
    cases += [(phi, 0, "steer", args.steer_seconds, False) for phi in args.phis]
    # A flat-only study repeats slope 0 in the two steering slope slots. Run
    # each case once rather than overwriting identical records and summaries.
    cases = list(dict.fromkeys(cases))
    for phi, slope, mode, duration, braked in cases:
        name = f"phi_{phi:g}_slope_{slope:g}_{mode}"
        if mode == "steer":
            name += f"_{duration:g}s_{'braked' if braked else 'free'}"
        name = name.replace(".", "p")
        print(f"Testing {name}", flush=True)
        run = SwerveLoadRun(args, phi, slope)
        try:
            if mode == "steer":
                run.steer_test(duration, braked)
            elif mode == "direction":
                run.direction_change()
            else:
                run.drive(mode)
            data = run.finish(name)
            runs[name] = data
            summary = summarize_run(name, data)
            summaries += summary
            print(f"  steer peak {max(r['peak_nm'] for r in summary if r['group']=='steering'):.3f} N m; "
                  f"wheel peak {max(r['peak_nm'] for r in summary if r['group']=='wheel'):.3f} N m; "
                  f"cruise error {summary[0]['cruise_translation_error_p95_m_s']}", flush=True)
        finally:
            run.env.close()
    plots(args, summaries, runs)
    write_report(args, summaries, runs, run.env)
    print(f"Finished: {args.output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
