"""Compare bounded feedback-only exploration on the existing eight-motor model.

No model files are rebuilt. Exact constraints are read only for an offline
audit; experimental commands use independent bounded servos and learned models.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from envs.mujoco_closed_loop import DEFAULT_CLOSED_MODEL
from exploration.closed_loop_probe import ClosedLoopProbe
from exploration.graph_search import (METHODS, candidate_commands, choose_command, fit_response,
                                      response_relations)


def scalar_trial(trial):
    return {key: trial[key] for key in ("valid", "reason", "peak_speed_rad_s",
                                      "peak_motor_torque_nm", "max_gap_mm", "max_angle_deg")}


def add_trace(traces, trial, role):
    traces.append({"role": role, **trial})


def model_audit(probe):
    """Privileged geometric diagnostics NEVER supplied to the search algorithm."""
    ik, m = probe.kinematics, probe.model
    phis, qs = ik.trajectory(121)
    d = mujoco.MjData(m)
    records = []
    for requested in (0, .25, .5, .75, 1, 1.25, 1.5, 2, 2.5, 3):
        index = np.argmin(abs(phis - requested))
        d.qpos[probe.qadr] = qs[index]
        mujoco.mj_forward(m, d)
        rows = d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        jac = d.efc_J.reshape(d.nefc, m.nv)[rows][:, probe.dofs]
        _, singular, vt = np.linalg.svd(jac)
        rank = int(sum(singular > singular[0] * 1e-8))
        projection = vt[rank:].T @ vt[rank:]
        records.append({"reference_phi": requested, "q_deg": np.rad2deg(qs[index]).tolist(),
                        "equality_rank": rank, "equality_nullity": len(probe.names) - rank,
                        "equality_singular_values": singular.tolist(),
                        "tangent_diagonal": np.diag(projection).tolist()})
    return {"records": records, "learner_access": False,
            "caveat": "Equality tangent excludes unilateral limits/contacts and finite-step feasibility"}


def run_case(probe, method, seed, args, test_commands, test_responses, output):
    rng = np.random.default_rng(seed)
    start = probe.initial
    lo, hi = probe.bounds(start, args.amplitude)
    baseline = probe.run(start, np.zeros(len(probe.names)))
    if not baseline["valid"]:
        raise RuntimeError("Initial baseline violates the experiment guard")
    X, Y, origins, traces, learning = [], [], [], [], []
    add_trace(traces, baseline, "initial_null")
    for tick in range(args.budget):
        # Common first n mixed probes give input coverage without pairing labels.
        pool, kinds = candidate_commands(lo, hi, args.amplitude, np.random.default_rng(seed * 1000 + tick))
        if tick < len(probe.names):
            choices = [i for i, kind in enumerate(kinds) if kind == "multi"]
            selected = int(rng.choice(choices))
        else:
            fitted = fit_response(X, Y, method, probe.edges, args.amplitude)
            selected = choose_command(method, pool, X, fitted, rng)
        trial = probe.run(start, pool[selected])
        add_trace(traces, trial, "local_train")
        if not trial["valid"]:
            raise RuntimeError(f"Local trial rejected: {method}/{seed}/{tick}")
        X.append(pool[selected]); Y.append(trial["delta"] - baseline["delta"]); origins.append(start.q)
        if tick + 1 in (8, 16, args.budget):
            fitted = fit_response(X, Y, method, probe.edges, args.amplitude)
            error = np.sqrt(np.mean((fitted.predict(test_commands) - test_responses) ** 2))
            learning.append({"probes": tick + 1, "heldout_rmse_deg": float(np.rad2deg(error))})
    fitted = fit_response(X, Y, method, probe.edges, args.amplitude)
    local_matrix = fitted.matrix.copy()
    relations = response_relations(X, Y, test_responses)
    # Held-out measured motions become small feasible requests; no test action is revealed.
    controls = []
    for desired in test_responses[:4] * .5:
        command = fitted.command_for_motion(desired, lo, hi, amplitude=args.amplitude, damping=.01)
        trial = probe.run(start, command)
        add_trace(traces, trial, "control_evaluation")
        achieved = trial["delta"] - baseline["delta"]
        controls.append({"desired_rad": desired.tolist(), "achieved_rad": achieved.tolist(),
                         "error_deg": float(np.rad2deg(np.sqrt(np.mean((achieved - desired) ** 2)))),
                         **scalar_trial(trial)})
    # A roadmap of physically executed local transitions. Every edge costs a
    # null-input trial plus one motion trial, even if it is rejected.
    states, visits, parents, edge_records = [start], [0], [-1], []
    for tick in range(args.extensions):
        qnodes = np.array([state.q for state in states])
        distances = np.linalg.norm(qnodes - start.q, axis=1)
        priority = (.25 + distances / args.amplitude) / (1 + np.asarray(visits))
        parent = int(np.argmax(priority))
        state = states[parent]
        visits[parent] += 1
        nearby = np.argsort(np.linalg.norm(np.asarray(origins) - state.q, axis=1))[:48]
        fitted = fit_response(np.asarray(X)[nearby], np.asarray(Y)[nearby], method,
                              probe.edges, args.amplitude)
        lo, hi = probe.bounds(state, args.amplitude, args.radius)
        pool, kinds = candidate_commands(lo, hi, args.amplitude,
            np.random.default_rng(seed * 1000 + 100 + tick),
            matrix=fitted.matrix if method.endswith("frontier") else None)
        chosen = choose_command(method, pool, np.asarray(X)[nearby], fitted, rng,
                                q=state.q, visited=qnodes)
        null = probe.run(state, np.zeros(len(probe.names)))
        trial = probe.run(state, pool[chosen])
        add_trace(traces, null, "search_null"); add_trace(traces, trial, "search_motion")
        if null["valid"] and trial["valid"]:
            X.append(pool[chosen]); Y.append(trial["delta"] - null["delta"]); origins.append(state.q)
        novelty = float(np.min(np.linalg.norm(qnodes - trial["state"].q, axis=1)))
        accepted = null["valid"] and trial["valid"] and novelty > np.deg2rad(.15)
        target = len(states) if accepted else None
        edge_records.append({"query": tick, "parent": parent, "child": target,
                             "kind": kinds[chosen], "accepted": bool(accepted),
                             "novelty_deg": float(np.rad2deg(novelty)),
                             "q_end_rad": trial["state"].q.tolist(),
                             "command_rad": pool[chosen].tolist(), "null_valid": null["valid"],
                             **scalar_trial(trial)})
        if accepted:
            states.append(trial["state"]); visits.append(0); parents.append(parent)
    qnodes = np.array([state.q for state in states])
    centered = qnodes - start.q
    values = np.linalg.svd(centered, compute_uv=False) / np.sqrt(len(centered))
    # Coverage is computed in all observed joints; the first two singular values
    # are a descriptive spread, not an assumed stage dimension or area estimate.
    summary = {"method": method, "seed": seed, "learning": learning,
        "prediction_rmse_deg": learning[-1]["heldout_rmse_deg"],
        "control_rmse_deg": float(np.mean([r["error_deg"] for r in controls])),
        "nodes": len(states), "accepted_edges": int(sum(e["accepted"] for e in edge_records)),
        "max_distance_from_start_deg": float(np.rad2deg(np.max(np.linalg.norm(centered, axis=1)))),
        "joint_span_deg": np.rad2deg(np.ptp(qnodes, axis=0)).tolist(),
        "spread_singular_values_deg": np.rad2deg(values).tolist(),
        "guard_rejections": sum(not e["valid"] or not e["null_valid"] for e in edge_records),
        "search_queries_reaching_torque_cap": sum(e["peak_motor_torque_nm"] >= probe.torque_limit - 1e-6
                                                  for e in edge_records),
        "peak_motor_torque_nm": max(t["peak_motor_torque_nm"] for t in traces),
        "peak_speed_rad_s": max(t["peak_speed_rad_s"] for t in traces),
        "max_gap_mm": max(t["max_gap_mm"] for t in traces),
        "max_angle_deg": max(t["max_angle_deg"] for t in traces),
        "training_motion_trials": args.budget + args.extensions,
        "training_null_trials": 1 + args.extensions,
        "controls": controls, "relations": relations,
        "state_graph": {"q_rad": qnodes.tolist(), "parents": parents, "edges": edge_records,
                        "meaning": "executed local motions; no phase labels or inferred unlock gates"}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    arrays = {"commands": np.asarray(X), "responses": np.asarray(Y), "origins": np.asarray(origins),
              "initial_response_matrix": local_matrix, "q_nodes": qnodes,
              "trial_roles": np.asarray([t["role"] for t in traces])}
    for i, t in enumerate(traces):
        for key in ("q", "qdot", "time", "command", "delta", "integrated_velocity"):
            arrays[f"trial_{i:03d}_{key}"] = t[key]
    np.savez_compressed(output / "measurements.npz", **arrays)
    return summary


def render_summary(report, output):
    cases = report["cases"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    labels = ["Random", "Information", "Graph + info", "Frontier", "Graph + frontier"]
    metrics = [("prediction_rmse_deg", "Held-out prediction RMSE (deg)"),
               ("max_distance_from_start_deg", "Maximum joint-space distance (deg)"),
               ("accepted_edges", "New states found"),
               ("control_rmse_deg", "Small motion tracking RMSE (deg)")]
    for ax, (metric, title) in zip(axes.flat, metrics):
        means, stds = [], []
        for method in METHODS:
            values = [c[metric] for c in cases if c["method"] == method]
            means.append(np.mean(values)); stds.append(np.std(values))
        ax.bar(labels, means, yerr=stds, capsize=4,
               color=["#99a6b4", "#4186ad", "#76a985", "#a98bb8", "#d29a42"])
        ax.set_title(title); ax.tick_params(axis="x", labelsize=9); ax.grid(axis="y", alpha=.2)
    fig.suptitle("Eight-motor loop | bounded angle-feedback exploration\nError bars: variation across command seeds, not confidence intervals")
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)
    # Shared data-derived PCA axes for a representative seed; only a visualization.
    seed = report["settings"]["seeds"][0]
    chosen = [c for c in cases if c["seed"] == seed]
    origin = np.deg2rad(report["model"]["settled_q_deg"])
    allq = np.concatenate([c["state_graph"]["q_rad"] for c in chosen]) - origin
    basis = np.linalg.svd(allq, full_matrices=False)[2][:2].T
    fig, axes = plt.subplots(1, len(METHODS), figsize=(17, 4), sharex=True, sharey=True, layout="constrained")
    for ax, case, label in zip(axes, chosen, labels):
        points = np.rad2deg((np.array(case["state_graph"]["q_rad"]) - origin) @ basis)
        for child, parent in enumerate(case["state_graph"]["parents"]):
            if parent >= 0:
                ax.plot(points[[parent, child], 0], points[[parent, child], 1], color="#618da4", lw=.7)
        ax.scatter(points[:, 0], points[:, 1], s=10, c=np.arange(len(points)), cmap="viridis")
        ax.scatter(0, 0, marker="*", s=100, color="#d85b3e")
        ax.set_title(label); ax.set_xlabel("Shared PCA coordinate 1 (deg)"); ax.grid(alpha=.2)
    axes[0].set_ylabel("Shared PCA coordinate 2 (deg)")
    fig.suptitle("Executed state graphs from the same start | no predefined motor groups")
    fig.savefig(output / "state_graphs.png", dpi=180)
    plt.close(fig)


def write_results(report, output):
    audit = report["geometry_audit"]["records"]
    ranks = sorted({row["equality_rank"] for row in audit})
    dimensions = sorted({row["equality_nullity"] for row in audit})
    settings = report["settings"]
    model_path = report["model"]["model"]
    command = ["python", "scripts/benchmark_closed_loop_search.py", "--model", model_path,
               "--output", str(output),
               "--seeds", *map(str, settings["seeds"])]
    for key in ("budget", "extensions", "amplitude", "radius", "hz"):
        command.extend(["--" + key, str(settings[key])])
    lines = ["# 虚拟八电机闭环：探索算法实验", "",
             f"实验日期：{report.get('run_date', '未记录（历史报告）')}。", "",
             "模型为推算几何；下面是本次仿真实测，不是实机性能或已确认的论文创新。", "",
             "## 相同预算下的结果", "",
             "| 方法 | 预测 RMSE / ° | 小目标误差 / ° | 新状态数 | 最大形态距离 / ° |", "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        cs = [c for c in report["cases"] if c["method"] == method]
        cells = []
        for metric in ("prediction_rmse_deg", "control_rmse_deg", "accepted_edges", "max_distance_from_start_deg"):
            values = [c[metric] for c in cs]
            cells.append(f"{np.mean(values):.4f} ± {np.std(values):.4f}")
        lines.append("| " + method + " | " + " | ".join(cells) + " |")
    lines += ["", "± 为不同指令随机种子的标准差；模型、传感器与物理参数没有随机化。", "",
        "形态距离使用全部八个关节角度的欧氏距离，不是单个电机的转角；新状态阈值为 0.15° 的该距离。",
        "小目标是留出试探实际响应的一半，各方法只得到目标角变化，不得到产生它的电机指令。", "",
        "## 设置与解释", "",
        f"- {report['settings']['budget']} 次局部训练 + {report['settings']['extensions']} 次状态扩展；每次扩展额外执行一次零增量基线。",
        f"- 指令总角增量范数上限 {report['settings']['amplitude']} rad；每个动作 {report['model']['duration']} s，前半段缓升、后半段保持。",
        f"- 物理频率 {report['model']['hz']} Hz，独立 PD 参数 kp={report['model']['kp']}、kd={report['model']['kd']}，每电机限矩 {report['model']['torque_limit_nm']} N·m。",
        "- 学习器只读角度与转速，使用已知装配连接；不读约束雅可比、Phi 标签、配对、阶段或解锁角度。",
        "- 初始姿态来自现有闭合 IK 的 Phi=0 对应解。几何本身曾用 Phi 公式拟合，因此不能称为完全独立于 Phi 的物理验证。",
        "- 所有已有碰撞排除项和掩码保留；XML 内容哈希在实验结束检查不变。",
        "- 状态树通过模拟器快照返回父节点，未计入实机返回成本；边是执行过的小幅运动，不是阶段转换。",
        "- 图先验为软正则，不是 GNN；图前沿搜索是当前原型启发式，不是某篇论文的完整复现。", "",
        "## 模型核对", "",
        f"`geometry_audit` 只供离线评价：沿既有参考路径的检查点，闭合等式秩集合为 {ranks}，形变关节等式切空间维度集合为 {dimensions}。",
        "这不等于每个点都有两条可持续运动路线；限位、接触和有限步长仍需实际执行验证。",
        f"初始几何中 joint2、joint6 的切空间投影对角元分别为 {audit[0]['tangent_diagonal'][1]:.4f}、{audit[0]['tangent_diagonal'][5]:.4f}，需结合单电机试探检查是否复现实机锁止。",
        "本次搜索只覆盖初始附近，未证明发现联合解锁，也未以无发现推断更远处没有阶段变化。", "",
        "## 数值与执行检查", "",
        f"- 采样闭合位置误差最大 {max(c['max_gap_mm'] for c in report['cases']):.5f} mm，朝向误差最大 {max(c['max_angle_deg'] for c in report['cases']):.5f}°。",
        f"- 主体关节最大速度 {max(c['peak_speed_rad_s'] for c in report['cases']):.5f} rad/s；电机最大输出 {max(c['peak_motor_torque_nm'] for c in report['cases']):.3f} N·m。",
        "- 部分搜索触及有限力矩上限；未触发仿真停止条件不等于实机安全性已验证。累计目标偏差可能产生内力，下一轮需处理目标与实际运动之间的差距。",
        f"- 小目标直接限幅执行误差 {report['control_references']['clipped_direct_request_as_command_rmse_deg']:.5f}°，不运动参考误差 {report['control_references']['zero_motion_rmse_deg']:.5f}°。", "",
        "## 复现", "", "```bash",
        shlex.join(command),
        shlex.join(["python", "scripts/benchmark_closed_loop_search.py", "--model", model_path,
                    "--output", str(output),
                    "--hz", str(settings["hz"]), "--audit-only"]),
        "```", "", "原始反馈、指令、局部模型和状态图保存在各方法/种子目录。",
        "![指标比较](comparison.png)", "", "![实际执行的状态图](state_graphs.png)"]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")


def validation_diagnostics(probe, output):
    """Read-only extra checks, separated from all training and policy choices."""
    null = probe.run(probe.initial, np.zeros(len(probe.names)))
    records, arrays = [], {}
    for i, name in enumerate(probe.names):
        command = np.zeros(len(probe.names))
        command[i] = -.03 if probe.initial.target[i] > 0 else .03
        trial = probe.run(probe.initial, command)
        delta = trial["delta"] - null["delta"]
        records.append({"joint": name, "command_deg": float(np.rad2deg(command[i])),
                        "own_response_deg": float(np.rad2deg(delta[i])),
                        "response_deg": np.rad2deg(delta).tolist(), **scalar_trial(trial)})
        arrays[name + "_q"] = trial["q"]
        arrays[name + "_qdot"] = trial["qdot"]
    (output / "single_motor_audit.json").write_text(json.dumps(records, indent=2) + "\n")
    np.savez_compressed(output / "single_motor_audit.npz", **arrays)
    with np.load(output / "heldout.npz") as heldout:
        high = ClosedLoopProbe(probe.path, hz=probe.hz * 2, duration=probe.duration,
                               kp=probe.kp, kd=probe.kd, torque_limit=probe.torque_limit,
                               initial_angles=probe.reference)
        try:
            baseline = high.run(high.initial, np.zeros(len(high.names)))
            trials = [high.run(high.initial, command) for command in heldout["commands"]]
            responses = np.array([trial["delta"] - baseline["delta"] for trial in trials])
            error = responses - heldout["responses"]
            result = {"low_hz": probe.hz, "high_hz": high.hz, "same_heldout_commands": len(trials),
                "response_difference_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(error ** 2)))),
                "response_difference_max_component_deg": float(np.rad2deg(np.max(abs(error)))),
                "all_valid": all(trial["valid"] for trial in trials),
                "max_gap_mm": max(trial["max_gap_mm"] for trial in trials),
                "max_angle_deg": max(trial["max_angle_deg"] for trial in trials),
                "note": "Local response check only; not convergence of entire state search or physical validation."}
            (output / "timestep_check.json").write_text(json.dumps(result, indent=2) + "\n")
            np.savez_compressed(output / "timestep_check.npz", responses_high_hz=responses,
                                q=np.array([t["q"] for t in trials]), qdot=np.array([t["qdot"] for t in trials]))
        finally:
            high.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=DEFAULT_CLOSED_MODEL)
    p.add_argument("--output", type=Path,
                   default=ROOT / "experiments" / f"closed_loop_search_{datetime.now():%Y%m%d_%H%M%S_%f}")
    p.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    p.add_argument("--budget", type=int, default=32)
    p.add_argument("--extensions", type=int, default=48)
    p.add_argument("--amplitude", type=float, default=.03)
    p.add_argument("--radius", type=float, default=.3)
    p.add_argument("--hz", type=int, default=1920)
    p.add_argument("--audit-only", action="store_true", help="Add motor and doubled-frequency checks to existing results")
    args = p.parse_args()
    if args.budget < 16 or args.extensions < 1 or not 0 < args.amplitude <= .04 or not .04 <= args.radius <= .4:
        p.error("Use budget>=16, extensions>=1, amplitude in (0,.04], radius in [.04,.4]")
    args.output.mkdir(parents=True, exist_ok=True)
    probe = ClosedLoopProbe(args.model, hz=args.hz)
    try:
        if args.audit_only:
            previous = json.loads((args.output / "report.json").read_text())
            if previous["model"]["sha256"] != probe.source_hash or previous["model"]["hz"] != probe.hz:
                raise ValueError("Audit requires the same model and base frequency as the recorded run")
            validation_diagnostics(probe, args.output)
            print("Saved independent diagnostics", args.output, flush=True)
            return
        report = {"run_date": datetime.now().date().isoformat(),
                  "settings": {k: v for k, v in vars(args).items() if k not in ("model", "output")},
                  "model": probe.metadata(), "geometry_audit": model_audit(probe), "cases": []}
        lo, hi = probe.bounds(probe.initial, args.amplitude)
        pool, kinds = candidate_commands(lo, hi, args.amplitude, np.random.default_rng(991))
        multi = np.array([i for i, kind in enumerate(kinds) if kind == "multi"])
        test_commands = pool[np.random.default_rng(1991).choice(multi, 16, replace=False)]
        null = probe.run(probe.initial, np.zeros(len(probe.names)))
        test_trials = [probe.run(probe.initial, command) for command in test_commands]
        if not null["valid"] or any(not t["valid"] for t in test_trials):
            raise RuntimeError("Held-out evaluation trial violated guard")
        test_responses = np.array([t["delta"] - null["delta"] for t in test_trials])
        np.savez_compressed(args.output / "heldout.npz", commands=test_commands, responses=test_responses,
                            q=np.array([t["q"] for t in test_trials]),
                            qdot=np.array([t["qdot"] for t in test_trials]))
        # Fair control references use the same requests and initial snapshot.
        direct_errors = []
        for desired in test_responses[:4] * .5:
            trial = probe.run(probe.initial, np.clip(desired, lo, hi))
            direct_errors.append(float(np.rad2deg(np.sqrt(np.mean(
                (trial["delta"] - null["delta"] - desired) ** 2)))))
        report["control_references"] = {"clipped_direct_request_as_command_rmse_deg": float(np.mean(direct_errors)),
            "zero_motion_rmse_deg": float(np.mean(np.rad2deg(np.sqrt(np.mean((test_responses[:4] * .5) ** 2, axis=1)))))}
        print("Model:", json.dumps(report["model"]), flush=True)
        for seed in args.seeds:
            for method in METHODS:
                case = run_case(probe, method, seed, args, test_commands, test_responses,
                                args.output / f"{method}_seed_{seed}")
                report["cases"].append(case)
                print(method, seed, "prediction", round(case["prediction_rmse_deg"], 4),
                      "new_states", case["accepted_edges"], "distance", round(case["max_distance_from_start_deg"], 3), flush=True)
        if hashlib.sha256(args.model.read_bytes()).hexdigest() != probe.source_hash:
            raise RuntimeError("Source model changed during the experiment")
        report["source_model_unchanged"] = True
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        render_summary(report, args.output)
        write_results(report, args.output)
        print("Saved", args.output, flush=True)
    finally:
        probe.close()


if __name__ == "__main__":
    main()
