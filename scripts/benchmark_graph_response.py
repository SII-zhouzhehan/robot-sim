"""Small graph-prior comparison on the incomplete MuJoCo model, at Phi=0.

All models use identical measured training, validation and test probes. This
compares model structure, not active-exploration policies or real loop closure.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from exploration.active_motion import ActiveMotionLearner
from exploration.graph_response import (graph_penalty, response_graph_evidence,
                                        select_response_model)
from exploration.mujoco_probe import MujocoJointProbe


def rmse(left, right):
    return float(np.sqrt(np.mean((np.asarray(left) - right) ** 2)))


def bounded_random(probe, rng, amplitude):
    direction = rng.normal(size=len(probe.names))
    direction = np.where(probe.upper <= 1e-10, -abs(direction), direction)
    direction = np.where(probe.lower >= -1e-10, abs(direction), direction)
    command = ActiveMotionLearner.bounded_direction(direction, probe.lower, probe.upper, amplitude)
    if command is None:
        raise RuntimeError("No random command fits the current bounds")
    return command


def training_commands(probe, rng, amplitude):
    commands = []
    for fraction in (.5, 1.):
        for i in range(len(probe.names)):
            direction = np.eye(len(probe.names))[i]
            if probe.upper[i] <= 1e-10:
                direction *= -1
            command = ActiveMotionLearner.bounded_direction(direction, probe.lower, probe.upper,
                                                           amplitude * fraction)
            if command is None:
                raise RuntimeError(f"No independent probe is available for {probe.names[i]}")
            commands.append(command)
    commands.extend(bounded_random(probe, rng, amplitude) for _ in range(12))
    return np.asarray(commands)


def collect(probe, commands, feedback, split, records):
    responses = []
    for index, command in enumerate(commands):
        trial = probe.run(command)
        records.append({**trial, "label": f"{split}_{index}"})
        responses.append(probe.response(trial, feedback))
    return np.asarray(responses)


def run_case(probe, seed, amplitude, feedback, control, output):
    rng = np.random.default_rng(seed)
    n = len(probe.names)
    X = training_commands(probe, rng, amplitude)
    validation_x = np.array([bounded_random(probe, rng, amplitude) for _ in range(8)])
    test_x = np.array([bounded_random(probe, rng, amplitude) for _ in range(16)])
    records = [dict(probe.baseline, label="null_baseline")]
    Y = collect(probe, X, feedback, "train", records)
    validation_y = collect(probe, validation_x, feedback, "validation", records)
    test_y = collect(probe, test_x, feedback, "test", records)
    permutation = np.random.default_rng(seed + 500).permutation(n)
    permuted_edges = [(int(permutation[a]), int(permutation[b])) for a, b in probe.physical_edges]
    penalties = {"dense": np.ones((n, n)),
                 "graph_prior": graph_penalty(n, probe.physical_edges),
                 "shuffled_graph_prior": graph_penalty(n, permuted_edges),
                 "diagonal": np.ones((n, n))}
    results, fits = {}, {}
    for name, penalty in penalties.items():
        history = []
        for budget in (n, 2 * n, len(X)):
            model, validation_error = select_response_model(
                X[:budget], Y[:budget], validation_x, validation_y, penalty,
                scale=amplitude, diagonal=name == "diagonal")
            history.append({"training_probes": budget, "validation_rmse_rad": validation_error,
                            "test_rmse_rad": rmse(model.predict(test_x), test_y),
                            "regularization": model.regularization})
        fits[name] = model
        evidence = response_graph_evidence(model, X, Y, validation_y, names=probe.names,
                                           bootstrap=80, seed=seed)
        results[name] = {"history": history, "test_rmse_rad": history[-1]["test_rmse_rad"],
                         "validation_rmse_rad": validation_error, "matrix": model.matrix.tolist(),
                         "penalty": penalty.tolist(), "graph_evidence": evidence}
    # The final test set is excluded from both regularization and model selection.
    usable = ("dense", "graph_prior", "diagonal")
    chosen = min(usable, key=lambda name: results[name]["validation_rmse_rad"])
    control_requests = [bounded_random(probe, rng, amplitude * .25) for _ in range(3)]
    control_reference = {}
    if control:
        direct_errors = []
        for index, desired in enumerate(control_requests):
            trial = probe.run(desired)
            records.append(dict(trial, label=f"control_direct_command_{index}"))
            direct_errors.append(rmse(probe.response(trial, feedback), desired))
        control_reference = {
            "direct_command_tracking_rmse_rad": float(np.mean(direct_errors)),
            "zero_motion_tracking_rmse_rad": float(np.mean([rmse(np.zeros(n), desired)
                                                           for desired in control_requests]))}
        for name, model in fits.items():
            trials = []
            for index, desired in enumerate(control_requests):
                command = model.command_for_motion(desired, probe.lower, probe.upper, amplitude=amplitude)
                trial = probe.run(command)
                measured = probe.response(trial, feedback)
                records.append(dict(trial, label=f"control_{name}_{index}"))
                trials.append({"desired_motion_rad": desired.tolist(), "command_rad": command.tolist(),
                               "measured_motion_rad": measured.tolist(),
                               "tracking_rmse_rad": rmse(measured, desired),
                               "prediction_rmse_rad": rmse(measured, model.predict(command))})
            results[name]["control"] = trials
            results[name]["control_tracking_rmse_rad"] = float(np.mean([t["tracking_rmse_rad"] for t in trials]))
    parity = max(float(np.max(abs(t["displacement"] - t["integrated_velocity"]))) for t in records)
    peak_speed = max(float(np.max(abs(t["velocity_rad_s"]))) for t in records)
    case = {"seed": seed, "amplitude_rad": amplitude, "feedback": feedback,
            "training_probe_count": len(X), "validation_probe_count": len(validation_x),
            "test_probe_count": len(test_x), "selected_by_validation": chosen,
            "control_reference": control_reference,
            "shuffled_edges": permuted_edges, "models": results,
            "peak_measured_morph_speed_rad_s": peak_speed,
            "position_velocity_integration_max_error_rad": parity,
            "zero_prediction_rmse_rad": rmse(np.zeros_like(test_y), test_y)}
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "measurements.npz", train_commands=X, train_responses=Y,
                        validation_commands=validation_x, validation_responses=validation_y,
                        test_commands=test_x, test_responses=test_y,
                        positions=np.array([t["position_rad"] for t in records]),
                        velocities=np.array([t["velocity_rad_s"] for t in records]),
                        targets=np.array([t["target_rad"] for t in records]),
                        labels=np.array([t["label"] for t in records]))
    (output / "report.json").write_text(json.dumps(case, indent=2) + "\n")
    print(f"seed={seed}, amplitude={amplitude:.3f} rad; "
          + ", ".join(f"{name}={np.rad2deg(result['test_rmse_rad']):.4f} deg"
                      for name, result in results.items()), flush=True)
    return case


def summarize(output, cases, metadata, control):
    labels = ("dense", "graph_prior", "shuffled_graph_prior", "diagonal")
    summary = {}
    for name in labels:
        errors = np.rad2deg([case["models"][name]["test_rmse_rad"] for case in cases])
        summary[name] = {"test_rmse_deg_mean": float(errors.mean()),
                         "test_rmse_deg_sample_std": float(errors.std(ddof=1)) if len(errors) > 1 else 0.,
                         "selected_by_validation_count": sum(case["selected_by_validation"] == name for case in cases)}
        if control:
            tracking = np.rad2deg([case["models"][name]["control_tracking_rmse_rad"] for case in cases])
            summary[name]["control_tracking_rmse_deg_mean"] = float(tracking.mean())
    report = {"metadata": metadata, "cases": cases, "aggregate": summary,
              "protocol": "Same data; separate training/validation/test; graph vs dense vs permuted prior",
              "implemented_algorithm": "Graph-distance-weighted ridge response model; not paper reproductions",
              "limitations": ["Current model has seven morph joints and no loop closure equality",
                              "All trials restore one simulation snapshot at Phi=0",
                              "Seeds change probe datasets, not physical model or sensor noise",
                              "No RL, GNN, torque/current observations or stage-transition inference",
                              "Graph relations concern measured servo response, not certified geometry"]}
    if control:
        report["control_reference"] = {
            key.removesuffix("_rad") + "_deg": float(np.rad2deg(np.mean([case["control_reference"][key] for case in cases])))
            for key in ("direct_command_tracking_rmse_rad", "zero_motion_tracking_rmse_rad")}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    memory = {"physical_graph": {"joint_names": metadata["joint_names"],
                                  "edges": metadata["physical_joint_edges"]},
              "local_patches": [{"id": "phi0_observation_patch", "initial_phi": 0,
                                  "anchor_q_rad": metadata["actual_initial_angles_rad"],
                                  "scope": "One pose; tested probe amplitudes and servo setup only",
                                  "evidence_files": [f"amplitude_{case['amplitude_rad']:.3f}_seed_{case['seed']}/report.json"
                                                     for case in cases]}],
              "transitions": [], "transition_learning_implemented": False}
    (output / "local_memory.json").write_text(json.dumps(memory, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    for name in labels:
        errors = np.array([[np.rad2deg(h["test_rmse_rad"]) for h in case["models"][name]["history"]]
                           for case in cases])
        axes[0].plot([h["training_probes"] for h in cases[0]["models"][name]["history"]], errors.mean(axis=0),
                     marker="o", label=name)
    axes[0].set(xlabel="Training probes (+ 8 validation)", ylabel="Test displacement RMSE (deg)",
                title="Same data; graph changes model prior")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.2)
    positions = np.arange(len(labels))
    means = [summary[name]["test_rmse_deg_mean"] for name in labels]
    spread = [summary[name]["test_rmse_deg_sample_std"] for name in labels]
    axes[1].bar(positions, means, yerr=spread, color=["#6987a2", "#439785", "#a58baf", "#d39860"], capsize=4)
    axes[1].set_xticks(positions, ["dense", "graph", "shuffled", "diagonal"])
    axes[1].set(ylabel="RMSE (deg), mean +/- case std", title=f"{len(cases)} dataset/amplitude cases")
    if control:
        axes[2].bar(positions, [summary[name]["control_tracking_rmse_deg_mean"] for name in labels],
                    color=["#6987a2", "#439785", "#a58baf", "#d39860"])
        axes[2].set_xticks(positions, ["dense", "graph", "shuffled", "diagonal"])
        axes[2].set(ylabel="Motion-request tracking RMSE (deg)", title="Small command via learned response inversion")
        axes[2].axhline(report["control_reference"]["direct_command_tracking_rmse_deg"],
                        color="#777777", linestyle="--", label="direct command")
        axes[2].axhline(report["control_reference"]["zero_motion_tracking_rmse_deg"],
                        color="#999999", linestyle=":", label="no motion")
        axes[2].legend(fontsize=8)
    else:
        axes[2].axis("off")
        axes[2].text(.1, .5, "Control not executed", fontsize=14)
    fig.suptitle("Incomplete model at Phi=0 | bounded angle probes | position/velocity only")
    fig.savefig(output / "graph_response_comparison.png", dpi=150)
    plt.close(fig)
    lines = ["# 图先验局部辨识：残缺模型实测", "",
             "模型：七个主体关节、无首尾闭合等式约束；初始化 Phi=0。",
             "所有方法共用同一组数据，验证集选正则化，测试集只报告结果。",
             "随机种子改变指令数据，不代表不同机构或真实传感器噪声。", "",
             "| 方法 | 测试 RMSE 均值（度） | 各案例标准差 | 验证集选中次数 |",
             "|---|---:|---:|---:|"]
    for name, result in summary.items():
        lines.append(f"| {name} | {result['test_rmse_deg_mean']:.5f} | "
                     f"{result['test_rmse_deg_sample_std']:.5f} | {result['selected_by_validation_count']} |")
    lines.extend(["", "| 幅值（rad） | seed | dense | graph | shuffled | diagonal |",
                  "|---|---:|---:|---:|---:|---:|"])
    for case in cases:
        lines.append(f"| {case['amplitude_rad']:.3f} | {case['seed']} | " +
                     " | ".join(f"{np.rad2deg(case['models'][name]['test_rmse_rad']):.5f}" for name in labels) + " |")
    if control:
        lines += ["", "| 控制方式 | 小幅运动请求跟踪 RMSE 均值（度） |", "|---|---:|"]
        for name in labels:
            lines.append(f"| {name} | {summary[name]['control_tracking_rmse_deg_mean']:.5f} |")
        lines.append(f"| 直接把运动请求当指令 | {report['control_reference']['direct_command_tracking_rmse_deg']:.5f} |")
        lines.append(f"| 不产生增量运动 | {report['control_reference']['zero_motion_tracking_rmse_deg']:.5f} |")
    lines += ["", "这些误差衡量短时位移预测，不能替代实机形变成功率。",
              "图距离只改变软正则化，不禁止非相邻电机影响，也不提供人工配对或阶段答案。",
              "候选关系通过输入激发、bootstrap 稳定性及验证集残差筛查；仍不是机械约束证明。",
              "当前只记忆一个局部状态，不生成未经验证的阶段转换边。",
              "", "![方法比较](graph_response_comparison.png)"]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(output), "aggregate": summary}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[.01, .02])
    parser.add_argument("--feedback", choices=("position", "velocity"), default="position")
    parser.add_argument("--duration", type=float, default=.5)
    parser.add_argument("--no-control", action="store_true")
    args = parser.parse_args()
    if (not np.isfinite(args.amplitudes).all() or any(a <= 0 or a > .04 for a in args.amplitudes)
            or not np.isfinite(args.duration) or args.duration < .25):
        parser.error("Use finite amplitudes in (0, .04] rad and duration >= .25 s")
    output = (args.output or PROJECT_ROOT / "experiments" /
              f"graph_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    probe = MujocoJointProbe(args.model, duration=args.duration, radius=max(args.amplitudes) * 1.5)
    try:
        cases = []
        for amplitude in args.amplitudes:
            for seed in args.seeds:
                case_path = output / f"amplitude_{amplitude:.3f}_seed_{seed}"
                cases.append(run_case(probe, seed, amplitude, args.feedback, not args.no_control, case_path))
        metadata = probe.metadata()
        metadata["physical_joint_edges"] = [list(edge) for edge in probe.physical_edges]
        summarize(output, cases, metadata, not args.no_control)
    finally:
        probe.close()


if __name__ == "__main__":
    main()
