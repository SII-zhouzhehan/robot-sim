"""Discover a local response graph on the existing robot, starting at Phi=0.

python scripts/demo_motion_discovery.py
python scripts/demo_motion_discovery.py --gui --strategies graph_active

On macOS, use `mjpython` for the GUI command.
"""
import argparse
import csv
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
from exploration import ActiveMotionLearner
from exploration.mujoco_probe import MujocoJointProbe


def rmse(predicted, measured):
    return float(np.sqrt(np.mean((np.asarray(predicted) - measured) ** 2)))


def heldout_commands(probe, amplitude, count, seed):
    rng = np.random.default_rng(seed)
    result = []
    while len(result) < count:
        direction = rng.normal(size=len(probe.names))
        direction = np.where(probe.upper <= 1e-10, -abs(direction), direction)
        direction = np.where(probe.lower >= -1e-10, abs(direction), direction)
        bounded = ActiveMotionLearner.bounded_direction(direction, probe.lower, probe.upper, amplitude)
        if bounded is not None:
            result.append(bounded)
    return np.asarray(result)


def draw_graph(ax, graph):
    n = len(graph["nodes"])
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    xy = np.column_stack((np.cos(theta), np.sin(theta)))
    for i, j in graph["physical_edges"]:
        ax.plot(*xy[[i, j]].T, color="#c1c9d2", linewidth=1.5, zorder=0)
    for edge in graph["response_edges"]:
        source, target = edge["source"], edge["target"]
        ax.annotate("", xy=xy[target] * .85, xytext=xy[source] * .85,
                    arrowprops=dict(arrowstyle="->", color="#ea923b", linewidth=1.2))
    for i, node in enumerate(graph["nodes"]):
        color = "#377fa6" if node["status"] == "response_observed" else "#9098a1"
        ax.scatter(*xy[i], s=490, c=color, zorder=2)
        ax.text(*xy[i], node["name"].replace("joint", "j"), ha="center", va="center", color="white")
    for k, factor in enumerate(graph["coupled_groups"]):
        members = factor["members"]
        center = np.mean(xy[members], axis=0) * .5
        for i in members:
            ax.plot(*np.stack((center, xy[i])).T, color="#9462ad", linewidth=2)
        ax.scatter(*center, marker="s", c="#9462ad", s=140)
        ax.text(center[0], center[1] + .14, f"factor {k + 1}", ha="center", fontsize=8)
    ax.set(xlim=(-1.4, 1.4), ylim=(-1.4, 1.4), aspect="equal")
    ax.axis("off")
    ax.set_title("Grey: model adjacency; orange: observed influence\nPurple: candidate coupling factors")


def plots(output, results, measured):
    primary = next(iter(results.values()))
    graph = primary["graph"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    matrix = np.asarray(graph["response_matrix"])
    limit = max(float(np.max(abs(matrix))), 1e-3)
    im = axes[0, 0].imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit)
    names = [n["name"].replace("joint", "j") for n in graph["nodes"]]
    axes[0, 0].set_xticks(range(len(names)), names)
    axes[0, 0].set_yticks(range(len(names)), names)
    axes[0, 0].set(xlabel="Commanded joint", ylabel="Observed joint", title="Learned local response M")
    fig.colorbar(im, ax=axes[0, 0], label="rad / rad")
    draw_graph(axes[0, 1], graph)
    for label, result in results.items():
        history = result["history"]
        axes[1, 0].plot([h["probe"] for h in history], np.rad2deg([h["heldout_rmse_rad"] for h in history]),
                        label=label)
    null_error = np.rad2deg(rmse(np.zeros_like(measured), measured))
    axes[1, 0].axhline(null_error, color="#777777", linestyle="--", label="predict no response")
    axes[1, 0].set(xlabel="Training probes", ylabel="Held-out displacement RMSE (deg)",
                   title="Same initial state, amplitude budget and held-out commands")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=.25)
    axes[1, 1].bar(np.arange(1, len(names) + 1), graph["singular_values"], color="#377fa6")
    axes[1, 1].axhline(primary["rank_threshold"], color="#ea923b", linestyle="--")
    axes[1, 1].set(xlabel="Response singular value index", ylabel="Sensitivity",
                   title=f"Observed response rank: {graph['observed_response_rank']}\nNot a certified mechanical DOF count")
    fig.suptitle("Phi=0 local motion discovery | position/velocity feedback | no RL", fontsize=15)
    fig.savefig(output / "discovery_summary.png", dpi=150)
    plt.close(fig)


def save_trials(output, records):
    np.savez_compressed(output / "probe_traces.npz",
                        increments=np.array([t["increment"] for t in records]),
                        positions=np.array([t["position_rad"] for t in records]),
                        velocities=np.array([t["velocity_rad_s"] for t in records]),
                        targets=np.array([t["target_rad"] for t in records]),
                        labels=np.array([t["label"] for t in records]))
    with (output / "probes.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["label", "increment_rad", "displacement_rad", "integrated_velocity_rad"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for trial in records:
            writer.writerow({"label": trial["label"],
                             "increment_rad": json.dumps(trial["increment"].tolist()),
                             "displacement_rad": json.dumps(trial["displacement"].tolist()),
                             "integrated_velocity_rad": json.dumps(trial["integrated_velocity"].tolist())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--feedback", choices=("position", "velocity"), default="velocity")
    parser.add_argument("--budget", type=int, default=28)
    parser.add_argument("--validation-probes", type=int, default=12)
    parser.add_argument("--amplitude", type=float, default=.04, help="Total command increment norm, radians")
    parser.add_argument("--duration", type=float, default=.25)
    parser.add_argument("--settle-seconds", type=float, default=2)
    parser.add_argument("--rank-threshold", type=float, default=.08)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--strategies", nargs="+", choices=("graph_active", "information_only", "random"),
                        default=["graph_active", "information_only", "random"])
    parser.add_argument("--gui", action="store_true", help="Run with a native MuJoCo window")
    args = parser.parse_args()
    if (args.budget < 1 or args.validation_probes < 1 or args.amplitude <= 0
            or args.duration <= 0 or args.settle_seconds < 0 or args.rank_threshold <= 0):
        parser.error("Counts, amplitude, duration and threshold must be positive")
    output = (args.output or PROJECT_ROOT / "experiments" /
              f"motion_discovery_phi0_{datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    probe = MujocoJointProbe(args.model, duration=args.duration, settle_seconds=args.settle_seconds,
                            radius=args.amplitude * 1.5, gui=args.gui)
    try:
        if args.budget < len(probe.names):
            parser.error("Budget must cover at least one independent command per joint")
        records = [dict(probe.baseline, label="null_baseline")]
        commands = heldout_commands(probe, args.amplitude, args.validation_probes, args.seed + 1000)
        # Held-out measurements stay outside every learner and action selector.
        measured = []
        for i, command in enumerate(commands):
            trial = probe.run(command)
            records.append(dict(trial, label=f"heldout_{i}"))
            measured.append(probe.response(trial, args.feedback))
        measured = np.array(measured)
        results = {}
        for strategy in args.strategies:
            learner = ActiveMotionLearner(probe.names, probe.physical_edges,
                                          rank_threshold=args.rank_threshold, seed=args.seed)
            random = np.random.default_rng(args.seed)
            history = []
            for index in range(args.budget):
                if strategy == "random" and learner.command_rank == learner.n:
                    options = learner.candidates(probe.lower, probe.upper, args.amplitude)
                    candidate = options[int(random.integers(len(options)))]
                    score = None
                else:
                    candidate, score = learner.choose(probe.lower, probe.upper, args.amplitude,
                                                      use_graph=strategy != "information_only")
                trial = probe.run(candidate.increment)
                records.append(dict(trial, label=f"{strategy}_{index + 1}_{candidate.kind}"))
                learner.observe(candidate.increment, probe.response(trial, args.feedback), kind=candidate.kind)
                predicted = np.array([learner.predict(c) for c in commands])
                error = rmse(predicted, measured)
                history.append({"probe": index + 1, "kind": candidate.kind, "score": score,
                                "heldout_rmse_rad": error, "command_rank": learner.command_rank,
                                "response_rank": learner.observed_rank})
                if (index + 1) % 7 == 0 or index + 1 == args.budget:
                    print(f"{strategy}: {index + 1}/{args.budget}, "
                          f"held-out RMSE={np.rad2deg(error):.4f} deg, "
                          f"response rank={learner.observed_rank}", flush=True)
            # Reuse the learned factors to coordinate an independent request.
            requested = commands[0]
            reuse = learner.coordinated_request(requested, probe.lower, probe.upper, args.amplitude)
            reuse_report = {"executed": reuse is not None}
            if reuse is not None:
                trial = probe.run(reuse)
                records.append(dict(trial, label=f"{strategy}_learned_relation_reuse"))
                response = probe.response(trial, args.feedback)
                reuse_report.update(command_rad=reuse.tolist(), response_rad=response.tolist(),
                                    prediction_rmse_rad=rmse(learner.predict(reuse), response),
                                    projected_command_residual=float(np.linalg.norm(learner.constraints @ reuse)))
            results[strategy] = {"graph": learner.graph(), "history": history,
                                 "heldout_rmse_rad": error, "rank_threshold": args.rank_threshold,
                                 "relation_reuse": reuse_report}
            (output / f"{strategy}_graph.json").write_text(json.dumps(learner.graph(), indent=2) + "\n")
        parity = max(float(np.max(abs(t["displacement"] - t["integrated_velocity"]))) for t in records)
        report = {"algorithm": "active local system identification; not reinforcement learning",
                  "feedback": args.feedback, "metadata": probe.metadata(), "seed": args.seed,
                  "training_probe_budget_per_strategy": args.budget,
                  "heldout_probe_count": args.validation_probes,
                  "position_velocity_integration_max_error_rad": parity,
                  "zero_prediction_rmse_rad": rmse(np.zeros_like(measured), measured),
                  "results": results,
                  "limitations": ["No real closed-chain or transition guards in the current default XML",
                                  "Local response of position servos, contacts and limits; not passive coupling",
                                  "Single-seed prototype comparison, not evidence of graph superiority",
                                  "No stage count, group count, Phi labels, torque or current observations supplied"]}
        save_trials(output, records)
        plots(output, results, measured)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        lines = ["# Phi=0 自主探索 demo 实测结果", "", "算法：主动局部辨识，不是 RL。", "",
                 f"反馈：{args.feedback}；主体关节：{len(probe.names)}；等式约束：{probe.env.model.neq}。",
                 f"每种策略 {args.budget} 次学习试探，另有 {args.validation_probes} 次共同留出验证。", "",
                 "| 策略 | 留出位移 RMSE（度） | 观察到的局部响应秩 | 图生成的组合试探数 |",
                 "|---|---:|---:|---:|"]
        for label, result in results.items():
            graph = result["graph"]
            count = sum(graph["probe_kinds"].get(k, 0) for k in ("physical_graph_pair", "learned_graph_group"))
            lines.append(f"| {label} | {np.rad2deg(result['heldout_rmse_rad']):.5f} | "
                         f"{graph['observed_response_rank']} | {count} |")
        lines += ["", "当前模型不含真实首尾闭合，不能据此确认实机耦合、阶段或解锁条件。",
                  "响应秩受伺服、摩擦、接触、限位、观测时长和阈值影响，不等同机构自由度。",
                  "图参与生成组合验证动作与协调请求；本轮对照不证明图优于其他表示。",
                  "本轮使用同一仿真快照复位，每个试探从 phi=0 的同一实测状态开始。",
                  "角度变化和转速积分的一致性最大误差：" + f"{parity:.3g} rad。", "",
                  "![探索结果](discovery_summary.png)"]
        (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
        print(json.dumps({"output": str(output), "feedback": args.feedback,
                          "equality_count": int(probe.env.model.neq),
                          "integration_error_rad": parity,
                          "rmse_deg": {k: float(np.rad2deg(v["heldout_rmse_rad"])) for k, v in results.items()}}, indent=2))
    finally:
        probe.close()


if __name__ == "__main__":
    main()
