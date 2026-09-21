"""Export only reviewed scalar outcomes, never geometry, paths, or traces."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluation.obstacle_protocol import wilson_interval

KINDS = ("narrow_gate", "low_platform", "bridge", "cylinders")
REASONS = {"success", "timeout", "obstacle_contact", "invalid_state", "bypassed_corridor", "execution_error"}


def read_batch(folder):
    protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    records = summary["trials"]
    expected = {(r["kind"], r["seed"]) for r in protocol["seeds"]}
    actual = {(r["kind"], r["seed"]) for r in records}
    if expected != actual or len(actual) != len(records):
        raise ValueError("Missing or duplicate trials; refusing to export an incomplete batch")
    for row in records:
        if type(row["success"]) is not bool or row["reason"] not in REASONS:
            raise ValueError("Unexpected outcome schema")
        if row["success"] != (row["reason"] == "success"):
            raise ValueError("Inconsistent success flag")
    return protocol, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--development", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, default=ROOT / "BENCHMARK_RESULTS.md")
    args = parser.parse_args()
    lines = ["# Four-Obstacle Headless Simulation Results", "",
             "Evaluation date: 2026-09-21. This is a source-only, simulation-only research record.",
             "Robot geometry, private logs, trajectories, weights, and hardware drawings are not included.", "",
             "## Scope and Acceptance", "",
             "Each obstacle is tested in isolation using the analytical Phi decoder, scripted brain,",
             "MPC, and swerve execution. This is not a compound-course, learned-policy, or physical-robot result.",
             "The obstacle geometry and success criteria are unchanged between the randomized batches.", "",
             "The nominal smoke batch is a development check. Randomized batches perturb initial x/y",
             "by up to 0.03 m, initial yaw by up to 3 degrees, and wheel/ground friction over [0.9,1.1].",
             "All batch sizes and seeds are written before running; failures remain in the denominator.", "",
             "Success requires full-body clearance, no corridor bypass, goal distance <0.20 m,",
             "actual seven-joint error <=0.05 rad relative to the required morphology, and at least",
             "two ground-contacting wheels for ten consecutive 0.05 s control steps. Contact force >1 N",
             "or penetration >2 mm with the obstacle at any 240 Hz physics tick fails the trial.",
             "The timeout is 40 simulated seconds. Platform contact is forbidden in this straddling task.",
             "These criteria do not certify general self-collision safety or hardware stability.", "",
             "The headless policy pauses motion during shape changes and uses a geometric heading",
             "estimate near Phi=3. It uses finite motor commands, without the GUI demo's per-tick",
             "joint-state-reset patch. The revised policy additionally aligns the ring configuration",
             "sideways before platform approach and morphing. The original required Phi=2 is retained.", ""]
    latest_protocol, latest_records = read_batch(args.runs[-1])
    total_successes = sum(row["success"] for row in latest_records)
    headline = ["## Latest Fixed Batch", "",
                f"**{total_successes}/{len(latest_records)} successful trials "
                f"({total_successes/len(latest_records):.2%}).**",
                "Per-obstacle counts, uncertainty, and every failure are listed below.",
                "The latest batch uses new seeds after development; no additional trials were appended",
                "to dilute failures. The target is a per-class sample rate >=95%, not strictly >95%",
                "in every class and not a confidence guarantee on the underlying probability.", ""]
    lines[5:5] = headline
    latest_met = False
    for folder in args.runs:
        protocol, records = read_batch(folder)
        lines += [f"## {folder.name}", "", f"Trials: {len(records)}. Nominal only: {protocol['nominal']}. "
                  f"Platform pre-alignment: {protocol.get('align_platform', False)}. "
                  f"Platform Phi: {protocol.get('platform_phi', 2.0):g}.", "",
                  "| Obstacle | Successes / trials | Sample rate | 95% Wilson interval | Failures |",
                  "| --- | ---: | ---: | ---: | --- |"]
        latest_met = True
        for kind in KINDS:
            selected = [r for r in records if r["kind"] == kind]
            n = len(selected)
            k = sum(r["success"] for r in selected)
            low, high = wilson_interval(k, n)
            failures = Counter(r["reason"] for r in selected if not r["success"])
            failure_text = ", ".join(f"{key}: {value}" for key, value in sorted(failures.items())) or "None"
            rate = k/n if n else 0.
            latest_met &= n > 0 and rate >= .95
            lines.append(f"| {kind} | {k}/{n} | {rate:.1%} | [{low:.1%}, {high:.1%}] | {failure_text} |")
        lines += ["", f"At least 95% sample success in **every** obstacle class: **{'YES' if latest_met else 'NO'}**.", "",
                  "A 100% sample rate from 20 trials does not establish a true success probability",
                  "of at least 95% at 95% confidence. Rates are not pooled across controller versions.", "",
                  "Record fingerprints (raw records retained privately):", ""]
        for name in ("protocol.json", "summary.json"):
            lines.append(f"- `{name}` SHA-256: `{hashlib.sha256((folder/name).read_bytes()).hexdigest()}`")
        for name in ("scripts/eval_four_obstacles.py", "src/evaluation/obstacle_protocol.py"):
            lines.append(f"- Evaluated `{name}` SHA-256: `{protocol['source_hashes'][name]}`")
        lines += ["", "<details>", "<summary>All trial outcomes (sanitized scalar fields)</summary>", "",
                  "| Obstacle | Seed | Outcome | Simulated seconds |",
                  "| --- | ---: | --- | ---: |"]
        for row in records:
            lines.append(f"| {row['kind']} | {row['seed']} | {row['reason']} | {row.get('duration_s', 0):.2f} |")
        lines += ["", "</details>", ""]
    if args.development:
        lines += ["## Development Diagnostics", "",
                  "The following nominal parameter checks informed the revised controller. They are",
                  "not held-out trials and are excluded from every randomized success-rate denominator.", "",
                  "| Check | Seed | Platform Phi | Heading (deg) | Pre-align | Outcome |",
                  "| --- | ---: | ---: | ---: | --- | --- |"]
        for folder in args.development:
            rows = json.loads((folder / "development_results.json").read_text(encoding="utf-8"))
            for row in sorted(rows, key=lambda r: r["seed"]):
                lines.append(f"| {folder.name} | {row['seed']} | {row['required_phi']:g} | "
                             f"{row['platform_heading_deg']:g} | {row.get('align_platform', False)} | {row['reason']} |")
        lines += [""]
    lines += ["## Reproduction", "", "Private local robot assets and the documented Python environment are required.",
              "See the README for protocol details. Each output directory must be new.", "", "```bash",
              "python scripts/eval_four_obstacles.py --trials 20 --workers 4 --seed 2026092100 --output experiments/reproduce_v1",
              "python scripts/eval_four_obstacles.py --trials 20 --workers 4 --seed 2026093100 --align-platform --output experiments/reproduce_v2",
              "python -m unittest discover -s tests -v", "```", "",
              "The repository is evidence of implemented evaluation and reported results, not an",
              "independently reproducible open-geometry benchmark. Hardware validation remains pending.", ""]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Exported {args.output.name}; latest per-class sample target met: {latest_met}")


if __name__ == "__main__":
    main()
