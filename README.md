# Phi-Conditioned Hierarchical Control for a Morphing Mobile Robot

This repository contains the research code for a PyBullet-based control system
for a continuously reconfigurable omnidirectional mobile robot. The central
idea is to use a shared, quantitative morphology context, called `Phi`, to
connect three control layers:

1. a brain layer for task and morphology planning;
2. a cerebellar layer for trajectory tracking and motion control;
3. a spinal layer for morphology decoding and constrained joint execution.

The current robot follows one predesigned continuous reconfiguration path, so
`Phi in [0, 3]` is a scalar coordinate on that one-dimensional morphology
manifold. It is not presented as a universal topology descriptor.

> **Source-only public release**
>
> The robot URDF, CAD meshes, raw datasets, trained weights, and raw experiment
> outputs are confidential and intentionally excluded from Git. Reviewed,
> geometry-free benchmark summaries may be published as Markdown. A fresh clone
> is useful for reading and developing the control code, but it cannot reproduce
> the complete PyBullet experiments without the private local assets.

## Research Question

Can one observable morphology context coordinate task decisions, predictive
motion control, and joint-level execution more effectively than discrete mode
switching or independently tuned controllers?

The intended closed loop is:

```text
environment observation + task goal + estimated morphology
                            |
                            v
Brain: local path and target Phi
                            |
                            v
Cerebellum: trajectory MPC and morphology-aware swerve execution
                            |
                            v
Spinal layer: MorphGNN decoding, lock constraints, joint targets
                            |
                            v
                robot state and morphology feedback
```

The present code establishes the feed-forward execution path in simulation.
Explicitly conditioning the MPC model on measured morphology and closing the
morphology feedback loop on the physical robot are the next research steps.
The resulting system will test whether sharing the same morphology context
across all three layers improves tracking, transition stability, robustness,
and task success.

## Current Phi Representation

The implemented scalar `Phi` parameterizes four regions along a fixed
reconfiguration path:

| Phi range | Configuration region |
| --- | --- |
| `[0, 1)` | compact figure-eight-like configuration |
| `[1, 2]` | ring-like configuration |
| `(2, 3)` | gate-like to line-like transition |
| `3` | line-like configuration with locked joints |

Each morphology-graph node has ten input features:

```text
[7D node identity | normalized Phi | lock state | base angle]
```

For a broader family of robots, the scalar should become a learned or measured
morphology representation:

```text
z_phi = Encoder(graph, joint state, lock state, geometry, dynamics)
```

This generalization is a research objective, not a demonstrated result in the
current repository. The project uses the term *continuous reconfiguration*
unless module connectivity actually changes.

## System Architecture

### Brain Layer

`ScriptedBrain` is the current implemented upper-level baseline. It uses a
finite-state machine to select a local path segment and a target morphology.
The principal states are `CRUISE`, `PREPARE`, and `MORPHING`.

An experimental PPO pipeline is also included. It uses a 21-dimensional
observation and a three-dimensional action for planar motion and target
morphology. This branch is under development and should not yet be interpreted
as a successful learned planner.

### Cerebellar Layer

`KinematicMPC` tracks a ten-step planar reference trajectory at a nominal
`0.05 s` interval. It optimizes `[vx, vy, yaw_rate]` commands with SLSQP and
penalizes state error, control effort, and command variation.

Motion is handled in a world-referenced control pipeline because the geometric
wheel center moves relative to the base link during reconfiguration. Swerve
inverse kinematics and steering-reference calibration convert the optimized
motion into wheel commands. A morphology-aware PID gain schedule is available
for ablation experiments.

### Spinal Layer

`MorphGNN` maps the morphology graph and scalar `Phi` to seven morphing-joint
targets. It uses three GATv2 layers followed by an output MLP. Locked joints are
enforced with hard gating:

```text
q_target = q_raw * (1 - lock) + q_base * lock
```

A smaller two-layer `MorphGNNLite` model and analytical interpolation are
included as comparison methods.

## Benchmark Tasks

| Scene | Purpose |
| --- | --- |
| A | narrow-passage traversal with morphology switching |
| B | random multi-waypoint trajectory tracking |
| C | a compound course with several obstacle types |

The benchmark stack combines task generation, the scripted or learned brain,
MPC, swerve kinematics, morphology commands, and PyBullet execution.

## Current Evidence

### v0.1 Four-Obstacle Simulation Milestone (2026-09-21)

The revised headless policy achieved **79/80 successful trials (98.75%)** in
a fixed batch with new seeds and small initial-state/friction perturbations:

| Isolated obstacle | Successful trials | Sample success rate |
| --- | ---: | ---: |
| Narrow gate | 20/20 | 100% |
| Low platform | 20/20 | 100% |
| Bridge | 20/20 | 100% |
| Double cylinders | 19/20 | 95% |

This meets the declared **at least 95% per obstacle** sample-rate target for
this protocol. It does not establish a population success probability of 95%,
performance on the full compound course, or hardware reliability. The cylinder
timeout is retained: the robot cleared the obstacle but remained 0.247 m from
the goal after 40 s, outside the 0.20 m arrival tolerance.

Development exposed platform failures when turning and morphing overlapped.
Completing the sideways turn in the ring configuration before approach and
morphing resolved the platform failures in the new batch, without changing
obstacle dimensions, required Phi=2, or acceptance thresholds. The earlier
60/80 batch, all development checks, and the final 79/80 batch remain separately
reported in [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md).

### Earlier Module-Level Evidence

These numbers are preliminary development results, not final paper claims:

- the retained MorphGNN training record reports a validation loss of
  approximately `0.00216` and a joint-angle MAE of approximately `0.026 rad`;
- three archived Scene B simulation runs reached the task criterion, with mean
  path errors in the approximate range of `0.019-0.030 m`;
- Phi-aware PID gain scheduling has shown only small and inconsistent gains;
- saved PPO policies have not yet demonstrated reliable benchmark completion;
- no physical-robot experiment has been completed.

The headless four-obstacle evaluation is documented in
[BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md). It reports all trials in each
fixed batch, per-obstacle success rates, failure reasons, and uncertainty.
Publishing source and a summary provides an auditable project record; it does
not make private-geometry experiments independently reproducible from a clone.

### Headless Obstacle Evaluation

The runner tests the four obstacle types from Scene C **individually**, using
the analytical decoder, scripted morphology selection, MPC, and swerve
execution. It pauses translation during morphology transitions and retains
the existing geometric heading estimate near Phi=3. This is an explicit
headless evaluation policy, not a claim that the historical GUI or learned
GNN/PPO policy achieved the same results.

```bash
# Four nominal development checks
python scripts/eval_four_obstacles.py --trials 1 --workers 4 --nominal --output experiments/four_smoke
# A fixed batch with the revised platform preparation (80 total)
python scripts/eval_four_obstacles.py --trials 20 --workers 4 --seed 2026093100 --align-platform --output experiments/four_batch
```

Each trial starts with independent uniform x/y offsets of up to 0.03 m,
heading offsets up to 3 degrees, and ground/wheel friction in [0.9, 1.1].
Obstacle dimensions remain those of the existing scene. These are small local
perturbations, not evidence of general terrain robustness or hardware success.

`--align-platform` completes the sideways turn while the robot is still in its
ring configuration, before approaching and morphing for the platform. Omitting
it reproduces the original evaluation policy. `--platform-phi` and
`--platform-heading-deg` are explicit development overrides recorded in the
protocol; their defaults retain Phi=2 and a 90-degree platform approach.

Success requires the full robot AABB to clear the obstacle, the centroid to
remain in the designated crossing corridor, and the final goal distance to be
below 0.20 m. The commanded morphology must be reached, all seven actual joint
errors must be at most 0.05 rad, and at least two wheels must contact the ground
for ten consecutive control steps. Significant robot-obstacle contact
(force >1 N or penetration >2 mm) at any physics tick is a failure; timeout is
40 simulated seconds. This protocol treats the platform as a clearance/straddle
task, not an intentional wheel-climbing task. It does not certify mechanical
stability, general self-collision safety, or physical-robot reliability.

The runner uses PyBullet DIRECT without cameras or sleeps. It does not use the
GUI demonstration's per-tick joint-state reset patch. The run directory must
be new: protocol, seeds, source hashes, all outcomes, and private traces are
retained instead of overwriting or retaining only successful trials. There is
no stop-when-95%-is-reached rule. Twenty successes out of twenty is a sample
rate, not a 95% lower confidence bound on the population success probability.

## Repository Layout

```text
robot-sim/
|-- src/
|   |-- brains/          # upper-level interfaces and scripted policy
|   |-- controllers/     # MPC, PID, and swerve kinematics
|   |-- envs/            # PyBullet robot and benchmark environments
|   |-- evaluation/      # explicit obstacle traversal acceptance rules
|   |-- models/          # MorphGNN models and graph datasets
|   `-- utils/           # morphology graph and coordinate utilities
|-- scripts/             # data, training, evaluation, and visualization tools
|-- tests/               # graph-contract and model-interface tests
|-- data/README.md       # private-data policy and expected local layout
|-- PAPER_DIRECTION.md   # paper thesis, hypotheses, and required baselines
|-- PAPER_READING_LIST.md # focused literature-reading plan
|-- BENCHMARK_RESULTS.md # reviewed aggregate results; no private geometry
|-- total_plan.md        # implementation and experiment plan
`-- requirements.txt
```

## Private Local Prerequisites

Full simulation runs expect the confidential robot description to be restored
locally under:

```text
data/my_robot/
```

GNN evaluation and GUI inference additionally require a local checkpoint under
`experiments/`. Training and evaluation scripts write datasets and outputs to
ignored directories. Do not force-add these files to Git.

## Environment

Python 3.9 is the currently tested interpreter version. The project was
developed in a Conda environment with PyBullet, NumPy, SciPy, PyTorch, PyTorch
Geometric, Gym, Matplotlib, and Stable-Baselines3.

```bash
conda create -n robot-sim python=3.9 -y
conda activate robot-sim
pip install -r requirements.txt
```

For CUDA installations, install mutually compatible PyTorch and PyTorch
Geometric builds for the local driver and toolkit.

## Main Workflows

Generate the supervised morphology dataset:

```bash
python scripts/collect_gnn_data.py
```

Train the full or lightweight morphology decoder:

```bash
python scripts/train.py --model full
python scripts/train.py --model lite
```

Run the interactive PyBullet controller:

```bash
python scripts/run_mpc_gui.py
```

Run the three-layer scripted benchmark:

```bash
python scripts/evaluate_brain.py
```

Train or evaluate the experimental PPO brain:

```bash
python scripts/train_drl_brain.py
python scripts/evaluate_drl.py
```

## Evaluation and Ablation

```bash
python scripts/eval_gnn_ablation.py
python scripts/eval_gnn_attention.py
python scripts/eval_gnn_robustness.py
python scripts/eval_mpc_benchmark.py
python scripts/eval_pid_ablation.py
python scripts/eval_scene_b_tracking.py
python scripts/test_mpc_pure_kinematics.py
python scripts/test_mpc_pybullet.py
```

The paper-oriented comparison matrix should cover:

- discrete morphology ID versus scalar Phi versus a higher-dimensional
  morphology description;
- Phi in the spinal layer only versus the spinal and cerebellar layers versus
  all three layers;
- analytical mapping, lookup or spline interpolation, MLP, GCN, and GATv2;
- fixed control, per-configuration tuning, discrete gain scheduling, and
  continuous Phi-conditioned control;
- target Phi used as state versus measured or estimated actual morphology;
- nominal operation versus payload, friction, delay, slip, and joint-fault
  perturbations.

## Tests

The source-level tests do not require private robot geometry:

```bash
python -m unittest discover -s tests -v
```

They currently verify the seven-node morphology graph contract, graph-to-robot
joint ordering, full/lite model output interfaces, obstacle traversal criteria,
and notebook hygiene. The
notebook check prevents committed execution counts and embedded outputs.

## Research Roadmap

1. Estimate actual morphology from encoder and joint feedback instead of
   treating the target Phi command as the physical state.
2. Condition MPC geometry, constraints, and residual dynamics on morphology.
3. Complete the baseline and layer-wise ablation matrix with repeated trials
   and confidence intervals.
4. Add system identification and simulation randomization for mass, friction,
   delay, backlash, sensor bias, and wheel slip.
5. Deploy the same closed loop on the physical robot and report synchronized
   morphology, tracking, transition, energy, robustness, and success metrics.
6. Extend scalar Phi to `z_phi` only after collecting multiple morphology
   families, connection graphs, or structural-fault cases.

See [PAPER_DIRECTION.md](PAPER_DIRECTION.md) for the core thesis and
[PAPER_READING_LIST.md](PAPER_READING_LIST.md) for the curated reading plan.
