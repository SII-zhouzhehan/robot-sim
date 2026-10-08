# Phi-Conditioned Hierarchical Control for a Morphing Mobile Robot

This repository contains the research code for a MuJoCo-based control system
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
> the complete MuJoCo experiments without the private local assets.

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
MPC, swerve kinematics, morphology commands, and MuJoCo execution.

## Current Evidence

The MuJoCo backend has been verified locally with the private robot assets:
26 automated tests plus one subtest pass, the model loads with 19 actuators,
headless A/B/C evaluation runs, PPO environment stepping, and the native macOS
viewer all complete successfully. These checks validate the implementation and
interfaces; they are not a claim of benchmark success across randomized trials.

### MuJoCo Obstacle Evaluation

The MuJoCo evaluator runs the scripted brain, MPC, morphology controller, and
contact checks in scenes A, B, and C. Reports are written to the ignored
`experiments/mujoco_evaluations/` directory.

```bash
python scripts/evaluate_mujoco.py --scene A --seconds 60 --source formula
python scripts/evaluate_mujoco.py --scene B --seconds 60 --source formula
python scripts/evaluate_mujoco.py --scene C --seconds 100 --source formula
```

Use `--source gnn --gnn-checkpoint <path>` to evaluate a trained morphology
decoder. Use `mjpython ... --gui` on macOS for a native viewer. The report
records goal completion, obstacle contacts, support contacts, morphology error,
and traversal status; a short smoke run is not a benchmark success claim.

## Repository Layout

```text
robot-sim/
|-- src/
|   |-- brains/          # upper-level interfaces and scripted policy
|   |-- controllers/     # MPC, PID, and swerve kinematics
|   |-- envs/            # MuJoCo robot and benchmark environments
|   |-- evaluation/      # explicit obstacle traversal acceptance rules
|   |-- models/          # MorphGNN models and graph datasets
|   `-- utils/           # morphology graph and coordinate utilities
|-- scripts/             # data, training, evaluation, and visualization tools
|-- tests/               # graph-contract and model-interface tests
|-- data/README.md       # private-data policy and expected local layout
|-- PAPER_DIRECTION.md   # paper thesis, hypotheses, and required baselines
|-- PAPER_READING_LIST.md # focused literature-reading plan
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

Use Python 3.11 in an isolated Conda environment on Windows x86-64 or macOS
Apple Silicon. Install the native ARM64 Anaconda distribution on Apple Silicon.
The shared requirements contain MuJoCo, MorphGNN/PPO dependencies, Jupyter,
and debugging tools. MuJoCo is the only physics backend and uses its official
Python wheels on both Windows and Apple Silicon macOS.

On macOS, the passive native viewer must be launched with `mjpython`; headless
runs and Windows native windows use `python`.

```bash
conda env create -f environment.yml
conda activate robot-sim
python scripts/setup_environment.py
python -m pip check
python -m pytest tests -q
```

The installer reads the same requirements on both systems.

Alternatively, create the interpreter and install the same requirements. This
explicit channel override also avoids the default-channel Terms of Service
check in fresh Anaconda installations that run it before reading the YAML:

```bash
conda create -n robot-sim -c conda-forge --override-channels python=3.11 pip -y
conda env config vars set -n robot-sim PYTHONUTF8=1
conda activate robot-sim
python scripts/setup_environment.py
```

Conda manages the interpreter; pip manages the project's Python packages.
Keep project packages out of `base`. NumPy is pinned for consistent scientific
and training dependencies. The robot environment uses Gymnasium. Windows-only
dependencies have platform markers and are skipped on macOS. PyTorch is
pinned without a CUDA suffix so pip can install the native platform package.
For a Windows CPU-only build, use:

```bash
python -m pip install --force-reinstall --no-deps torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

For Windows with an NVIDIA driver compatible with CUDA 11.8, install that
build before the shared requirements (or replace the installed build):

```bash
python -m pip install --force-reinstall --no-deps torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python scripts/setup_environment.py
```

Existing scripts automatically select CPU on macOS; they do not automatically
select Apple's MPS backend. PyG's optional compiled extensions are not required
for the current GATv2 models. CPU is the baseline for debugging.

For notebooks, register the environment once and select **Python (robot-sim)**:

```bash
python -m ipykernel install --user --name robot-sim --display-name "Python (robot-sim)"
python -m jupyterlab
```

In VS Code or PyCharm, select the `robot-sim` Conda interpreter and use the
repository root as the working directory. On macOS, MuJoCo scripts using
`mujoco.viewer.launch_passive` must be launched with the environment's
`mjpython` command. To open the standalone MuJoCo viewer:

```bash
python -m mujoco.viewer
```

Verified locally on 2026-10-08 with macOS 26.6 ARM64 and Python 3.11.17:
dependency consistency, the existing test suite, GNN checkpoint inference and
training, MPC, robot URDF loading and PPO environment stepping, the existing
MuJoCo physics, offscreen rendering, and passive GUI. Windows
Python 3.11 package availability and platform markers were checked; Windows
runtime validation still requires a Windows machine.

## Main Workflows

See [scripts/README.md](scripts/README.md) for the purpose of each retained entry point.

Generate the supervised morphology dataset:

```bash
python scripts/collect_gnn_data.py
```

Train the full or lightweight morphology decoder:

```bash
python scripts/train.py --model full
python scripts/train.py --model lite
```

Run the interactive MuJoCo controller:

```bash
mjpython scripts/run_mujoco_gui.py --source formula
```

Run the three-layer scripted benchmark:

```bash
python scripts/evaluate_mujoco.py --scene C --seconds 40 --source formula
```

Train or evaluate the experimental PPO brain:

```bash
python scripts/train_mujoco_brain.py --scene B --source formula
python scripts/evaluate_mujoco.py --mode ppo --policy experiments/mujoco_ppo_<time>/ppo_mujoco.zip
```

## Evaluation and Ablation

```bash
python scripts/eval_gnn_ablation.py
python scripts/eval_gnn_attention.py
python scripts/eval_mpc_benchmark.py
python scripts/test_mpc_pure_kinematics.py
python scripts/test_phi3_stability.py
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
