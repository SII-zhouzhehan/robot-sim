# Three-Graph Control for a Closed-Chain Morphing Robot

This repository contains research code and simulation tools for a closed-chain
morphing mobile robot. The research focus is discovering and validating
stage-dependent mechanical constraints with a three-layer graph representation.

- **Brain** handles FSM navigation and supports PPO-based policy exploration.
- **Cerebellum** coordinates swerve-module control and whole-body displacement
  while shape change is underway. Balancing locomotion and reconfiguration
  concurrently is a central challenge because morphing changes wheel geometry,
  support conditions, and motion response.
- **Spinal layer** explores and optimizes three graph layers representing
  effective topology modes, within-stage joint constraints, and transitions.

Phi remains a useful coordinate along a known reference path. It cannot
represent the full reconfiguration space and is not the spinal controller.
Graph-based discovery and simulation validation are the algorithmic focus.

> **Source-only public release**
>
> The robot URDF, CAD meshes, raw datasets, trained weights, and raw experiment
> outputs are confidential and intentionally excluded from Git. Reviewed,
> geometry-free benchmark summaries may be published as Markdown. A fresh clone
> is useful for reading and developing the control code, but it cannot reproduce
> the complete MuJoCo experiments without the private local assets.

## Research Question

How can a robot discover and use its stage-dependent mechanical constraints to
coordinate shape change with locomotion, including transitions where joint
couplings, allowable motion, or accessible modes change?

## Three-Layer Control Architecture

### Brain: FSM Navigation and PPO Exploration

The brain selects navigation behavior and task-level goals. The scripted FSM
provides a controllable baseline, while PPO supports exploration of task and
transition decisions. PPO should select goals or graph modes, not directly
command every motor.

### Cerebellum: Locomotion and Morphing Coordination

The cerebellum translates upper-layer goals into swerve-module commands and
whole-body displacement while the robot changes shape. It must balance
translation, turning, wheel geometry, support, and transition progress
concurrently. Stop-then-morph can serve as a baseline, while coordinated
movement during reconfiguration is a key research challenge.

### Spinal Layer: Three-Layer Graph Discovery and Control

The spinal layer is the main algorithmic research focus:

1. **Effective topology-mode graph** groups parts with stable relative pose and
   records active constraints and remaining motion.
2. **Within-stage constraint graph** represents joint couplings and admissible
   motion within a mode.
3. **Stage-transition graph** records reachable modes, transition conditions,
   and validated outcomes.

The graphs are inferred from model feedback and checked through simulation.
Phi can initialize or index a known reference trajectory, but it is not a
complete morphology representation. A GNN decoder is not the current method
focus.

## Benchmark Tasks

| Scene | Purpose |
| --- | --- |
| A | narrow-passage traversal with morphology switching |
| B | random multi-waypoint trajectory tracking |
| C | a compound course with several obstacle types |

The benchmark stack combines task generation, the scripted or learned brain,
MPC, swerve kinematics, morphology commands, and MuJoCo execution.

## Current Evidence

Linux workstation validation on 2026-10-26: 67 tests and one subtest passed;
formula and GNN controllers completed scenes A/B/C without obstacle collisions.
CPU/CUDA GNN short training, PPO save/reload, eight-motor playback, EGL rendering,
and all five loop-search methods with three seeds were also exercised. See the
local [validation record](artifacts/maintenance/20261026/linux_validation/README.md)
and [Linux loop-search results](experiments/closed_loop_search_linux_20261026/RESULTS.md).
These records contain local experiment outputs and remain outside public Git.

The MuJoCo backend has been verified locally with the private robot assets:
26 automated tests plus one subtest pass, the model loads with 19 actuators,
headless A/B/C evaluation runs, PPO environment stepping, and the native macOS
viewer all complete successfully. These checks validate the implementation and
interfaces; they are not a claim of benchmark success across randomized trials.

### MuJoCo Obstacle Evaluation

The legacy MuJoCo evaluator runs the scripted brain, MPC, morphology controller, and
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
|   |-- models/          # legacy MorphGNN models and graph datasets
|   `-- utils/           # morphology graph and coordinate utilities
|-- scripts/             # data, training, evaluation, and visualization tools
|-- tests/               # graph-contract and model-interface tests
|-- data/README.md       # private-data policy and expected local layout
|-- MUJOCO.md            # MuJoCo setup and run instructions
|-- CLOSED_LOOP_SEARCH.md # closed-loop model and exploration notes
|-- SINGULARITY_GRAPH_PIPELINE_2026.md # current research pipeline
`-- requirements.txt
```

## Private Local Prerequisites

Full simulation runs expect the confidential robot description to be restored
locally under:

```text
data/my_robot/
```

Legacy GNN evaluation and GUI inference additionally require a local checkpoint under
`experiments/`. Training and evaluation scripts write datasets and outputs to
ignored directories. Do not force-add these files to Git.

## Environment

Use Python 3.11 in an isolated Conda environment on the Linux workstation.
Windows x86-64 and Apple Silicon macOS remain supported development platforms.
The shared requirements contain MuJoCo, graph exploration/PPO dependencies, Jupyter,
and debugging tools. MuJoCo is the only physics backend and uses its official
Python wheels on Linux, Windows and Apple Silicon macOS.

Linux and Windows use `python` for headless runs and native windows. On macOS,
the passive native viewer must be launched with `mjpython`.

```bash
conda env create -f environment.yml
conda activate robot-sim
python scripts/setup_environment.py
python -m pip check
python -m pytest tests -q
```

The installer reads the same requirements on all supported systems.

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
dependencies have platform markers and are skipped on Linux and macOS. PyTorch is
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

Legacy MorphGNN training selects CUDA when available and otherwise uses CPU. MuJoCo
control inference and PPO default to CPU; Apple's MPS backend is not selected
automatically. PyG's optional compiled extensions are not required
for the current GATv2 models. CPU is the baseline for debugging.

For notebooks, register the environment once and select **Python (robot-sim)**:

```bash
python -m ipykernel install --user --name robot-sim --display-name "Python (robot-sim)"
python -m jupyterlab
```

In VS Code or PyCharm, select the Linux `robot-sim` Conda interpreter and use
the repository root as the working directory. On macOS, MuJoCo scripts using
`mujoco.viewer.launch_passive` must be launched with the environment's
`mjpython` command. To open the standalone MuJoCo viewer on Linux:

```bash
python -m mujoco.viewer
```

The original implementation was verified locally on 2026-10-08 with macOS
26.6 ARM64 and Python 3.11.17:
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
python scripts/run_mujoco_gui.py --source formula
```

On macOS, use `mjpython` for this native-window command.

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
   treating target morphology commands as measured physical state.
2. Condition MPC geometry, constraints, and residual dynamics on morphology.
3. Complete the baseline and layer-wise ablation matrix with repeated trials
   and confidence intervals.
4. Add system identification and simulation randomization for mass, friction,
   delay, backlash, sensor bias, and wheel slip.
5. Deploy the same closed loop on the physical robot and report synchronized
   morphology, tracking, transition, energy, robustness, and success metrics.
6. Extend scalar Phi to `z_phi` only after collecting multiple morphology
   families, connection graphs, or structural-fault cases.

See [MUJOCO.md](MUJOCO.md) for simulation setup,
[CLOSED_LOOP_SEARCH.md](CLOSED_LOOP_SEARCH.md) for closed-loop exploration, and
[SINGULARITY_GRAPH_PIPELINE_2026.md](SINGULARITY_GRAPH_PIPELINE_2026.md) for the current research direction.
