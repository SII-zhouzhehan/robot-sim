# Script entry points

Run these scripts from the repository root in the `robot-sim` Conda environment.
The Linux workstation uses `python` for both headless runs and native MuJoCo
windows. On macOS, replace `python` with the environment's `mjpython` only for
native-window runs.

| Script | Purpose |
| --- | --- |
| `setup_environment.py` | Install shared Python 3.11 dependencies. |
| `build_mujoco_model.py` | Convert the private URDF/STL assets into MuJoCo XML and meshes. |
| `run_mujoco_gui.py` | Manual control with the native viewer and browser control panel; supports headless runs. |
| `evaluate_mujoco.py` | Run scripted MPC or a PPO policy in scene A, B, or C and save a JSON report. |
| `analyze_motor_torque.py` | Estimate all 19 joint loads through formula-based folding, compare speeds/directions, and audit CAD mass/inertia; writes local CSV/NPZ and PNG/SVG. |
| `analyze_swerve_loads.py` | Measure steering and wheel torque in start/stop, lateral, yaw, steering and 0–15° slope cases; includes wheel rpm and contact/tracking checks. |
| `analyze_pivot_friction.py` | Sweep passive pivot dry friction/damping; compare 6-second folding and 26 shape holds, tire loads, clearance and joint torque. |
| `train_mujoco_brain.py` | Train a PPO policy in a chosen MuJoCo scene. |
| `train_drl_brain.py` | Run the three-stage PPO curriculum with changing imitation rewards and observation normalization. |
| `collect_gnn_data.py` | Generate supervised morphology data along the Phi trajectory. |
| `train.py` | Train the full or lightweight MorphGNN decoder. |
| `eval_gnn_ablation.py` | Compare morphology-decoder baselines on supervised data. |
| `eval_gnn_attention.py` | Inspect trained GNN attention. |
| `eval_mpc_benchmark.py` | Compare trajectory controllers using planar kinematics. |
| `test_mpc_pure_kinematics.py` | Visualize a standalone kinematic MPC example. |
| `test_phi3_stability.py` | Inspect folding toward Phi=3 and ground support in MuJoCo. |
| `plot_pid_gain_schedule.py` | Plot the PID gains used by the controller. |
| `plot_scene_a_phi_timeline.py` | Plot the scripted brain's switching logic with synthetic inputs. |

Automated regression tests live in `tests/`; run `python -m pytest tests -q`.
Model assets, weights, reports, plots, and IDE settings stay local under the
repository's ignore rules. Legacy Phi/GNN workflows require private data or checkpoints;
use `--source formula` for MuJoCo control without trained weights.

The curriculum trainer saves `vecnorm_*.pkl` alongside its policies. Those
policies require matching observation normalization during inference. The
generic evaluator directly accepts policies from `train_mujoco_brain.py`.

For motor sizing, run `python scripts/analyze_motor_torque.py` (headless;
`mjpython` is unnecessary). This disables motor torque clamps in the analysis
model, tests 12/6/3-second folding in both directions, and samples holding
loads at 13 shapes. Results are ignored under `experiments/motor_sizing_*`.
Change `--rates` (Phi units/second) and `--settle-seconds` to match the intended
motion. Read the generated report's tracking errors and settling flags before
using its joint-output torque and RMS values to size motors/gearboxes.

The original servo can lag substantially during folding. To inspect a closer
approximation to the formula path without changing the saved model, compare
`--block-gain-scale 30 --control-steps 1` against the default run. Higher gains
can also increase contact transients: these are a sensitivity study, not
validated hardware-controller gains. For half-timestep checking at the same
240 Hz control rate, use `--timestep-divisor 2 --control-steps 2`.

For the four swerve units, run `python scripts/analyze_swerve_loads.py`.
The default model mass is retained, with no payload, 0.4 m/s translation,
0.5 s acceleration/deceleration and slopes of 0/5/10/15 degrees. This analysis
disables torque/speed clamps and uses a wheel PI servo for slope holding; it
models the four pivot hinges as gravity-driven passive joints with zero
actuator torque. They are excluded from motor sizing; joint limits and model
bearing damping remain. `--no-passive-pivots` reproduces the old active-hold
proxy for comparison. The analysis does not replace the application's controller.
Inspect the report's tracking
and support checks before interpreting peaks as successful motion. Swerve
analysis defaults to 480 Hz physics and 240 Hz target updates: the original
240 Hz physics run produced a contact-sensitive FR holding oscillation.
Compare `--steering-gain-scale 10` for more accurate steering. At the default
physics timestep, `--control-steps 8` gives 60 Hz target updates; compare
`--timestep-divisor 4 --control-steps 4` for 960 Hz physics at the same 240 Hz
control rate. All measured torques refer to the output shafts.
Steering target changes are limited to 337.5 degrees/second in the analysis,
matching the maximum speed of a smooth 90-degree turn over 0.5 seconds. This
prevents abrupt projected-heading commands from unloaded wheels; it does not
limit motor torque. Use `--steering-rate-limit-deg-s 0` to reproduce the older
unlimited commands. For a flat-only study, add `--slopes 0`; duplicate flat
steering cases are run once.
Both load-analysis scripts default to passive pivots. Reports created before
this correction used an active pivot hold and require revalidation for the
passive mechanism.

Pivot bearing resistance now defaults to **0.05 Nm dry friction** and
**0.05 Nm s/rad viscous damping** at each of joint8/11/14/17. These are
provisional sensitivity parameters, not measured bearing data. The environment
applies them to old XML caches as well, and the model builder writes them into
new XML. Pass `pivot_frictionloss=None, pivot_damping=None` to the Python
environment constructor to preserve custom XML values. The GUI and both sizing
scripts accept `--pivot-frictionloss` / `--pivot-damping`; use
`--pivot-frictionloss 0 --pivot-damping 0.01` to compare the previous resistance.

Manual GUI runs now default to passive pivots, like sizing runs. There is no
pivot motor torque or software position lock in this mode. The option
`--no-passive-pivots` retains the historical upright servo for comparison.
Existing benchmark/training wrappers keep their historical environment mode;
for direct Python use, select `MujocoRobotEnv(..., passive_pivots=True)`.

Run `python scripts/analyze_pivot_friction.py` for the 10-case default sweep.
Resistance is applied before the initial fall in every independent run.
The script records every physics tick by default; decimating at a stride
aligned with the controller can bias torque measurements. `joint_friction_nm`
separates dry friction from motor torque and viscous `passive_nm`; the contact
constraint torque otherwise includes joint friction. Tire contact and whole
module contact are reported separately, with tire normal forces and gaps.
The October 13 study found that friction slowed the pivots and improved some
holding poses, but four driven tires still did not stay grounded throughout
folding. Holding support also changed between 480 and 960 Hz physics at the
same 240 Hz control rate. Calibrate pivot resistance and the auxiliary caster
contact geometry before final motor selection; increasing friction alone does
not guarantee four driven tires support the robot.

The independent **eight-motor closed-loop trial** joins `block4` to `block8`.
Run `python scripts/run_mujoco_closed_loop.py` on Linux for animated playback,
or `python scripts/run_mujoco_closed_loop.py --headless --cycles 1` without a
viewer. On macOS, use `mjpython` for the animated run. This builds
`data/my_robot/mujoco/robot_closed_loop.xml`, preserving the
seven-joint XML and original URDF. The new morphing hinge is `joint_closure`;
the original `joint8` remains a passive swerve pivot. An orange disk marks the
inferred eighth motor. No GNN checkpoint is needed.

Run `python scripts/analyze_closed_loop.py` for six-second forward/reverse
cycles, 26 holding cases, original-model comparisons and assumed added-mass
cases (0 / 0.15 / 0.30 kg). Results, raw NPZ files, CSV and figures go to the
ignored `experiments/closed_loop_YYYYMMDD_HHMMSS/` directory. Physics defaults to
1920 / 3840 Hz with 240 Hz target updates because this nearly singular CAD
closure is sensitive to coarse integration. `--skip-holds` and
`--skip-mass-sensitivity` provide a shorter smoke test.
Add `--animation` to export a six-second forward/six-second reverse GIF using
offscreen rendering, including in macOS sessions without an active display.

This is a provisional geometry study. The hinge point/axis are inferred from
the old formula; the seven old target angles are projected onto a feasible
closed-loop path within their original angle limits. The controller distributes
torque in the loop's allowed motion space to avoid commanded internal preload.
Existing manual collision exclusions and collision masks must remain intact:
the user deliberately disabled some pairs because of CAD approximation error.
Mesh intersections in those pairs are not evidence of physical interference.
The inferred hinge geometry still needs assembly calibration; historical sizing
recommendations are withdrawn because of uncertain geometry and command-update
spikes. Target endpoints now interpolate across physics ticks at the same
240 Hz control rate. The old seven-node GNN is not an eight-motor
controller; the trial has its own entry point and coordinated controller.

`python scripts/diagnose_closed_loop_contacts.py` compares original commands,
interpolated commands and an in-memory self-contact ablation; it never changes
the saved model or ground contact. The original triangle audit is optional:
`scripts/audit_closed_loop_meshes.py --deps-dir <python-fcl-install> --render`.
It bypasses exclusions for geometry inspection only and must not be used to
automatically enable excluded collision pairs. See `CLOSED_LOOP_TRIAL.md`.

For the corrected controller's preliminary eight-motor sizing table, use
`python scripts/report_closed_loop_sizing.py --simulate`. This repeats the
1920 / 3840 Hz full motion and 26 holding cases per frequency, verifies the
manual collision exclusions/masks and recomputes loads from raw NPZ samples.
The rated estimate uses max(motion RMS, worst holding RMS); both this estimate
and the measured simulation peak receive 50% margin before rounding upward.
Omit `--simulate` to verify and export existing data. Results are in
`experiments/closed_loop_sizing_20261020_interpolated/`. These are preliminary
values for the inferred hinge geometry; the withdrawn historical table remains
separate. `joint_closure` is J8 in the table, not the passive URDF `joint8`.
