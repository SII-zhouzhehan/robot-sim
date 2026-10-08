# Script entry points

Run these scripts from the repository root in the `robot-sim` Conda environment.
For native MuJoCo windows on macOS, use `mjpython` in place of `python`.

| Script | Purpose |
| --- | --- |
| `setup_environment.py` | Install shared Python 3.11 dependencies. |
| `build_mujoco_model.py` | Convert the private URDF/STL assets into MuJoCo XML and meshes. |
| `run_mujoco_gui.py` | Manual control with the native viewer and browser control panel; supports headless runs. |
| `evaluate_mujoco.py` | Run scripted MPC or a PPO policy in scene A, B, or C and save a JSON report. |
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
repository's ignore rules. GNN workflows require private data or checkpoints;
use `--source formula` for MuJoCo control without trained weights.

The curriculum trainer saves `vecnorm_*.pkl` alongside its policies. Those
policies require matching observation normalization during inference. The
generic evaluator directly accepts policies from `train_mujoco_brain.py`.
