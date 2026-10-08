# MuJoCo 运行与调试

项目现在只使用 MuJoCo。原始 URDF、STL 和训练权重作为输入资产保留，运行入口统一指向 MuJoCo。

## 在当前 Mac 上启动

在项目根目录执行：

```bash
conda activate robot-sim
mjpython scripts/run_mujoco_gui.py
```

程序打开 MuJoCo 窗口和本机浏览器控制面板。面板提供 Φ、世界坐标系 X/Y 速度、旋转速度、GNN/几何公式切换、PID 开关、停止与重置。

默认加载最新的 `experiments/phi_run_*/best_model.pt`。没有 GNN 权重时，可以独立使用几何公式：

```bash
mjpython scripts/run_mujoco_gui.py --source formula
```

只使用原生窗口、通过键盘控制：

```bash
mjpython scripts/run_mujoco_gui.py --no-panel
```

| 按键 | 功能 |
|---|---|
| W / S | 增减世界 X 方向速度 |
| A / D | 增减世界 Y 方向速度 |
| Q / E | 增减旋转速度 |
| Z / X | 增减 Φ |
| 空格 | 清零行驶速度 |
| R | 重置机器人及行驶指令 |
| G | 切换 GNN / 公式，须已加载 GNN |
| P | 开关 PID |

键盘指令修改并保持目标值，不需要一直按住。退出时关闭 MuJoCo 窗口或在终端按 Ctrl+C。

无窗口运行不需要 `mjpython`：

```bash
python scripts/run_mujoco_gui.py --headless --seconds 10 --vx 0.15
python scripts/run_mujoco_gui.py --headless --seconds 10 --phi 3
```

Windows 原生窗口也通过同一环境运行，将上述 `mjpython` 换成 `python`。Mac 的 `launch_passive` 窗口要求使用 `mjpython`；这是 MuJoCo 官方提供的启动器，不是额外环境。

## 调试

当前 Mac 的 `.vscode/launch.json` 已配置以下入口。由于 macOS 的 `launch_passive` 要求 `mjpython` 管理主线程，GUI 配置采用“VS Code 启动 mjpython 调试服务，再 attach”的方式。

- `robot-sim: MuJoCo GUI (Mac mjpython)`：自动启动同一环境的 `mjpython` 调试服务并 attach，可在新控制器和环境中设置断点。
- `robot-sim: MuJoCo headless`：普通 Python，适合单步调试物理状态和控制计算。
- `robot-sim: MuJoCo scripted scene B`：自动任务、MPC 和 GNN 的窗口调试，也通过 mjpython attach。

在 VS Code 的“运行和调试”面板中选择 **`robot-sim: MuJoCo GUI (Mac mjpython)`** 后点击绿色启动按钮。第一次启动会先在集成终端运行后台调试服务，随后 VS Code 自动连接；不要选择 `Current Python file` 或普通 Python 的 MuJoCo GUI 配置。

在 IDE 中选择 `robot-sim` Conda 环境的解释器。IDE 配置属于本机文件，不提交到仓库；在 Windows 或其他 Mac 上重新选择各自的解释器。

也可以通过终端等待调试器连接：

```bash
mjpython -m debugpy --listen 127.0.0.1:5678 --wait-for-client scripts/run_mujoco_gui.py
```

## 模型与资产

首次运行时，如果模型不存在，程序自动读取私有原始资产并生成：

```text
data/my_robot/mujoco/
  robot.xml             MuJoCo 浮动底座、碰撞、执行器模型
  meshes/               独立的轻量视觉网格、凸包碰撞网格
  conversion.json       转换统计
```

手动重新生成：

```bash
python scripts/build_mujoco_model.py
```

输入仍为 `data/my_robot/robot1231_2.urdf` 和它的 `meshes/*.STL`。这些私有资产及派生资产沿用仓库的忽略规则；部署到另一台电脑时需要带上原始资产。生成脚本和所有新 Python 源码可以正常提交。

模型包含 7 个变形关节、4 个转向关节、4 个驱动轮及 4 个 pivot 执行器，总计 19 个执行器；自由底座占 7 个位置变量，不能把 URDF 的关节编号直接当作 `qpos` 下标。

原视觉模型共 1,012,742 个三角面。默认 4 mm 顶点聚类将视觉网格降至约 132,948 个面；原始 STL 不变，碰撞网格单独从原网格生成凸包。需要更精细的显示时：

```bash
python scripts/build_mujoco_model.py --visual-voxel 0.0025
```

修改原 URDF 后应重新运行转换脚本，已有模型不会被静默重建。

## 新后端的物理与控制设置

- 物理步长：1/240 秒；手动 GUI 的控制和画面同步：60 Hz；自动任务的 MPC：20 Hz。
- 主循环按时间期限等待，扣除计算与绘制耗时。GNN 使用 CPU 单线程，缓存重复 Φ 的结果。
- 转向和滚动方向从实际关节轴计算，控制器接收世界坐标系速度；机器人位置取四个转向支撑点的几何中心。
- 折叠到 Φ=3 时底座前轴接近竖直，朝向使用机体横轴的水平投影，避免直接取底座欧拉 yaw 产生跳变。自动任务在变形尚未收敛时暂停行驶。
- 驱动轮使用半径 25.5 mm 的圆柱接触形状，速度上限 160 RPM、力矩上限 0.35 N·m。
- 变形执行器力矩上限 10 N·m，转向执行器上限 2 N·m。
- pivot 使用有限力矩控制保持转向柱竖直，力矩上限取 URDF 的 5 N·m。稳定形态捕获保持目标，变形时调整目标；不通过重置 `qpos` 强行锁定关节。
- 原 CAD 把辅助滚珠支撑并入 steering 刚体，没有独立滚动自由度。新模型对该支撑与地面的接触使用 0.02 的低切向阻力作为滚动近似；驱动轮仍使用正常轮地摩擦。
- 其他零件的碰撞采用凸包，因此细小凹槽不等同于原始 CAD 表面。模型适合当前控制调试，精确接触研究应进一步拆分滚珠与非凸碰撞几何。

MuJoCo 的执行器、摩擦及接触参数需要通过本后端的实验重新验证；历史结果不能直接替代新后端的结果。

## 自动任务与评估

```bash
# 自动大脑 → MPC → PID/Swerve → MuJoCo，默认 GNN 形态控制
mjpython scripts/evaluate_mujoco.py --scene B --gui --seconds 40

# 无窗口评估 A 窄门、B 航点、C 复合场景
python scripts/evaluate_mujoco.py --scene A --seconds 60
python scripts/evaluate_mujoco.py --scene B --seconds 60
python scripts/evaluate_mujoco.py --scene C --seconds 100
```

结果默认保存到新的 `experiments/mujoco_evaluations/`，记录实际轨迹、关节误差、支撑接触、障碍接触力和穿越状态。`goal_reached` 表示到达目标；`success` 还要求无障碍碰撞、保持支撑及按通道穿越。超时或未通过不会被标记为成功。

B 场景保留原随机航点，在新后端使用 PCHIP 曲线连接，避免原三次样条越过场地边界；任务生成逻辑保持独立。

## PPO 训练与推理

新 `MujocoBrainGymEnv` 保持 21 维观测、3 维动作和每个动作 1 秒的接口，内部物理 240 Hz、控制 60 Hz、MPC 20 Hz。观测包含自身状态及最近两个障碍物；动作是局部目标位移 X/Y 和 Φ。

```bash
python scripts/train_mujoco_brain.py --scene B --timesteps 20000
```

训练结果写入独立的 `experiments/mujoco_ppo_<时间>/`，包含 `ppo_mujoco.zip`、Monitor 日志、TensorBoard 和参数记录。先在 B 场景调通，再使用 `--scene A` 或 `--scene C` 训练障碍物任务。

```bash
python scripts/evaluate_mujoco.py --mode ppo --policy experiments/mujoco_ppo_<时间>/ppo_mujoco.zip --scene B
```

已有 PPO 权重若维度一致可以加载用于迁移实验，但新环境的物理、奖励与终止条件有差异，应重新训练或微调。已有 MorphGNN 权重可以直接加载，节点与关节映射保持一致。

## 文件对应关系

| 新文件 | 用途 |
|---|---|
| `src/envs/mujoco_model.py` | 从原资产构建独立 MuJoCo 模型 |
| `src/envs/mujoco_robot_env.py` | 状态、执行器、物理、接触及窗口接口 |
| `src/controllers/mujoco_control.py` | 复用现有 GNN、PID、Swerve，组合为新后端控制链 |
| `src/envs/mujoco_benchmark_env.py` | A/B/C 场景与感知 |
| `src/envs/mujoco_brain_gym_env.py` | Gymnasium / PPO 接口 |
| `src/utils/mujoco_control_panel.py` | 本机浏览器滑块及键盘控制 |
| `scripts/build_mujoco_model.py` | 模型重新生成入口 |
| `scripts/run_mujoco_gui.py` | 手动 GUI / 无窗口运行入口 |
| `scripts/evaluate_mujoco.py` | 规则大脑或 PPO 评估 |
| `scripts/train_mujoco_brain.py` | 独立 PPO 训练 |
| `tests/test_mujoco_backend.py` | 接地、行驶、变形、场景和接口测试 |

执行全部回归测试：

```bash
python -m pytest tests -q
```

公开仓库未带私有 URDF/STL 时，依赖私有模型的物理测试会跳过；当前本机可执行完整测试。

2026-10-08 本机验证：26 个测试及 1 个子测试通过；六个采样形态完成接地、变形与行驶检查；原生窗口通过 `mjpython` 与 `debugpy` 启动，3 秒仿真耗时约 3.004 秒。原始 URDF/STL 输入资产和历史实验输出保留在各自目录中。

现有 GNN + 规则大脑在随机种子 42 下的单次端到端检查：

| 场景 | 仿真时间 | 结果 |
|---|---:|---|
| A 窄门 | 14.22 秒 | 到达目标，无障碍碰撞，按通道穿越 |
| B 航点 | 31.22 秒 | 到达目标，无障碍碰撞 |
| C 复合场景 | 56.93 秒 | 穿越全部四个障碍并到达目标，无障碍碰撞 |

这是运行验证，不是多次随机试验的成功率统计。检查记录在 `data/my_robot/mujoco/verification.json`，完整轨迹记录在 `experiments/mujoco_evaluations/setup_*.json`。

PPO 已完成 8 步短训练、保存及加载推理，用于验证接口；短训练模型不代表已学会导航，保存在本机临时测试目录中，不覆盖原有实验模型。

参考：[MuJoCo Python 与 Mac 窗口启动](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer)、[URDF 与 MJCF 模型](https://mujoco.readthedocs.io/en/stable/modeling.html#urdf-extensions)。
