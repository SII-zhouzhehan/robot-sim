# 闭链可变形机器人：奇异感知三层图 Pipeline 与近期论文

更新日期：2026-11-14

> **当前 Exp A 口径（2026-11-25）**：本文件保留较早的约束扫描与文献分析，不能直接作为新探索器的数据接口或动作先验。新 Exp A 从实测关节零位开始，学习器只获得 XML 硬件连接和事先声明的传感器反馈。姿态进度、参考姿态库、内部约束矩阵和解析雅可比不得输入探索器。当前图名统一为 `G_struct`（XML 硬件连接）、`G_local(s)`（反馈学到的阶段内运动关系）和 `G_transition`（反馈学到的阶段转换）。旧名 `G_mode` 中的分组信息并入相应阶段的 `G_local(s)`；旧 `G_state` 对应 `G_local(s)`，旧 `G_phase` 对应 `G_transition`。详见 [Exp A 三图自主发现计划](experiments/EXP_A_THREE_GRAPH_DISCOVERY_PLAN.md)。

## 0. 总体判断

这版 Pipeline 的总体方向是对的：

- 用 MuJoCo 扫描闭环约束流形；
- 用三张图分别表示有效拓扑、阶段内耦合和阶段转换；
- 大脑用 FSM/PPO 负责导航与策略探索，小脑协调形变期间的舵轮位移，脊髓负责三层图算法探索；
- 把切换时机、速度条件和失败恢复纳入图。

可以作为下一阶段主线。但有几处技术假设必须修正，否则扫描结果和三图一致性会产生看似正确、实际不可解释的结论。

最重要的五点：

1. `efc_J` 混合 equality、limit、contact 和摩擦约束，不能把全部 `nefc` 行直接当成闭环 Jacobian。
2. nullspace 基不是唯一的，不能直接对任意基向量的“零分量”做聚类。
3. 秩不变但耦合关系改变，也是阶段切换，不能只靠 rank jump 检测。
4. 移动层和变形层不能完全解耦，变形会改变轮距、支撑多边形、转向中心和动力学。
5. spatial tendon 适合距离或路径长度约束，不能无条件替代刚体连杆的完整 3D 闭合约束。

## 1. 分阶段审查

### 阶段 0：建模

原方案：

```text
8 电机闭环转 MJCF；闭链用 spatial tendon 替代部分 connect；
碰撞体简化 + exclude；solref 不过硬。
明确 4 阶段是显式 equality 还是几何自然分支。
```

评价：方向正确，但需要拆成“扫描模型”和“控制模型”两套精度目标。

#### 建议 0.1：先确定闭链的真实物理构成

当前旧 8-block 替身是：

```text
block4 -> joint_closure -> closure_rotor --weld--> block8
```

完整机构更可能是：

```text
block4 -> 关节 A -> closure_link -> 关节 B -> block8
```

要先确认：

- closure link 是一根刚杆还是柔性杆；
- 两端是转动、球副、虎克副还是接触约束；
- 哪一端主动，哪一端被动；
- 真实长度、质量、惯量和轴线；
- 连接处是否有间隙、摩擦、限位或卡扣。

在这个问题明确前，当前模型的 rank/nullity 只能解释为替身模型性质。

#### 建议 0.2：spatial tendon 的适用范围

spatial tendon 适合：

- 两端距离约束；
- 绳索路径长度约束；
- 柔性传动；
- 单向张紧约束。

如果真实 closure link 要求完整的相对位姿约束，spatial tendon 不足以表达 3D 姿态闭合。需要：

- 刚体 link 加两个真实关节；
- MuJoCo `connect`/`weld`；
- 多行 equality；
- 或支持闭链的 SDF/其他建模格式。

#### 建议 0.3：扫描模型和控制模型分开

扫描几何秩与奇异流形时：

- 使用较硬的 equality，保证约束残差足够小；
- 记录约束残差，避免软约束造成假秩变化；
- 使用统一单位和行缩放。

控制仿真时：

- 可以放宽 `solref` 提高数值稳定性；
- 但要单独量化软约束漂移；
- 不能让控制模型的软约束漂移反过来污染几何扫描标签。

#### 建议 0.4：四阶段必须标注生成机制

每个转换点必须标记为以下之一：

- `explicit_equality_switch`：显式启停 equality；
- `active_contact_switch`：接触/支撑改变；
- `limit_switch`：限位或单向约束激活；
- `geometric_branch`：几何自然分支或奇异穿越；
- `actuator_mode_switch`：执行器模式或传动关系改变。

不能把 hidden equality 产生的变化宣传成几何奇异。

### 阶段 1：三图定义

原方案把三图定义为：

- 图 A：等效刚体块与剩余自由关节；
- 图 B：阶段内关节耦合；
- 图 C：阶段转换。

评价：三图职责清楚，但术语和字段需要收紧。

#### 图 A：建议称为“有效拓扑模式图”

节点不是简单按 block 编号分组，而是当前阶段内相对位姿保持不变的等效刚体集合。建议保存：

```text
mode_id
rigid_group_partition
group_pose_relation
remaining_free_joint_set
active_constraint_set
valid_configuration_region
evidence_confidence
```

“等效刚体”不能用运动同步单独证明。至少要检查完整相对位姿是否保持不变。

#### 图 B：建议称为“阶段内约束图”

节点是可动或可观测关节；因子节点表示多关节约束。建议保存：

```text
joint_role: active/passive/weak/limited/uncertain
constraint_factor
constraint_residual
nullspace_dimension
nullspace_projection
participation_score
evidence_window
```

不要只保存 `locked_joint_list`。真实系统中很多关节不是永久锁死，而是在当前负载和方向下弱响应。

#### 图 C：建议称为“阶段转换图”

节点是已验证模式；边必须带类型：

```text
edge_type
source_mode
target_mode
active_set_change
rank_change
jacobian_change
switch_configuration
approach_direction
velocity_and_momentum_condition
contact_and_support_condition
success_failure_statistics
hysteresis
quarantine_status
```

至少区分：

- `singular_crossing`：Jacobian 秩下降或运动分支交换；
- `constraint_switch`：约束集合或有效 DOF 改变；
- `jacobian_switch`：秩不变，但控制映射或耦合方向改变；
- `assembly_switch`：物理拓扑变化。

### 阶段 2：MuJoCo 扫描

原方案：

```text
qpos -> mj_forward -> mj_jac 与 efc_J -> rank/nullspace/efc_type/sigma_min
```

评价：方向正确，但这是全 Pipeline 中最容易出错的一步。

#### 必须修正 2.1：过滤约束类型

`efc_J[:nefc]` 包含当前所有 active constraints。扫描时必须按 `efc_type` 分类：

- equality；
- joint limit；
- contact normal；
- friction cone；
- tendon limit；
- connect/weld。

不要直接对混合矩阵求一个“总 rank”并称其为闭环秩。

#### 必须修正 2.2：统一单位和行缩放

平动约束、转动约束、关节约束和 tendon 约束单位不同。建议：

1. 对每一行按物理单位或约束尺度归一化；
2. 先做行平衡，再计算 SVD；
3. 使用相对秩阈值和奇异值间隙；
4. 报告原始值和缩放后的值。

#### 必须修正 2.3：nullspace 基不是唯一表示

`null(J)` 的任意正交基都可以乘一个旋转矩阵。因此不能直接把基中“恰好为零”的分量聚类成 block。

建议使用：

- nullspace projection matrix：`P = I - J^+J`；
- principal angles；
- sparse nullspace basis；
- QR with column pivoting；
- 对候选关节子空间做正交性/包含性检验；
- subspace clustering。

#### 必须修正 2.4：rank jump 不是唯一阶段信号

同样需要扫描：

- `sigma_min` 和谱间隙；
- active constraint set；
- nullspace projection；
- 主被动映射；
- 输入输出传递方向；
- 速度/动量可行性；
- 接触和支撑状态。

#### 建议 2.5：粗扫和局部加密可以保留

粗扫建议覆盖：

1. 已知 Phi 参考路径；
2. 多起始点；
3. 接近限位和接触的边界；
4. 不同的零空间方向；
5. 多个速度方向，不只 qpos。

局部加密应使用 continuation 或 branch tracking，而不是只在失败点附近随机加点。

### 阶段 3：三图构建

原方案：

```text
null(A) 恒零分量聚类 -> 图 A
块内零空间结构 -> 图 B
块划分变化与秩跳变 -> 图 C
```

评价：思路可以，但需要替换其中的基依赖操作。

#### 图 A 构建

不要用单次 nullspace 基的零分量。建议：

1. 在局部邻域内拟合切空间；
2. 计算坐标子空间之间的 principal angles；
3. 检查完整相对位姿是否保持；
4. 用跨多个采样点的稳定性做 block partition；
5. 对 partition 做 bootstrap 或局部扰动验证。

#### 图 B 构建

建议从以下证据分别构建：

- 独立干预响应矩阵；
- 约束 Jacobian 行空间；
- 零空间投影；
- 因子残差；
- 主被动映射；
- 留出动作验证。

图 B 中的边应带符号、比例、适用范围和置信度，而不是只写“有耦合”。

#### 图 C 构建

转换由三类信号共同决定：

1. active set 改变；
2. 约束或任务 Jacobian 的秩/子空间改变；
3. 图 A 的 block partition 或图 B 的耦合关系改变。

只有三者一致时，才创建高置信度阶段边。仅 rank 变化但 partition 不变，可能是 same-rank regime change；仅 partition 变化但 rank 不变，可能是约束或执行映射切换。

### 阶段 4：前向仿真验证

原方案用轨迹跟踪验证图 C 采样路径，正确。建议增加：

1. 每个转换要求执行正反两个方向；
2. 从不同接近速度和起始姿态尝试；
3. 验证 exit branch；
4. 检查闭合残差而非只看关节到达；
5. 记录失败模式；
6. 验证切换期间接触和支撑；
7. 对失败路径回写图 C 的 hysteresis 和 quarantine 字段。

下层约束控制可以参考 2026 年的 Projection-Retraction MPPI：采样速度投影到等式切空间，再 retraction 回约束流形，能保证闭环等式残差，而不是只靠代价惩罚。

### 阶段 5：三层控制

当前分工：

- 大脑：FSM 导航与 PPO 探索任务/模式决策；
- 小脑：一边变形一边协调舵轮模块与整体位移；
- 脊髓：自主探索、优化并应用三层图算法，表达拓扑模式、阶段内约束和阶段转换。Phi 仅作参考坐标。

评价：小脑需要在形变过程中协调舵轮位移。移动和变形不能简单解耦，这是重要研究难点。

#### 修正 5.1：小脑联合协调形变与舵轮位移

变形会改变：

- 轮距和转向中心；
- 支撑多边形；
- 质心；
- 轮地正压力和摩擦；
- 转向轴方向；
- 可通行宽度。

因此在协调运动与形变时，小脑至少需要接收：

- 当前模式；
- 支撑状态；
- 几何中心；
- 轮位和转向中心；
- 预计变形时间；
- 是否处于 switchable window。

#### 修正 5.2：高层 RL 不要直接输出全部关节动作

建议高层动作：

```text
navigation_goal
desired_mode
request_switch
approach_direction
speed_or_risk_budget
```

高层 RL 的输出由 `G_phase` 动作掩码过滤。低层由约束 MPC、QP 或图条件策略执行。

#### 修正 5.3：切换窗口由多层共同确认

切换窗口至少需要：

- 图 C 的几何/速度 guard；
- 中层速度低于阈值；
- 支撑稳定；
- 闭环残差可接受；
- 执行器权限足够；
- 无碰撞或接触冲突。

#### 修正 5.4：用 options 表示模式转换

一个 `G_phase` 边可以视为一个 option：

```text
option = (source_mode, target_mode, initiation_set,
          transition_policy, termination_set, failure_set)
```

高层选择 option，中层负责移动协调，下层负责执行。

### 阶段 6：连续变形导航

方向有潜力，但应分阶段验证：

1. 静态目标模式导航；
2. 停车后切换模式；
3. 低速行进中切换；
4. 不停车连续切换；
5. 真实地形与接触扰动。

不要一开始就把“连续变形导航”作为已完成能力。每一级都要有单独的成功率和安全指标。

## 2. 修订后的 Pipeline

```text
阶段 0：双模型建模
  0A 几何扫描模型：真实闭链 link/joint、硬约束、低漂移
  0B 控制仿真模型：合理柔顺、接触、摩擦和传感器延迟

阶段 1：定义三图 schema
  G_mode：有效拓扑模式与等效刚体分组
  G_state：阶段内约束、零空间、因子和主被动映射
  G_phase：带类型、方向、速度、支撑和失败证据的转换图

阶段 2：多起点约束流形扫描
  qpos/qvel/control
  -> active constraints
  -> efc_type 过滤
  -> 行缩放
  -> rank/nullspace projection/sigma/principal angles
  -> 粗扫与局部 branch continuation

阶段 3：三图投影
  subspace clustering -> G_mode
  intervention + factor evidence -> G_state
  active set + subspace + partition change -> G_phase
  三图交叉一致性检验

阶段 4：图路径前向验证
  正反向、多速度、多起点、闭合残差、exit branch、失败回写

阶段 5：分层控制
  高层 constrained options RL + G_phase 动作掩码
  中层 morphology-aware MPC/PID + switchable window
  下层约束 MPPI/QP/MPC + G_state 执行
  中层与下层通过模式、支撑和切换窗口耦合

阶段 6：连续变形导航
  停车切换 -> 低速切换 -> 连续切换 -> 实机地形验证
```

## 3. 建议的核心算法贡献

可以把论文贡献收束为：

1. 从闭环约束流形扫描中构建状态相关三图，而不是手工写阶段机。
2. 用 active constraint set、nullspace subspace、Jacobian subspace 和有效拓扑分区共同定义 typed transitions。
3. 用图 C 作为 constrained options 的行为接口，由高层 RL 选择模式和切换意图，低层约束控制保证闭合和安全。
4. 证明三图相比无图 RL、人工 FSM、只看 rank 的方法在切换成功率、样本效率、安全性和泛化上更好。

## 4. 近期相关论文链接

以下按用途分组。2026 年 arXiv 工作可能仍在审稿，正式引用前需核对最终版本。

### 4.1 奇异点、运动分支和模式切换

1. **Output Mode Switching for Parallel Five-bar Manipulators Using a Graph-based Path Planner**
   ICRA 2023
   https://doi.org/10.1109/ICRA48891.2023.10160891
   用途：已知机构构型空间图、穿过奇异边界、实现非相邻输出模式切换。

2. **Output Mode Switching for Parallel Five-bar Manipulators Using a Projection-based Direct Collocation Method**
   ReMAR 2024
   https://doi.org/10.1109/ReMAR61031.2024.10617557
   用途：动态奇异切换、动量规划、执行器控制权受限时的最优轨迹。

3. **Jacobian linear subspace-based determination of kinematic paths for multi-degree-of-freedom multi-loop mechanisms**
   Mechanism and Machine Theory 2026
   https://doi.org/10.1016/j.mechmachtheory.2026.106632
   用途：多闭环机构、Jacobian 子空间、零空间、奇异构型和模式切换路径。

4. **Kinematic and Dexterity Analysis for Stable Jacobian-Switching Control of a Real-Time Reconfigurable Robotic Mechanism**
   Journal of Mechanisms and Robotics 2026
   https://doi.org/10.1115/1.4071263
   用途：非奇异 Jacobian 切换、passivity、稳定逆运动学。注意该工作明确避免 bifurcation singularity。

5. **Dynamic Reconstruction of Degrees of Freedom and Coupling Control in 3RPUR Metamorphic Parallel Mechanism**
   Machines 2025
   https://doi.org/10.3390/machines13100894
   用途：锁止/解锁导致有效自由度重构，3T 与 3T1R 模式切换。

6. **Robust Optimization and Workspace Enhancement of a Reconfigurable Delta Robot via a Singularity-Sensitive Index**
   Robotics 2025
   https://doi.org/10.3390/robotics15010011
   用途：可重构并联机器人的奇异敏感指标、工作空间和力/速度权衡。

7. **Reconfigurable Deformable Mechanisms: Review and Application Examples**
   Mechanisms and Machine Science 2026
   https://doi.org/10.1007/978-981-95-7904-4_88
   用途：可重构变形机构综述入口。

### 4.2 闭环约束建模与底层控制

8. **Modular Kinematic Reduction of Closed-Chain Mechanisms Using Path Assembly and Defect Homotopy**
   arXiv 2026
   https://arxiv.org/abs/2609.11338
   用途：闭环约束降维、被动坐标恢复、预测-校正 continuation。

9. **Projection-Retraction MPPI: Exact Constraint-Manifold Control for Manipulators**
   arXiv 2026
   https://arxiv.org/abs/2608.07573
   用途：闭环等式和不等式约束下的采样式 MPC，适合作为下层约束执行器。

10. **CRISP: Constraint-Resolved Interactions between Subsystems for Predictive Whole-Body Cooperation**
    arXiv 2026
    https://arxiv.org/abs/2611.07268
    用途：分层 NMPC、闭环子系统、交互扳手和动态耦合。

11. **Inertia Partitioning Modular Robust Control Framework for Reconfigurable Multibody Systems**
    arXiv 2025，更新 2026
    https://arxiv.org/abs/2510.23226
    用途：可重构多体系统的局部可更新惯性模型、闭链最小坐标控制。

### 4.3 闭链机器人的强化学习

12. **Robust RL Control for Bipedal Locomotion with Closed Kinematic Chains**
    arXiv 2025
    https://arxiv.org/abs/2507.10164
    用途：直接在闭链动力学上训练，避免串联近似带来的 sim-to-real 误差。

13. **Mechanical Intelligence-Aware Curriculum Reinforcement Learning for Humanoids with Parallel Actuation**
    IEEE Humanoids 2025
    https://arxiv.org/abs/2507.00273
    用途：MJX 原生并联机构仿真、课程 RL 和零样本实机部署。

14. **LiPS: Large-Scale Humanoid Robot Reinforcement Learning with Parallel-Series Structures**
    arXiv 2025
    https://arxiv.org/abs/2503.08349
    用途：大规模 RL 中处理串并联和闭链拓扑。

15. **Learning to Hop for a Single-Legged Robot with Parallel Mechanism**
    arXiv 2025
    https://arxiv.org/abs/2501.11945
    用途：并联到串联转换、力矩映射和 sim-to-real。

### 4.4 阶段门控、选项和跨构型泛化

16. **Phase-Gated Constrained Reinforcement Learning for End-to-End Capture and Detumbling of Non-Cooperative Targets**
    arXiv 2026，审稿中
    https://arxiv.org/abs/2610.15491
    用途：proximity gate 同时作用于 reward、action space 和 constraint set，可改造成 singularity gate。

17. **Reinforcement learning in linear embedding space unlocks generalizable control across soft robot configurations**
    arXiv 2026，Nature Communications 接收信息见论文说明
    https://arxiv.org/abs/2606.08104
    用途：共享 Koopman 线性嵌入空间，实现跨构型快速适配。

18. **Morphology-Aware Graph Reinforcement Learning for Tensegrity Robot Locomotion**
    arXiv 2025，更新 2026
    https://arxiv.org/abs/2510.26067
    用途：把物理拓扑图输入 GNN，再与 SAC 组成形态感知策略。

19. **Enabling Option Learning in Sparse Rewards with Hindsight Experience Replay**
    arXiv 2026
    https://arxiv.org/abs/2602.13865
    用途：稀疏多目标下学习 options，适合离散模式转换子目标。

20. **Certificated Actor-Critic: Hierarchical Reinforcement Learning with Control Barrier Functions for Safe Navigation**
    ICRA 2025
    https://arxiv.org/abs/2501.17424
    用途：分层 RL 与 CBF 安全约束。

21. **A Novel Multi-Timescale Stability-Preserving Hierarchical Reinforcement Learning Controller Framework**
    arXiv 2025
    https://arxiv.org/abs/2510.22420
    用途：半马尔可夫高层规划、低层控制、Lyapunov 稳定性约束。

### 4.5 动力学补偿和实机迁移

22. **Physics-Aware Sparse Learning and Selective Online Adaptation for Euler-Lagrange Robot Dynamics**
    arXiv 2026
    https://arxiv.org/abs/2606.09640
    用途：结构保持的残差学习，分别补偿惯性、Coriolis 和广义力。

23. **System Identification For Constrained Robots**
    arXiv 2024
    https://arxiv.org/abs/2408.08830
    用途：闭环系统辨识、摩擦和约束参数。

### 4.6 可重构机器人与形态控制

24. **DittoGym: Learning to Control Soft Shape-Shifting Robots**
    arXiv 2024，更新 2025
    https://arxiv.org/abs/2401.13231
    用途：把形态变化、运动和交互放入同一动作空间的可重构机器人 benchmark。

25. **Co-optimizing Physical Reconfiguration Parameters and Controllers for an Origami-inspired Reconfigurable Manipulator**
    arXiv 2025
    https://arxiv.org/abs/2504.10474
    用途：重构参数与控制策略联合优化。

26. **Multi-Modal Decentralized Reinforcement Learning for Modular Reconfigurable Lunar Robots**
    arXiv 2025
    https://arxiv.org/abs/2510.20347
    用途：模块级策略、零样本配置泛化和去中心化协调。

27. **GeoTrussRover: Morphological Computation with Contact-Semantic Control Primitives**
    arXiv 2026
    https://arxiv.org/abs/2609.11361
    用途：接触语义 primitives、可变几何桁架的阶段复用与局部重算。

## 5. 建议优先精读顺序

第一批，理解问题边界：

1. ICRA 2023 五杆模式切换；
2. ReMAR 2024 动态奇异穿越；
3. 2026 JMR Jacobian switching；
4. 2026 MMT Jacobian subspace paths。

第二批，确定下层执行：

1. PACDM 2026；
2. PR-MPPI 2026；
3. Inertia Partitioning 2025。

第三批，设计 RL：

1. Closed-chain RL 2025；
2. Phase-Gated Constrained RL 2026；
3. Koopman embedding RL 2026；
4. Morphology-aware Graph RL 2025/2026。

第四批，安全与 options：

1. CBF actor-critic 2025；
2. Lyapunov HRL 2025；
3. Options + HER 2026；
4. Structure-preserving residual 2026。

## 6. 旧信息入口

四份旧三图研究文档已经合并为 [LEGACY_THREE_GRAPH_NOTES.md](LEGACY_THREE_GRAPH_NOTES.md)。该文件只保留仍约束当前 Pipeline 的图定义、基础文献、实验结果、负结果和模型边界。

本文件是当前研究总入口；旧实验报告和实验数据保持原位置，不随文档合并移动或删除。

## 7. 最终建议

这版 Pipeline 可以作为主线，但应改名为“奇异感知的约束流形三图 + 分层 options 控制”。最重要的工程步骤不是先接 RL，而是先完成：

1. 真实闭链 link 和两端关节；
2. 扫描模型与控制模型分离；
3. 约束行过滤、行缩放和 subspace 检测；
4. rank、active set 和耦合变化三者联合定义转换；
5. 用 PR-MPPI/QP 等约束执行器验证图路径。

等这五步稳定后，再把高层导航策略替换成 constrained options RL，论文的证据链会明显更强。
