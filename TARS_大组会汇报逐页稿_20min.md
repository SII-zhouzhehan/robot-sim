# TARS 大组会汇报逐页稿

**题目：闭链可变形机器人的三层闭环控制架构**  
**英文题目：Closed-Loop Control Architecture for a Closed-Chain Morphing Robot**  
*Brain–Cerebellum–Spinal Control with Three-Graph Spinal Representation*

**建议 14 页，正文约 17 分 15 秒，问答约 2 分 45 秒（总计严格控制在 20 分钟内）。** 套用《PPT 模板.pptx》的 16:9 版式、标题、页脚和主题色。投屏只放英文，中文对照用于校稿，中文讲稿用于演讲者备注。

## 汇报故事与时间分配（合计正文 17:15 + 问答 2:45 = 20:00）

| 顺序 | 本汇报要回答的事 | 页码 | 建议时长 |
|---|---|---|---:|
| 研究目标与价值 | 闭链可变形机器人应用广阔但统一控制算法稀缺，TARS 要解决什么目标？ | 1–2 | 1:40 |
| 关键问题 | 实现该目标最难落地之处在哪里？与既有假设和研究的边界是什么？ | 3 | 1:20 |
| 已有工作 | 三篇前沿工作分别解决了什么，和 TARS 的具体差异在哪里？ | 4 | 1:50 |
| 我的思路 | 三层控制如何闭环？脊髓三图如何建立、更新并设计公平验证对照？ | 5–9 | 6:15 |
| 模型选型对比与实测进展 | 完整 CAD 模型相对旧估算差距多大？近期基础移动与工作站三图实验测出了什么？ | 10–12 | 4:35 |
| 下一步计划与讨论 | 如何收尾闭环验证并迁移到真机？希望讨论哪些关键设计问题？ | 13–14 | 1:35 |

**核心主张**　TARS 的目标是验证一个面向闭链可变形机器人的完整控制闭环：大脑提出任务与目标阶段，脊髓基于三张图发现并执行可行变形，小脑把变形和舵轮运动协调起来，实际反馈再用于判断动作是否可继续、是否到达或需要重规划。三图是控制信息的组织方式，研究价值要由可复现的闭环任务和对照实验来证明。

**进度口径**　完整 CAD 闭链 MuJoCo 模型（`tars_mjcf_grounded`）与 8 万向球弹簧悬挂已完成导出和基础选型/移动测试（第 10–11 页展示已保存的真实仿真数据）；工作站三图核心算法对照实验（实验 A–D）安排在第 12 页结果展位，汇报前按实际跑完进度填入实测图表，未跑完的项明确标为待完成，不夸大结论。

## 可编辑 PPT 插图通用视觉规范（适用于以下各页插图生成提示词）

> 若用代码/Agent（如 `pptxgenjs` / `python-pptx` / SVG 转 PPTX）生成下方各页配图，统一遵守以下约束以保持整套汇报视觉一致：
> 1. **输出格式**：原生可编辑 16:9 宽屏 `.pptx`（`13.333 × 7.5 in`，或页面内指定的安全绘图区域）及对应 `.svg` 预览；所有图形、卡片、箭头、标注文字必须是**原生可编辑矢量对象与文本框**，禁止输出扁平位图或栅格化文字。
> 2. **文字与语言**：图内全部文字仅用**英文**，无衬线字体（`Arial` / `Helvetica`），正文字号 `≥ 11 pt`，标题 `14–18 pt`，留足内边距避免重叠与截断。
> 3. **统一学术配色**：纯白画布 `#FFFFFF`；浅灰卡片底 `#F8FAFC`、边框 `#CBD5E1`；深石板主文字 `#1E293B`、辅助灰 `#64748B`；控制/指令蓝 `#1D4ED8`（浅底 `#EFF6FF`）；局部状态/耦合紫 `#7C3AED`（浅底 `#F5F3FF`）；阶段转换/门控琥珀 `#D97706`（浅底 `#FFFBEB`）；反馈/验证绿 `#059669`（浅底 `#ECFDF5`）。
> 4. **语义线型**：已验证关系/正常指令流用**实线**；候选/未验证关系、计划中模块（如 PPO）或可选辅助反馈用**虚线**；严禁绘制背景网格线，严禁在概念图中编造未经实验核实的数据曲线或性能数字。

---

# 一、研究目标与价值

## 1. 封面 · 0:25

### English slide text

**Closed-Loop Control Architecture for a Closed-Chain Morphing Robot**

*Brain–Cerebellum–Spinal Control with Three-Graph Spinal Representation*

TARS · 8 body joints · 4 swerve modules

### 中文对照

闭链可变形机器人的三层闭环控制架构。副标题为大脑—小脑—脊髓控制与三图脊髓表征。TARS 有八个机身关节和四组舵轮。

### 插图与版式

按模板封面放当前最新闭链 MuJoCo 模型（`tars_mjcf_grounded`）的斜俯视高分辨率实拍渲染图（白底或浅灰底）。在 PPT 中用少量原生英文引线标注 *8 Body joints*、*Closed-chain loop*、*4 Swerve modules*、*Spring-loaded casters*。标题使用陈述句，不放问题句和结果性口号。本页以仿真模型实拍渲染为主，不需要额外生成矢量插图。

### 中文讲稿

今天汇报 TARS 的总体控制思路。目标不是分别演示几个算法模块，而是把任务决策、闭链变形、舵轮移动和执行反馈接成一个完整闭环，并重点探索脊髓里的三张图怎样帮助机器人理解当前可做的变形动作。

## 2. 研究目标与价值 · 1:15

### English slide text

**Broad application potential, yet unified closed-loop control remains scarce**

- **Broad applications:** Closed-chain morphing robots combine structural stiffness and load capacity with shape adaptability for confined spaces, narrow passages, inspection, and multi-terrain transit.
- **Algorithmic gap:** Prior work largely focuses on mechanical design, fixed-topology closed-chain control, or open-chain/preset-mode morphing; unified online algorithms for stage-varying closed-chain morphing during locomotion remain underexplored.
- **Platform (TARS):** 8-block closed-chain body (8 active joints) coupled with 4 active swerve modules and passive suspension.

**Research goal**  
Build and validate a **three-layer closed-loop control architecture** (Brain–Cerebellum–Spinal) that identifies stage-dependent feasible morphing actions from feedback, coordinates changing body geometry with swerve locomotion, and closes the task-to-execution loop.

### 中文对照

闭链可变形机器人兼具高结构刚度/承载能力与空间形态适应性，在受限空间、窄道穿行、巡检与复杂地形通行等场景应用广阔。但现有研究多集中于机构硬件设计、固定拓扑闭链控制或开链/预设模式变形；针对“变形中约束随阶段变化且与移动底盘强耦合”的统一在线闭环控制算法仍然稀缺。围绕 TARS 平台（八块闭链机身 + 八主动形变关节 + 四主动舵轮与被动悬挂），本研究目标是构建并验证大脑—小脑—脊髓三层闭环控制架构：从反馈中识别阶段相关的可行形变与转换条件，协调变化中的机身几何与舵轮移动，打通任务决策到执行的完整信息闭环。

### 插图与版式

采用左右分栏布局：
- **左侧（平台与场景）**：放 TARS 闭链模型俯斜视图占位框（插入 MuJoCo 实拍图并标注 *8 blocks / 8 body joints / Closed loop / 4 swerve modules*），下方配三个紧凑应用场景图标（*Narrow passages · Confined inspection · Multi-terrain transit*）。
- **右侧（价值缺口与研究目标）**：上下两张卡片——上方用双栏对比突出 *High hardware & application potential* vs. *Scarce unified closed-loop algorithms*；下方画核心研究闭环箭头 *Task & Stage Goal → Stage-Aware Morphing → Swerve Locomotion → Execution Feedback → Task Update*。本页不放“沿途包络”细部或未经核实的性能数值。

#### 可编辑 PPT 插图生成提示词（第 2 页）

```text
请遵循【可编辑 PPT 插图通用视觉规范】，生成第 2 页（Research Goal & Motivation）的原生可编辑 PPTX 与 SVG 矢量插图，布局尺寸为 12.1 × 5.2 in（适配 16:9 模板正文区），图内全部文字使用英文，无任何虚构数值。

【布局结构：左右双栏（左 4.6 in，右 7.2 in，间距 0.3 in）】
1. 左侧栏（Platform & Application Context）：
   - 上部（高度 3.6 in）：浅灰圆角虚线占位框（label: "Insert TARS MuJoCo Render Here"），并在框内四周预设 4 组可编辑引线标签，方便贴图后微调指向：
     * "8 closed-chain blocks & joints"
     * "Loop-closure constraint"
     * "4 active swerve modules"
     * "Passive caster suspension"
   - 下部（高度 1.4 in）：三枚横向并列扁平徽章（极简矢量图标 + 英文短语）：
     * "Narrow passages"
     * "Confined inspection"
     * "Multi-terrain transit"

2. 右侧栏（Motivation Gap → Research Goal，上下两块）：
   - 上半块（高度 2.2 in，双列对比卡片 + 中间空心过渡箭头）：
     * 左卡（浅蓝底 `#EFF6FF`）：标题 "Hardware & Applications"，3 条极简标签："High stiffness & load (closed-chain)"、"Shape adaptation (morphing)"、"Broad real-world demand"。
     * 右卡（浅琥珀底 `#FFFBEB`）：标题 "Algorithmic Gap"，3 条极简标签："Hardware-heavy literature"、"Fixed-topology or open-chain focus"、"Scarce online morph + drive control"。
   - 下半块（高度 2.7 in，浅灰大框，标题 "Research Goal: Unified Three-Layer Closed Loop"）：
     * 绘制一条清晰顺时针闭环流程，4 个圆角节点用蓝色实线箭头依次连接：
       1) "Task & Stage Choice"（标签 Brain）
       2) "Feasible Closed-Chain Morphing"（标签 Spinal）
       3) "Coordinated Swerve Motion"（标签 Cerebellum）
       4) "Joint, Closure & Support Feedback"（绿色实线/虚线反馈弧回到节点 1 与节点 2，标注 "Verify · Adapt · Replan"）。
     * 底部一行小字说明："Goal: validate the full loop in simulation before hardware transfer."

【质量约束】所有卡片、图标、箭头与文本均须为独立可编辑矢量对象，文字字号 11–14 pt，严禁任何元素重叠或压线。
```

### 中文讲稿

闭链机构具备高刚度与高承载，可变形能力带来窄道、受限巡检和复杂地形的强适应性，因此闭链可变形移动机器人应用前景非常广阔。但对照现有文献，工作大多集中在硬件设计、固定拓扑闭链控制或开链/预设模式变形；真正面向“阶段可变闭链耦合 + 移动底盘协同”的统一在线闭环控制算法仍然稀缺。

围绕 TARS 平台——八块闭链机身、八主动关节加四组主动舵轮与被动悬挂——我的研究目标是构建并验证大脑—小脑—脊髓三层闭环控制架构，使机器人能从反馈识别当前可行变形与阶段条件，并与舵轮移动协同完成受限空间任务。但要把这个闭环落地并不容易，下一页我拆解其中最难实现的三个核心问题。

---

# 二、关键问题

## 3. 关键问题：闭链耦合、可变阶段与移动协同 · 1:20

### English slide text

**Three core bottlenecks make closed-chain mobile morphing fundamentally hard**

- **1. Stage-dependent coupling (beyond open chains or a single `φ`):** joints must coordinate within groups, different groups may advance independently, and later motions can require cross-group conditions.
- **2. Feedback-verified mobility & transitions (beyond known locking formulas):** effective constraints shift with pose and contact; admissible directions and transition guards must be tested from feedback rather than assumed fixed offline.
- **3. Coupled morphing and swerve locomotion (beyond stationary or stop-only control):** changing body geometry alters wheel positions, heading, clearance envelope, and four-wheel support in real time.

*Illustrative mechanism hypothesis — to be verified from simulation feedback*

### 中文对照

实现上述研究目标的核心难点集中在三个与既有研究假设不同的问题：第一，闭链耦合与分阶段可行运动——关节既不能像开链那样各自独立运动，也不能仅靠单一形态参数 `φ` 表达全部可行空间，组内需协调、组间可能独立推进、后续新方向可能需要跨组联合条件；第二，约束与转换条件不可离线写死——不同于已知锁止方程的固定变胞机构，有效可动方向与阶段转换条件需由执行反馈检验；第三，变形与舵轮移动强耦合——形变实时改变轮组相对位姿、外轮廓包络与四轮接地支撑，无法当作静止并联平台或纯“停下再变”问题处理。

### 插图与版式

采用上中下紧凑结构（或左机理图 + 右三大难点对比）：
- **主机理示意**：画八个关节的闭链环，旁边用两个独立进度条 *Group A*、*Group B* 表示候选的组间独立进度；用一个小型条件汇合节点表现“两组都到候选边界后再测试新可动方向”，明确标注 *Candidate condition*（不写成已知解锁角或既定分组）。
- **角落对比框**：把单参数 `φ` 作为已知参考轨迹坐标放在角落，标出 *Single `φ` = one preset path ≠ full multi-group & transition space*，直观点出它与现有单参数/开链假设的差异。
- **底部移动耦合条**：简画变形前后四组舵轮相对位置与支撑多边形的变化箭头，标出 *Wheel pose & support shift during morphing*。全图保留显著副注 *Illustrative mechanism hypothesis — to be tested in simulation*。

#### 可编辑 PPT 插图生成提示词（第 3 页）

```text
请遵循【可编辑 PPT 插图通用视觉规范】，生成第 3 页（Core Challenges: Closed-Chain Coupling, Stage Transitions & Swerve Coupling）的原生可编辑 PPTX 与 SVG 矢量插图，尺寸 12.1 × 5.0 in，图内全部文字使用英文。

【布局结构：左侧机理图（7.3 in）+ 右侧三难点与既有假设对比（4.5 in），间距 0.3 in】
1. 左侧机理示意区（浅灰容器，顶部小标签 "Illustrative mechanism hypothesis — to be tested in simulation"）：
   - 左上：8 节点闭链环示意图（8 个圆角小块 B1–B8 首尾相连，节点间标 j1–j8，其中一段闭合边用琥珀虚线标 "Loop closure"；用两种淡色虚线圈出候选 "Candidate Group A" 与 "Candidate Group B"）。
   - 右上：两条并排独立进度条 "Group A progress" 与 "Group B progress"，各带一处琥珀虚线标记 "Candidate boundary"，经一个汇合判断菱形（标注 "Candidate cross-group condition"）指向一个紫虚线框 "Test new feasible direction?"。
   - 右中角落对比小框（淡灰边框）：画一条弯曲一维线段标 `φ` (0 → 3)，配短句 "Single φ indexes one preset path; cannot capture independent group progress or guards."
   - 底部横条（浅蓝底 `#EFF6FF`）：左右两个简化四轮俯视框，用蓝色双箭头连接（标 "Morphing"），展示四个小轮矩形从宽方形变为窄长形，下方标 "Wheel positions, yaw frames & 4-wheel support shift concurrently."

2. 右侧三难点卡片（纵向等高 3 张卡片，每张显式写出“难点 vs 既有简化假设”）：
   - Card 1（蓝边）：标题 "1. Stage-Dependent Coupling"；对比 tag："vs. open-chain / single-φ preset"；一句话："Intra-group coordination + independent group progress."
   - Card 2（紫边）：标题 "2. Unknown Transition Guards"；对比 tag："vs. known analytical locking"；一句话："Effective mobility & guards require feedback validation."
   - Card 3（琥珀边）：标题 "3. Morphing–Locomotion Coupling"；对比 tag："vs. stationary / stop-only platforms"；一句话："Body shape directly perturbs swerve kinematics & support."

【质量约束】所有候选分组与条件边必须用虚线表示，严禁出现具体锁定角度数字（如 90°/0°），确保无重叠、对象全矢量可编辑。
```

### 中文讲稿

承接上一页的研究目标，为什么不能直接套用开链规划或固定结构底盘控制？因为落地时有三个核心瓶颈。

第一是阶段相关闭链耦合：八个电机不能像开链那样随意独立运动，也不能假设永远按单一形态参数 φ 同步推进；例如候选的 A、B 两组可能各自独立推进，而新运动方向只在跨组联合条件满足后才可动。第二是约束与转换条件不能离线写死：相比已知锁止方程的机构，我们的分组、稳定耦合与转换条件都必须从实测反馈检验。第三是变形与移动强耦合：形变实时改变轮位、朝向、外轮廓和四轮支撑，无法把机身和底盘割裂。带着这三个难点，我们看三篇最贴近的前沿工作分别做到了哪一步。

---

# 三、已有工作

## 4. 相关工作与本项目的差异 · 1:50

### English slide text

**Prior work covers closed-loop morphing, parallel-actuated control, and reconfigurable swerve bases**

#### Hierarchical Origami Metastructures · Closed-loop morphing *(Y. Li et al., Nature Communications 2024)*
Builds thick-panel spatial closed-loop polyhedral linkages that transform into `>10³` volumetric shapes with `<3` active DOFs via analytical rigid-folding kinematics on a **stationary** metastructure.  
**TARS adds:** feedback-based identification and repeatable retesting of stage couplings and transition conditions on an actively driven **mobile** closed-chain robot.

#### BRUCE · Closed-chain locomotion control *(Y. Tanaka et al., IEEE-RAS Humanoids 2025)*
Retains differential-pulley, 5-bar, and 4-bar parallel transmissions natively in MuJoCo MJX to train curriculum RL walking policies with zero-shot hardware transfer under a **fixed mode**.  
**TARS adds:** a queryable **spinal three-graph interface** for *stage-varying* closed-chain morphing coupled with swerve locomotion and task planning.

#### Reconfigurable Robotic Base · Morphing + swerve MPC *(J. Pankert et al., IEEE RA-L / IROS 2022)*
Uses unified MPC to coordinate active swerve steering and footprint morphing (narrowing for tight spaces, widening for stability) on a **single-mode, analytically parameterized triangular base**.  
**TARS adds:** swerve-and-morph coordination for an **8-block closed-chain body** whose couplings and admissible motions change across stages and are verified through spinal graphs.

### 中文对照

三项顶刊/顶会代表性工作分别从“闭链多态变形机构”“MuJoCo 闭链仿真控制”“可重构舵轮底盘运动规划”三个维度对应本课题核心边界：
1. **Hierarchical Origami Metastructures（Y. Li 等，*Nature Communications* 2024）**：采用厚板多面体空间闭环折纸机构，仅靠 `<3` 个主动自由度与解析刚性折叠运动学实现 `>10³` 种三维形态切换，奠定了厚块闭链多态变形的机构基础；但其对象为**静止超结构**且依赖离线解析折叠路径，不涉及移动底盘与反馈驱动的阶段条件复测。
2. **BRUCE（Y. Tanaka 等，*IEEE-RAS Humanoids* 2025，UCLA RoMeLa）**：在 MuJoCo MJX 中原生保留差动滑轮、五杆、四杆闭链传动并实现课程 RL 零样本真机行走迁移，验证了闭链仿真与控制路线；但其构型模式在行走中**固定不变**，端到端策略不向上层暴露可查询的形变阶段接口。
3. **Reconfigurable Robotic Base（J. Pankert & M. Hutter 等，*IEEE RA-L / IROS* 2022，ETH Zurich RSL）**：用统一 MPC 协同主动舵轮转向与底盘轮廓收缩/展开，兼顾窄通道穿行与操作稳定性；但其底盘为**单一模式、平滑解析参数化的三角形结构**，未涉及多阶段闭链耦合与反馈验证的阶段转换图。

### 插图与版式

严格按模板相关工作页排三张等宽卡片，每张包含“顶部示意图（自绘矢量或预留论文原图贴图框）+ 核心做法 + 蓝色/绿色高亮 `TARS adds` 差异栏”，底部加一条通栏定位条 *TARS bridges all three: Closed-loop multi-stage morphing + native MuJoCo closed-chain control + swerve-coordinated mobile navigation*：
1. **Origami Metastructures (*Nat. Commun.* 2024)**：自绘或贴入厚板闭环多面体形态变换示意，标出 *Closed-loop thick blocks → >10³ shapes (stationary, analytical)*。
2. **BRUCE (*Humanoids* 2025)**：自绘或贴入并联传动腿部示意，标出 *Native MuJoCo closed-chain → fixed-mode RL locomotion*。
3. **Reconfigurable Swerve Base (*RA-L / IROS* 2022)**：自绘或贴入三角形舵轮底盘窄/宽切换示意，标出 *Swerve + footprint MPC (single-mode base)*。
每张卡片底部保留准确的一作简称、顶刊/顶会名与年份。若直接使用原论文图片，按论文许可注明来源。

#### 可编辑 PPT 插图生成提示词（第 4 页）

```text
请遵循【可编辑 PPT 插图通用视觉规范】，生成第 4 页（Related Work & Positioning）的原生可编辑 PPTX 与 SVG 矢量插图，尺寸 12.1 × 5.3 in，图内全部文字使用英文。

【布局结构：上方三张等宽并列卡片（每张宽 3.83 in，高 4.25 in，间距 0.3 in）+ 底部 TARS 定位横条（宽 12.1 in，高 0.78 in）】
每张卡片内部统一分为 4 层：
  (1) 顶栏标题 + 右上角小标签（一作简称与顶刊/顶会年份）；
  (2) 简明矢量示意图区（高 1.65 in，保留浅灰边框，既可直接展示也可供后续替换论文原图）；
  (3) "What it solves" 摘要区（浅灰底，2 行精炼英文）；
  (4) "TARS adds" 差异高亮区（浅蓝/浅绿底，2 行精炼英文）。

三张卡片具体内容：
- Card 1: "Origami Metastructures" | Tag: "Y. Li et al., Nat. Commun. 2024"
  * 示意图：厚板闭环多面体小块经折叠箭头变成 2–3 种不同三维轮廓，标签 "Closed-loop thick blocks → >10³ shapes (stationary, analytical)"。
  * What it solves: "Hierarchical closed-loop thick origami achieves >10³ shapes with <3 DOFs via analytical folding."
  * TARS adds: "Feedback-based identification & repeatable retesting of stage transitions on a mobile robot."

- Card 2: "BRUCE · Parallel Humanoid" | Tag: "Y. Tanaka et al., Humanoids 2025"
  * 示意图：简化人形下肢旁画并联四杆/五杆与滑轮小图标，箭头指向 "MuJoCo MJX + Curriculum RL → Hardware"，标签 "Native closed-chain sim → fixed-mode locomotion"。
  * What it solves: "Retains parallel closed-chain transmissions in MuJoCo for zero-shot RL walking transfer."
  * TARS adds: "Queryable spinal three-graph interface for stage-varying morphing & upper-layer planning."

- Card 3: "Reconfigurable Swerve Base" | Tag: "J. Pankert et al., RA-L / IROS 2022"
  * 示意图：带 3 组舵轮的三角形底盘在 "Narrow pass" 与 "Wide stance" 之间切换，旁标 "Unified MPC"，标签 "Active swerve + footprint morph (single-mode kinematics)"。
  * What it solves: "Coordinates swerve steering and footprint morphing via unified MPC for narrow spaces."
  * TARS adds: "Extends morph + swerve control to an 8-block closed chain with stage-dependent couplings."

- 底部通栏定位条（深蓝边框、浅蓝底 `#EFF6FF`）：
  * 左侧加粗徽章 "TARS Positioning"；
  * 右侧横排三项连线："Feedback-verified multi-stage morphing (vs. Origami)  +  Queryable stage graphs (vs. BRUCE)  +  Closed-chain swerve coordination (vs. ETH Base)"。

【质量约束】确保卡片排版高度对齐，字体 ≥ 11 pt，无任何文字溢出或遮挡，全部对象可独立编辑。
```

### 中文讲稿

这里选取三篇顶刊与顶会代表工作，对应我们前页的三个核心维度来界定 TARS 的研究边界。

第一篇是 *Nature Communications* 2024 的分层折纸超结构（Li 等）：用厚板闭环多面体和不到 3 个主动自由度实现上千种三维形态，直观展现了闭链块体的变形潜力；但它面向静止超结构、依赖离线解析折叠，不涉及移动与反馈复测。

第二篇是 UCLA 发表于 *Humanoids 2025* 的 BRUCE：在 MuJoCo MJX 中原生保留并联闭链传动并实现课程 RL 零样本真机行走，说明闭链仿真迁移可行；但其构型模式固定，端到端策略不提供可查询的变形阶段接口。

第三篇是 ETH 发表于 *RA-L / IROS 2022* 的可重构舵轮底盘（Pankert & Hutter 等）：用统一 MPC 协同主动舵轮和底盘收缩穿窄道；但它基于平滑解析的单一模式三角形底盘，没有多阶段闭链耦合。TARS 正处在三者交汇空白：用反馈更新的三图脊髓，把多阶段闭链变形、原生闭链控制和主动舵轮移动闭合成统一框架。

### 已核对的论文

- Y. Li, A. Di Lallo, J. Zhu, Y. Chi, H. Su, and J. Yin, “Adaptive hierarchical origami-based metastructures,” *Nature Communications*, vol. 15, no. 1, Art. no. 6247, 2024. DOI: `10.1038/s41467-024-50497-5`. [期刊论文](https://www.nature.com/articles/s41467-024-50497-5)
- Y. Tanaka, A. Zhu, Q. Wang, Y. Liu, and D. Hong, “Mechanical Intelligence-Aware Curriculum Reinforcement Learning for Humanoids with Parallel Actuation,” *IEEE-RAS International Conference on Humanoid Robots (Humanoids)*, 2025 (`arXiv:2507.00273`). [论文](https://arxiv.org/abs/2507.00273) · [项目页](https://alvister88.github.io/BRUCE_RL_Website/)
- J. Pankert, G. Valsecchi, D. Baret, J. Zehnder, L. L. Pietrasik, M. Bjelonic, and M. Hutter, “Design and Motion Planning for a Reconfigurable Robotic Base,” *IEEE Robotics and Automation Letters (RA-L)*, vol. 7, no. 4, pp. 9012–9019, 2022 *(also presented at IROS 2022)*. DOI: `10.1109/LRA.2022.3189166`. [论文](https://arxiv.org/abs/2206.15298) · [DOI](https://doi.org/10.1109/LRA.2022.3189166)

---

# 四、我的思路

## 5. 大脑—小脑—脊髓的闭环职责 · 1:20

### English slide text

**Task decisions, morphing and locomotion are linked by feedback**

- **Brain — FSM navigation and PPO-based exploration**  
  FSM selects the route and target stage; PPO is the planned policy for exploring task-level choices.
- **Spinal layer — three-graph representation and morphing control**  
  Identifies stage-dependent feasible motions, returns a coordinated joint path and transition status.
- **Cerebellum — body-motion and swerve coordination**  
  Balances morphing with body translation, yaw, wheel steering and wheel speed.

**Feedback closes the loop:** joint response and closure status → spinal layer; clearance, support and tracking → cerebellum; outcome → brain.

### 中文对照

大脑用 FSM 导航并选择目标阶段，PPO 作为计划中的策略探索任务层选择。脊髓利用三图表征阶段相关的可行运动，输出协调关节轨迹和转换状态。小脑协调形变与机身平移、朝向、舵角和轮速。关节响应与闭合状态回脊髓，净空、支撑和跟踪误差回小脑，执行结果回大脑。

### 插图与版式

三层控制闭环框图，**视觉重心放在“三层分工 + 双向接口 + 三路反馈闭合”**（脊髓模块内只放紧凑的 `G_mode / G_state / G_phase` 三图徽章，把三图细节留到第 6 页展开，避免信息过载；注：仓库旧图 `outputs/three_layer_three_graph_20261114/three_layer_three_graph_framework.pptx` 术语与版面偏满，建议按下方提示词重生成与本稿术语完全一致的简洁新版）。PPO 用虚线框标为 *Planned option*。

#### 可编辑 PPT 插图生成提示词（第 5 页）

```text
请遵循【可编辑 PPT 插图通用视觉规范】，生成第 5 页（Brain–Cerebellum–Spinal Closed-Loop Architecture）的原生可编辑 PPTX 与 SVG 矢量框图，尺寸 12.1 × 5.2 in，图内全部文字使用英文。

【布局结构：四层纵向堆叠闭环框图（居中主干宽 8.2 in，左右两侧留 1.8 in 专用于清晰的指令下行与反馈上行箭头通道，杜绝箭线穿越文字）】
1. 顶层 — Brain（浅蓝底 `#EFF6FF`，蓝边框）：
   - 左侧标题："Brain · Task & Stage Decision"
   - 内部两枚子框：实线框 "FSM Navigation & Stage Selector" + 虚线框 "PPO Task Exploration (Planned)"
2. 中层（左右并列两张协同卡片，总宽 8.2 in，中间留双向协作箭头）：
   - 左卡 — Spinal Layer（浅紫底 `#F5F3FF`，紫边框，宽 4.0 in）：
     * 标题："Spinal Layer · Morphing & Three Graphs"
     * 内部 3 个紧凑徽章：`G_mode`、`G_state`、`G_phase`
     * 下方短句："Check reachability · generate coordinated 8-joint path"
   - 右卡 — Cerebellum（浅琥珀底 `#FFFBEB`，琥珀边框，宽 4.0 in）：
     * 标题："Cerebellum · Locomotion & Swerve Coordination"
     * 内部短句："Coordinate morph progress with body translation, yaw, 4× steer & wheel speed"
   - 两卡之间双箭头标注："shape path, expected time & status ↔ clearance & support gate"
3. 底层 — Closed-Chain Robot Plant（深灰边框、白底）：
   - 标题："TARS Plant (MuJoCo Simulation → Future Hardware)"
   - 内部标示："8 Morphing Joints (Closed Loop)  |  4 Active Swerve Modules + 8 Spring Casters"

【双向接口与反馈回路（严格走外侧通道，箭头折线横平竖直）】
- 左侧/中部下行指令箭头（蓝色实线 `#1D4ED8`）：
  * Brain → Spinal："Target shape / stage request"
  * Brain → Cerebellum："Route & motion goal"
  * Spinal → Plant："Coordinated 8-joint command"
  * Cerebellum → Plant："4× steering & wheel speed command"
- 右侧上行反馈主干（绿色实线/虚线 `#059669`，分三路接入）：
  * Plant → Spinal："8-axis joint response & closure residual"
  * Plant → Cerebellum："Clearance, 4-wheel support & tracking error"
  * Spinal & Cerebellum → Brain："Reached / Pause / Failure → Replan"

【图例】底部一行极简图例：蓝实线 = Control / query；绿线 = Execution feedback；虚线框 = Planned extension (PPO)。
```

### 中文讲稿

大脑负责任务目标和导航，FSM 请求目标阶段，PPO 是后续探索任务层选择的方向。脊髓根据当前反馈给出是否可达以及可执行的协调变形轨迹。小脑把机身形态变化与世界坐标中的移动、朝向、舵角和轮速协调起来。执行中的关节和闭合反馈更新脊髓状态；净空、支撑和跟踪情况交给小脑；到达、暂停或失败的结果回到大脑，供 FSM 继续或重新规划。

## 6. 脊髓三图：从物理连接到可执行阶段 · 1:25

### English slide text

**Three graphs represent physical structure, local motion and stage transitions**

| Graph | Main question | Information returned to the controller |
|---|---|---|
| **G_mode · Effective topology** | Which blocks behave as one group in this mode? | Candidate rigid groups and the range where they remain valid |
| **G_state · Stage-local constraints** | Which joint motions are feasible at this state? | Local motion directions, coupling evidence and confidence |
| **G_phase · Transition graph** | Which verified action reaches another stage? | Directed action, entry condition, exit test and failure evidence |

### 中文对照

G_mode 表示当前模式中哪些 block 可视为一组及其有效范围。G_state 表示当前姿态下可行的关节运动方向、耦合证据和置信度。G_phase 表示经过验证的阶段动作、进入条件、退出检测和失败证据。

### 插图与版式

页面上半部分放简洁三行表格（或将表格压缩为每图下方的信息条），主视觉为**三幅横向并列图**，共享同一套 8-block 闭链环基底（注意不要复用旧图 `outputs/three_graph_framework_20261105/three_graph_framework.pptx` 中的四电机 `G_phys/G_motion^(s)` 命名，统一改为本稿 8-block 的 `G_mode / G_state / G_phase`）：
- **G_mode**：8-block 物理相邻边作浅灰实线，闭合边作标记，候选刚性功能组作彩色虚线包络。
- **G_state**：8 个电机/关节节点、局部响应边、多关节约束因子节点及置信度标注。
- **G_phase**：阶段节点与带有 *Guard / Action / Exit test* 的有向边；候选边用虚线，经复测验证后才用实线。

#### 可编辑 PPT 插图生成提示词（第 6 页）

```text
请遵循【可编辑 PPT 插图通用视觉规范】，生成第 6 页（Three-Graph Spinal Representation: G_mode, G_state, G_phase）的原生可编辑 PPTX 与 SVG 矢量插图，尺寸 12.1 × 4.6 in（留出上方或下方空间给简表），图内全部文字使用英文。

【布局结构：三张等宽并列图卡（每张宽 3.83 in，高 3.9 in，间距 0.3 in）+ 底部共享数据源与图例条（高 0.55 in）】
1. Card 1 — G_mode · Effective Topology（蓝标题 `#1D4ED8`）：
   - 副标题："Which blocks behave as one group?"
   - 图示：规则八边形闭链（8 个小圆角块 B1–B8，浅灰实线代表已知 CAD 物理相邻，B8–B1 标注 "Loop weld"）；用 2–3 个半透明蓝色虚线圆角包络圈出候选功能组（标注 "Candidate rigid group"），另留一处标注 "Undecided edge (insufficient evidence)"。
   - 底部返回信息条："Returns: candidate groups & valid pose range"

2. Card 2 — G_state · Stage-Local Constraints（紫标题 `#7C3AED`）：
   - 副标题："Which joint motions are feasible here?"
   - 图示：上下两排节点——上排局部指令/电机节点 (u_1..u_8 精简示意 4–6 个代表节点 + "...")，下排关节响应节点 (j_1..j_8)，中间连接紫色耦合因子圆角小菱形 `f_1, f_2`；强稳定耦合画紫色实线标 "high conf."，弱/待确认边画灰虚线标 "candidate / low conf."。
   - 底部返回信息条："Returns: local directions, coupling evidence & confidence"

3. Card 3 — G_phase · Transition Graph（琥珀标题 `#D97706`）：
   - 副标题："Which verified action reaches another stage?"
   - 图示：3 个阶段圆角节点 `Stage S0`、`Stage S1`、`Stage S2`：
     * S0 → S1 画绿色/琥珀**实线**有向弧，贴三段小标签："Guard | Verified Action | Exit test"；
     * S1 → S2 画灰色**虚线**有向弧，标 "Candidate edge (unverified)"；
     * 反向画一条带小叉标记的红色/灰虚线短弧，标 "Failure evidence (not symmetric)"。
   - 底部返回信息条："Returns: directed action, guard, exit test & failure log"

4. 底部共享底座条（浅灰底）：
   - 左："Shared physical prior: CAD 8-block closed loop + 8-axis feedback"
   - 右图例："Gray solid = given physical link   |   Dashed = candidate (unverified)   |   Colored solid = retest-verified"

【质量约束】确保三张卡片的 8-block 编号与术语一致，虚实线区分一目了然，全部对象原生可编辑。
```

### 中文讲稿

这三张图不是三种神经网络，也不是把同一结构画三遍。模型提供原始机械连接。G_mode 再描述当前运动模式下哪些 block 的相对位姿保持不变，可以作为一个功能组。G_state 记录这一阶段附近，哪些关节方向可动、各轴如何响应以及证据有多稳定。G_phase 把经过验证的动作连成阶段图，告诉上层从当前阶段怎样进入下一阶段。它们给 FSM 和小脑提供各自能使用的信息。

## 7. 脊髓如何从反馈建立和更新图 · 1:15

### English slide text

**Candidate couplings become reusable actions only after validation**

1. Apply a bounded probe in the current state.
2. Observe all eight joint angles and velocities.
3. Estimate local response and closure residual.
4. Retest after retreating or resetting.
5. Store the relation with its valid range and evidence.

*The identification algorithm is an open design choice.*

### 中文对照

在当前状态施加有界试探，观察八轴角度和速度，估计局部响应与闭合残差。退回或复位后重测，只有通过复测的关系才连同适用范围和证据存入图中。具体辨识算法仍待比较。

### 插图与版式

画六步闭环流程 *1. Bounded Probe → 2. Observe (q, q̇) → 3. Estimate Response & Closure Residual → 4. Retreat / Reset & Retest → 5. Update Graph (Candidate → Verified) → 6. Execute / Replan*（注意不要使用旧中文四步图 `artifacts/figures/active_constraint_learning_loop_2x1.svg`，按下方提示词生成全英文六步矢量闭环）。主反馈（8 轴角度、速度、闭合残差）用实线，电流/力矩作为可选负载辅助信号用灰色虚线支路标出；候选态与已验证态用虚/实线型严格区分。

#### 可编辑 PPT 插图生成提示词（第 7 页）

```text
请遵循【可编辑 PPT 插图通用视觉规范】，生成第 7 页（Feedback-Driven Graph Discovery & Validation Loop）的原生可编辑 PPTX 与 SVG 矢量流程图，尺寸 12.1 × 4.8 in，图内全部文字使用英文。

【布局结构：两行三列 U 形闭环流（上排 1→2→3 从左到右，下排 6←5←4 从右到左，右侧 3→4 向下，左侧 6→1 反馈闭环向上）】
6 个圆角步骤卡片（每张宽 3.45 in，高 1.45 in，带左上角编号圆标）：
- Step 1（蓝）："1. Bounded Probe" — "Apply small, torque- & rate-limited joint command in current state"
- Step 2（蓝）："2. Observe 8-Axis Feedback" — "Primary (solid): 8 joint angles q & velocities q̇" + 右下角灰色虚线小附标 "Optional load aid: motor current / torque"
- Step 3（紫）："3. Estimate Local Model" — "Fit local response & check loop-closure residual (algorithm kept open)"
- Step 4（琥珀，高亮门控框）："4. Retreat / Reset & Retest" — "Return to source state and repeat probe; reject non-repeatable response"
  * 从 Step 4 引出一条向下的灰虚线分支标 "Failed retest → keep candidate / log failure"
- Step 5（绿）："5. Update Three Graphs" — "Promote candidate (dashed) → verified (solid) with valid range & evidence"
- Step 6（蓝/绿）："6. Execute or Replan" — "Serve verified path to Cerebellum & FSM; trigger re-probe on anomaly"

【视觉与语义重点】
- Step 3 下方放一条极简灰底提示胶囊："Open algorithmic choice: linear response / sparse graph / future learned models"
- 右下角小图例：Blue/Purple solid = primary flow；Green solid = verified update；Gray dashed = optional signal / unverified path。

【质量约束】箭头折线拐角整齐不压文字，6 张卡片严格网格对齐，全部形状与文本原生可编辑。
```

### 中文讲稿

脊髓的算法目前不预先承诺是强化学习还是某一种图网络。第一步是用幅度受限的试探获取真实输入与八轴输出的关系，并用闭合误差和复测来检查关系。随后才能比较简单响应矩阵、按状态区分的局部模型和图结构模型。图的作用是把部件、关节、阶段和证据组织起来，并为上层提供可查询的信息；图本身不能替代识别与验证算法。

## 8. 端到端闭环：从目标阶段到可验证的执行结果 · 1:05

### English slide text

**A stage request becomes a coordinated body-and-wheel action**

**Brain:** request target stage  
→ **Spinal layer:** check reachability and return a verified shape path  
→ **Cerebellum:** coordinate morphing progress with translation, yaw, steering and wheel speed  
→ **Robot feedback:** joint response, closure status, support and tracking  
→ **Brain / graphs:** mark reached, pause, update evidence or replan

### 中文对照

大脑请求目标阶段，脊髓检查可达性并返回验证过的形变路径，小脑协调形变进度与平移、朝向、舵角、轮速。机器人反馈关节响应、闭合状态、支撑和跟踪情况，供大脑与三图决定到达、暂停、更新证据或重规划。

### 插图与版式

不用窄道沿途包络案例。改画一条清晰的**五阶段闭环时序带**：*1. Request* → *2. Graph Query* → *3. Joint Path + Swerve Command* → *4. Observed Response* → *5. Reached / Pause / Replan*，并在右侧画一条反馈折回线连回 *Request* 与 *Graph Query*。下方横排三个圆角形态缩略图框（*Initial shape → Transitioning shape → Target shape*，预留 MuJoCo 实拍图贴图位或极简线框），**不填未经验证的角度或性能数值**。

#### 可编辑 PPT 插图生成提示词（第 8 页）

```text
请遵循【可编辑 PPT 插图通用视觉规范】，生成第 8 页（End-to-End Closed-Loop Execution Timeline）的原生可编辑 PPTX 与 SVG 矢量插图，尺寸 12.1 × 4.9 in，图内全部文字使用英文，不含任何未经验证的性能数值或具体角度。

【布局结构：上方五阶段闭环时序带（高 2.75 in）+ 下方三态形态演变占位条（高 1.85 in）】
1. 上方时序带（5 个等宽横向阶段卡片，每张宽 2.18 in，间距 0.25 in，顶部标执行层彩色徽章）：
   - Col 1（Badge: "Brain", 蓝）：标题 "1. Request" — "Select target stage & route goal"
   - Col 2（Badge: "Spinal", 紫）：标题 "2. Graph Query" — "Check reachability; return verified joint path"
   - Col 3（Badge: "Cerebellum", 琥珀）：标题 "3. Coordinated Cmd" — "Sync morph progress with twist, steer & wheel speed"
   - Col 4（Badge: "Robot Feedback", 绿）：标题 "4. Observe" — "Joint q/q̇, closure residual, support & tracking"
   - Col 5（Badge: "Brain / Graphs", 深蓝）：标题 "5. Decision" — 3 枚极简状态标签："Reached (green)"、"Pause (amber)"、"Replan (gray/red)"
   - 时序带底部画两条折返反馈线（绿色虚线/实线）：
     * 从 Col 4/5 折返到 Col 2："Update graph evidence"
     * 从 Col 5 折返到 Col 1："Continue or replan stage"

2. 下方三态形态示意区（3 个圆角虚线贴图框 + 中间 2 个过程箭头，总宽 12.1 in）：
   - Box 1："Start State (e.g., open / rest pose)" — 预留贴 MuJoCo 截图，底部小字 "Initial body & wheel footprint"
   - 箭头 1 标："Coordinated morph + swerve"
   - Box 2："Mid-Transition State" — 预留贴 MuJoCo 截图，底部小字 "Monitor closure & 4-wheel support"
   - 箭头 2 标："Verify exit test"
   - Box 3："Target Stage Reached" — 预留贴 MuJoCo 截图，底部小字 "Confirm target shape & stability"

【质量约束】五列卡片严格等高对齐，底部折返线与下方三态框之间留 ≥ 0.25 in 净空，全矢量可编辑。
```

### 中文讲稿

完整闭环可以用一次阶段请求说明。大脑提出目标形态和路线要求，脊髓查询图里是否有适用的阶段动作，返回可达性和关节轨迹。小脑安排机器人如何一边完成形态变化，一边控制机身运动和舵轮。执行过程中，关节是否按预期响应、闭链误差是否受控、轮组是否保持支撑以及路径是否跟踪到位，都会反馈回来。系统据此确认到达、暂停执行、更新图上的证据或让大脑重新规划。

## 9. 如何验证三层图脊髓的作用 · 1:10

### English slide text

**Test prediction, transition validation and closed-loop task execution**

- **Stage-local motion:** compare independent-joint, dense-response and graph-structured models on held-out joint commands.
- **Transition conditions:** test alternative group-progress conditions with the same probe, then retreat and repeat.
- **Closed-loop control:** compare a reference-path controller with a graph-informed stage request and coordinated swerve control.

**Metrics:** joint-angle error · closure residual · probes to reliable prediction · false transition rate · task completion · path and shape error · support/contact status

### 中文对照

验证三个层面：在留出关节指令上比较独立关节、稠密响应和图结构模型；用相同试探测试不同关节组进度条件，并退回复测；再比较参考路径控制与图信息驱动的阶段请求及舵轮协调。指标包括角度误差、闭合残差、达到可靠预测所需试探数、错误转换率、任务完成情况、路径与形态误差、支撑与接触状态。

### 插图与版式

画**三行“问题—对照协议—核心指标”的实验验证矩阵卡片**，顶部醒目标注 *Planned validation — identical model version, initial state, probe budget & chassis parameters*，每行左侧配小型流程图标（第 2 行嵌入 2×2 条件对照格 *Both far / Only A near / Only B near / Both near*，并注明 *Actual grouping determined by feedback*）。**严禁画虚构曲线、柱形图或预测性数值。**

#### 可编辑 PPT 插图生成提示词（第 9 页）

```text
请遵循【可编辑 PPT 插图通用视觉规范】，生成第 9 页（Planned Validation Matrix）的原生可编辑 PPTX 与 SVG 矢量图，尺寸 12.1 × 5.2 in，图内全部文字使用英文，严禁出现任何柱状图、折线图或虚构实验数值。

【布局结构：顶部统一公平性横幅（高 0.45 in）+ 三行水平实验卡片（每行高 1.38 in，间距 0.18 in）】
- 顶部横幅（浅琥珀底 `#FFFBEB`，琥珀边框）：
  "Planned Validation Protocol · Fair controls under identical model version, initial state, probe budget & swerve parameters"

- 每行卡片横向分 3 列：Col 1（Research Question，宽 3.1 in）| Col 2（Controlled Comparison，宽 5.3 in）| Col 3（Key Metrics，宽 3.3 in）：
  * Row 1（蓝左边条，Tag: "Exp A · G_state"）：
    - Question: "Does graph structure improve held-out motion prediction?"
    - Comparison（3 枚并列小盒 + 同数据标记）: "Same probe budget & held-out joint cmds → [Independent] vs. [Dense linear] vs. [Sparse validated graph]"
    - Metrics: "Per-joint & total angle error · Closure residual · Probes vs. error · Trial repeatability"
  * Row 2（紫左边条，Tag: "Exp B–C · G_mode & G_phase"）：
    - Question: "Do group-progress conditions truly govern transitions?"
    - Comparison（内置微型 2×2 四格示意 + 回退箭头）: "2×2 probe grid: [Both far | Only A near | Only B near | Both near] + Retreat & Retest (groups verified by feedback)"
    - Metrics: "Transition repeatability · False unlock rate · Group break rate · Closure & torque safety"
  * Row 3（绿左边条，Tag: "Exp D · Full Closed Loop"）：
    - Question: "Do spinal graphs improve task execution & swerve coordination?"
    - Comparison（3 条平行路径小条）: "Same task & chassis → [Ref φ replay] vs. [Global model] vs. [Verified three-graph query + FSM + swerve]"
    - Metrics: "Task completion · Path & shape error · Clearance & 4-wheel support · Aborts & time"

【质量约束】三行卡片垂直对齐清晰，2×2 小格内文字简洁不拥挤，全部对象可编辑。
```

### 中文讲稿

实验要回答图结构是否真正帮助控制，而不只是把数据画成图。首先检查它能否从相同数量的试探中更准确地预测新联合动作。然后用干预和回退复测区分哪些条件真的改变可行运动。最后把得到的阶段信息接入大脑和小脑，和仅沿参考路径执行的方式比较任务完成与安全指标。所有对照都在相同模型版本、初态、动作预算和底盘条件下进行。

---

# 五、模型选型对比与实测进展

## 10. 完整 CAD 模型导出与电机选型对比（预计 vs 实测） · 1:25

### English slide text

**Full CAD MuJoCo export reveals a 44% heavier robot, lower body peaks, and a tighter hub-drive cap**

| Item | Early Surrogate Estimate *(Oct 2026)* | Full Exported CAD Model *(Nov 2026)* | Engineering Takeaway |
|---|---|---|---|
| **Closed-chain & support** | Inferred 8th hinge (~5.18 mm gap, ~14.6° offset); rigid caster proxy (`R=25.5 mm`) | Native 8-joint CAD + exact 6-DoF `weld` (`<1e-10`); **8 explicit 6 mm spring casters** (`R=28 mm`) | Eliminates artificial closure fighting; realistic tire/caster load sharing |
| **Total mass** | `6.27–6.42 kg` (assumed +0.15 kg closure) | **`9.26 kg`** full CAD mass (**+44%**) | Higher inertial & rolling load on drive modules |
| **8 body joints (`φ=0→3`, 6 s)** | RMS `0.39–1.40 N·m`; peak **`8.31–9.02 N·m`** *(1.5× spec ≥ 2.5 / 14 N·m)* | CAD sweep peak **`4.28 N·m`** (RMS `0.61–1.38`); w/ ramps & wheel coord: **`3.56 N·m`** (RMS `0.38–1.18`) | **Peak dropped ~50–60%**; zero saturation under `11 N·m` cap *(flat ground, no payload)* |
| **4 swerve modules (hub / steer)** | `6.27 kg` up to `0.4 m/s`, 15° slope: hub `0.21–0.43 N·m`, steer `0.71–0.98 N·m` | `±0.45 N·m` hub saturates above **`0.15–0.20 m/s`** (steady flat max **`~0.23 m/s`**, 86 rpm); steer RMS `0.18–0.30 N·m` | **Key gap:** `0.45 N·m` hub caps flat speed below `0.4 m/s`; ok for low-speed tests, upgrade needed for fast/slope driving |

*Flat-ground unloaded comparison; static multi-pose holds, slopes, and real friction/gearbox losses remain to be calibrated.*

### 中文对照

将此前 PPT 中基于 7 关节推导替身模型的电机预估，与最新导出的完整 9.26 kg CAD 闭链 MuJoCo 模型（含 8 个显式 6 mm 弹簧万向球悬挂）实测数据逐项对比，发现两处核心差距：
1. **整机更重但机身形变峰值反而显著下降**：完整 CAD 整机质量从 `6.42 kg` 增至 **`9.26 kg`（+44%）**，但消除推算铰点误差（原 ~14.6° 对称偏移）并引入缓升与轮组协调后，8 个机身形变电机 6 s 全程峰值扭矩由旧推算的 **`8.31–9.02 N·m` 降至 `3.56–4.28 N·m`**（RMS `0.38–1.38 N·m`，全程不触 `11 N·m` 上限）；
2. **轮毂驱动成为新的速度瓶颈**：在真实 `9.26 kg` 质量下，当前 `±0.45 N·m` 轮毂上限在平地指令速度超过 `0.15–0.20 m/s` 后即饱和，平地稳态极速约 **`0.22–0.23 m/s`（~86 rpm）**，低于早期 `6.27 kg` 模型按 `0.4 m/s` 的预期；转向电机稳态 RMS `0.18–0.30 N·m`（阶跃对齐瞬态触及 `3.0 N·m`，平滑折叠 `< 0.84 N·m`）。

### 插图与版式

采用“左表右图”结构：
- **左侧（约占 60%）**：放上方四行对比精简表（高亮 `+44% mass`、`-50%~60% body peak`、`hub saturates @ 0.15–0.20 m/s` 三个关键数字）。
- **右侧（约占 38%）**：上下叠放两张**已有真实测试图**（直接取自仓库，若重绘请按无网格规范导出）：
  1. 上图：完整 CAD 8 轴扭矩曲线（参考 `experiments/grounded_phi0_3/cad_torques.png`）；
  2. 下图：轮毂速度-扭矩饱和扫频曲线（参考 `experiments/grounded_block50_torque/wheel_speed_sweep.png`，直观标出 `0.45 N·m` 饱和拐点）。

#### 可编辑 PPT 插图生成提示词（第 10 页）

```text
请遵循【可编辑 PPT 插图通用视觉规范】，生成第 10 页（Model Upgrade & Motor-Sizing Comparison: Early Estimate vs. Full CAD）的原生可编辑 PPTX 与 SVG 矢量图，尺寸 12.1 × 5.2 in，图内全部文字使用英文，无任何背景网格线。

【布局结构：左侧对比表（宽 7.4 in）+ 右侧三枚核心差距数字卡片与已有实测图贴图区（宽 4.4 in，间距 0.3 in）】
1. 左侧精简对比表（4 行 × 4 列，深蓝表头、浅灰斑马纹、原生可编辑单元格）：
   - 列标题："Metric" | "Early Surrogate (Oct)" | "Full CAD Model (Nov)" | "Impact"
   - Row 1: "Mass & Loop" | "6.42 kg · inferred 8th hinge (~14.6° offset)" | "9.26 kg (+44%) · exact CAD weld + 8 spring casters" | "Higher rolling load; no closure fight"
   - Row 2: "Body Torque (φ=0→3, 6s)" | "Peak 8.31–9.02 N·m · RMS 0.39–1.40" | "Peak 3.56–4.28 N·m · RMS 0.38–1.38" | "Peak -50–60%; 0 s at 11 N·m cap"
   - Row 3: "Hub Drive (4×)" | "6.27 kg est.: 0.21–0.43 N·m (to 0.4 m/s)" | "0.45 N·m cap saturates >0.15–0.20 m/s" | "Flat steady max ~0.23 m/s (86 rpm)"
   - Row 4: "Steering (4×)" | "Peak 0.71–0.98 N·m" | "RMS 0.18–0.30 N·m (<0.84 N·m smooth fold)" | "3.0 N·m cap sufficient w/ smooth slew"
   - 表底一行小字注脚："Flat ground, unloaded; multi-pose holds, slopes & real gearbox losses pending calibration."

2. 右侧栏（上中下 3 块）：
   - 顶部双 KPI 徽章（并列，高 1.15 in）：
     * 绿底徽章："-50–60%" + "Body Peak Torque"（副注 "9.02 → 3.56–4.28 N·m"）
     * 琥珀底徽章："+44% Mass" + "Hub Speed Cap"（副注 "0.40 → ~0.23 m/s @ 0.45 N·m"）
   - 中部贴图框（高 1.85 in，虚线边框）："Insert cad_torques.png (8-Joint Morphing Torques)"
   - 底部贴图框（高 1.85 in，虚线边框）："Insert wheel_speed_sweep.png (Hub Torque Saturation vs. Speed)"

【质量约束】数字完全严格采用上述真实保存数据，不得改动或编造额外曲线；表格和徽章全原生可编辑。
```

### 中文讲稿

在展示近期控制测试前，我先用一页交代完整 CAD 模型导出后，和之前 PPT 电机选型预估之间的实际差距。早期我们在 6.42 公斤的推算闭链替身模型上，测得机身 8 电机峰值高达 8.3 到 9.0 牛米。这两天导出完整的 9.26 公斤 CAD 模型、并加入 8 个显式 6 毫米万向球弹簧悬挂后，发现了两个重要变化。

第一，虽然整机重了 44%，但消除推算铰点误差并加入轮组协调后，机身形变峰值扭矩反而降到了 3.56 到 4.28 牛米，下降超过一半，全程不触及 11 牛米上限，说明机身电机的余量比之前乐观。

第二，反过来，轮毂驱动电机在 9.26 公斤质量下变得更吃紧：0.45 牛米的限幅在平地超过每秒 0.15 到 0.20 米就开始饱和，平地稳态极速在每秒 0.23 米左右，低于早期 0.4 米每秒的预期；转向电机在平滑调速下余量充足。这对我们后续低速闭环实验是够用的，但真机若要跑高速或爬坡需要重点复核轮毂选型。

## 11. 近期实测进展：基础移动、显式悬挂与形变支撑验证 · 1:35

### English slide text

**Grounded chassis, 8-caster suspension, moving-morph, and local branch feasibility are verified**

- **Explicit 8-caster spring suspension (`k=2400 N/m`, 6 mm):** raises coast-morph 4-wheel support from **`50.8% → 90.3%`** (`82.1 → 23.6 mm` peak error) and halves tire penetration (`3.24 → 1.71 mm`) with **0 N** housing scrape.
- **Planar drive & 4 mm bump (`TarsGroundedEnv`):** 6-direction travel (`x± / y± / yaw±`) achieves **`96.5–98.9%`** 4-wheel support and **`≤ 0.28°`** yaw drift; single-caster 4 mm bump recovers **`100%`** support.
- **Moving-while-morphing & steering decoupling (`RP¹` scrub-gradient + block-yaw FF):** four-frame decoupling (`{W}/{B}/{R₀}/{Mᵢ}`) + urgency-gated projective-circle (`RP¹`) steering eliminates `φ≈2` 180° flips (`lateral` max `|q|` `530° → 102.5°`, all 4 wheels return to `0°`; `moving` `φ∈[1.9,2.2]` swing `200.6° → 127.6°`), reaching **`100%`** (`lateral`), **`99.9%`** (`still`), **`97.2–98.6%`** (`moving/reverse/half`) 4-wheel support, COM error **`11.5–15.2 mm`**, weld gap **`< 0.09 mm`**.
- **Spinal-readiness checks:** closure rank dips `6 → 5` near `φ ≈ 1.434` with **14/14** signed finite branches passing stock limits (`peak < 3.95 N·m`); encoder-based 21-group envelope runs in **`0.80 ms`** (p95) and flags a `-21.3 mm` mid-corridor overlap fixed to `+7.2 mm` by a `28.5 mm` lateral shift.

### 中文对照

近两日在最新 `tars_mjcf_grounded` 模型上完成了基础移动框架与底层物理前提的系统验证并已冻结定版：
1. **8 万向球显式弹簧悬挂**：相比旧刚性接触代理，滑行变形四轮着地率由 `50.8%` 提升至 `90.3%`（峰值偏移 `82.1 → 23.6 mm`），轮胎最大压入减半（`3.24 → 1.71 mm`），全程无轮罩/机身刮地；
2. **六向平面行驶与 4 mm 越障**：`x± / y± / yaw±` 四轮承载率 `96.5%–98.9%`、偏航漂移 `≤ 0.28°`，单侧 4 mm 凸起越障后恢复 `100%` 四轮承载；
3. **边走边变形与四坐标系舵轮解耦（已定版）**：建立 `{W}/{B}/{R₀}/{Mᵢ}` 四坐标系与无根块脊髓变形接口 `spinal_morph_kinematics()`，并采用紧迫度门控射影圆（$\mathbb{R}P^1$）侧滑能量梯度 + 自适应安装块偏航前馈，彻底消除 `φ≈2` 附近 180° 急转/回弹（`lateral` 单轮最大转角从 `530°` 压至 **`102.5°`** 且四轮归零、`100%` 接地、倾角 `6.0°`；`moving` `φ∈[1.9,2.2]` 摆幅从 `200.6°` 降至 `127.6°`；5 种主动工况四轮承载率 **`97.2%–100%`**，闭链 `weld` 间隙 `< 0.09 mm`）；
4. **脊髓前置验证**：在 `φ≈1.434` 附近测得闭链雅可比秩降（6→5）且 **14/14** 个带符号有限位移分支全部在原厂力矩上限内通过（`< 3.95 N·m`）；编码器包络计算 `p95 = 0.80 ms`，并在 0.88 m 虚拟窄道上验证了沿途包络净空对进场横移的价值。

### 插图与版式

采用 **2×2 实测证据网格**，直接嵌入仓库已生成的真实可视化与动图（每格配一行醒目英文量化结论）：
- **左上（显式悬挂对照）**：贴入 `experiments/grounded_caster_suspension_20261124/suspension_closeup_demo.gif`（单舵轮双万向球 6 mm 弹簧压缩特写）或 `baseline_vs_suspension_comparison.svg`。
- **右上（舵轮解耦与边走边变形）**：并排播放 `experiments/grounded_steering_decouple_20261124/lateral_steering_comparison.gif`（最直观：`±530°` → `±102.5°` 归零），配合无网格对比曲线 `steering_angles_before_after.svg`（或 `moving_steering_comparison.gif`、`steering_metrics_summary.svg`）。
- **左下（六向驱动与 4 mm 越障）**：贴入 `experiments/grounded_caster_suspension_20261124/swerve_drive_and_bump.gif` 关键帧 + 实测指标角标。
- **右下（局部多分支与沿途包络预检）**：贴入 `experiments/grounded_phi_local/tracking.png` 或 `experiments/grounded_support_navigation/tire_loads.png` + `14/14 branches passed` 与 `0.80 ms envelope` 徽章。

#### 可编辑 PPT 插图生成提示词（第 11 页 · 2×2 实测证据画框）

```text
请遵循【可编辑 PPT 插图通用视觉规范】，生成第 11 页（Recent Grounded Chassis & Substrate Validation）的原生可编辑 PPTX 与 SVG 矢量版式框架，尺寸 12.1 × 5.2 in，图内全部文字使用英文。

【布局结构：2×2 等大证据卡片网格（每卡宽 5.9 in，高 2.48 in，横纵间距 0.25 in），每卡分为“顶部标题+结果徽章条（0.45 in）” + “中部真实图像/GIF 贴图框（1.58 in）” + “底部一行量化实测数据（0.42 in）”】
- Card 1 (左上，绿标 "Done"): 标题 "1. Explicit 8-Caster Spring Suspension"
  * 贴图框提示："Insert suspension_closeup_demo.gif / baseline_vs_suspension_comparison.svg"
  * 底部实测："Coast 4-wheel support: 50.8% → 90.3%  |  Tire penetration: 3.24 → 1.71 mm  |  Housing scrape: 0 N"

- Card 2 (右上，绿标 "Done"): 标题 "2. Moving-Morph & RP¹ Steering Decoupling"
  * 贴图框提示："Insert lateral_steering_comparison.gif / steering_angles_before_after.svg"
  * 底部实测："Lat |q|max: 530° → 102.5° (0° return)  |  4-wheel: 100% (lat), 97.2–98.6% (mov/rev/half)  |  φ≈2 flip fixed"

- Card 3 (左下，绿标 "Done"): 标题 "3. 6-Dir Planar Drive & 4 mm Bump"
  * 贴图框提示："Insert swerve_drive_and_bump.gif keyframe"
  * 底部实测："6-dir 4-wheel support: 96.5–98.9%  |  Yaw drift ≤ 0.28°  |  4 mm bump recovery: 100%"

- Card 4 (右下，紫标 "Pre-Check"): 标题 "4. Local Branching & Encoder Envelope"
  * 贴图框提示："Insert grounded_phi_local / grounded_support_navigation figure"
  * 底部实测："φ≈1.434 rank 6→5: 14/14 branches pass (<3.95 N·m)  |  21-group envelope: 0.80 ms (p95)"

【质量约束】所有贴图区留白居中并带虚线框，方便一键替换为仓库已有 SVG/PNG/GIF；实测数字全部原生文本框可编辑。
```

### 中文讲稿

这一页总结近两天在新模型上已经完成并冻结定版的底层验证，证明开展脊髓三图探索和三层耦合闭环的基础移动与物理前提已经完全到位。

左上角是显式 8 万向球 6 毫米弹簧悬挂的特写与前后对照：相比旧刚性代理，滑行变形的四轮着地率从 50.8% 提到 90.3%，轮胎最大压入减半，全程零轮罩刮地。左下角是六向平面行驶和 4 毫米单侧越障测试，四轮承载率保持在 96.5% 到 98.9%，偏航漂移不到 0.3 度，越障后完全恢复四轮支撑。

右上角是边变形边移动与四坐标系舵轮解耦的实测对比：我们在脊髓层抽出了与根块无关的内部变形运动学接口，在小脑层结合安装块偏航前馈和射影圆侧滑能量梯度，彻底消除了旧版舵轮甩到 ±530 度再急转回弹，以及 φ=2 附近的 180 度突转问题。可以看到横移边变形中四轮全程稳定在 ±102.5 度以内并对称归零，四轮接地率达到 100%，5 种主动移动变形工况接地率全部在 97.2% 到 100%，67 项单元测试全部通过。右下角还做了两项前置核查：在 φ≈1.434 附近确实存在局部降秩且 14 个带符号有限分支全部在限幅内通过；基于编码器的 21 刚体包络单次计算仅 0.8 毫秒。这套底座现在已经具备作为理想解析“上界参考”的能力，下一步即可在工作站用脊髓三图记忆逐步替代解析接口做对照验证。

## 12. 工作站核心实验结果展位：脊髓三图与闭环对照（Exp A–C / D） · 1:35

> **使用说明（演讲前替换）**：本页专门预留给你推送最新底盘代码、在 Linux 工作站拉取并跑完 **实验 A / B / C（及有余力时跑完实验 D）** 后的实测结果展示。跑完哪几项就替换对应面板的真实无网格图表与 `[TBD]` 数字；若汇报前仅完成部分实验（例如跑完 Exp 0 + A/B，Exp C/D 仍在进行），保留对应面板的 *In progress on Linux workstation* 标签并按下方降级口径口播即可，严禁填未跑出的假数。

### English slide text

**Workstation validation of the three-graph spinal layer (Exp A–C / D)**

- **Exp A (`G_state` local prediction):** held-out joint error across probe budgets — `independent` `[TBD°]` vs. `dense` `[TBD°]` vs. `sparse validated graph` `[TBD°]`; closure residual `< [TBD] mm`.
- **Exp B (`G_mode` candidate groups):** multi-direction repeated inference reduces held-out group break rate from `[TBD%]` (single-dir) to `[TBD%]`; `[TBD]` edges kept `undecided`.
- **Exp C (`G_phase` transition retest):** controlled probe across condition grid + retreat-and-retest achieves `[TBD%]` repeatability and `[TBD%]` false unlock rate *(or reports negative evidence if no repeatable transition emerges)*.
- **Exp D (Closed-loop narrow corridor, if completed):** `ref-φ` vs. `global model` vs. `three-graph FSM + swerve` — success `[TBD]`, min clearance `[TBD] mm`, 4-wheel support `[TBD%]`.

### 中文对照

本页用于集中汇报即将于 Linux 工作站运行的三图核心对照实验结果（对应第 9 页验证设计）：左栏展示实验 A 的预算-留出误差曲线与三模型误差对比；中栏展示实验 B 候选功能组的留出破裂率对比与 `undecided` 边界；右栏展示实验 C 条件对照探测与回退复测复现率（及已验证 `G_phase` 边或如实负结果）；底部横条呈现实验 D 物理窄道闭环三方法对照汇总（若汇报前尚未完成则标注“待 A–C 锁定后执行”）。

### 插图与版式

采用 **上方三列核心结果图卡（Exp A / Exp B / Exp C）+ 底部实验 D 闭环对照汇总条**。所有待嵌入曲线图均由 Linux 工作站按**无横纵背景网格线**规范自动生成后直接拖入占位框。

#### 可编辑 PPT 插图生成提示词（第 12 页 · 实验结果展位模板）

```text
请遵循【可编辑 PPT 插图通用视觉规范】，生成第 12 页（Workstation Three-Graph Experiment Results Showcase Template）的原生可编辑 PPTX 与 SVG 矢量版式模板，尺寸 12.1 × 5.2 in，图内全部文字使用英文，严禁伪造任何数据曲线或假数字。

【布局结构：上方三张等宽结果展位卡（每卡宽 3.83 in，高 3.85 in，间距 0.3 in）+ 底部 Exp D 状态/汇总横条（宽 12.1 in，高 1.05 in）】
1. Col 1（蓝顶条，Tag: "Exp A · G_state"）：
   - 标题："Held-Out Prediction vs. Budget"
   - 图表占位框（高 2.25 in，浅灰虚线框）："Drop Exp A Budget–Error Plot Here (no gridlines)"
   - 底部 3 行 `[TBD]` 指标槽："Independent: [TBD] deg" / "Dense linear: [TBD] deg" / "Validated graph: [TBD] deg · gap <[TBD] mm"

2. Col 2（紫顶条，Tag: "Exp B · G_mode"）：
   - 标题："Candidate Groups & Break Rate"
   - 图表占位框（高 2.25 in，浅灰虚线框）："Drop Exp B Group Break-Rate Plot Here (no gridlines)"
   - 底部 3 行 `[TBD]` 指标槽："Single-dir break rate: [TBD]%" / "Multi-dir break rate: [TBD]%" / "Undecided edges: [TBD] / 8"

3. Col 3（琥珀顶条，Tag: "Exp C · G_phase"）：
   - 标题："Condition Grid & Retreat Retest"
   - 图表占位框（高 2.25 in，浅灰虚线框）："Drop Exp C Condition & Retest Plot Here (no gridlines)"
   - 底部 3 行 `[TBD]` 指标槽："Repeatability: [TBD]% (N=[TBD] cycles)" / "False unlock rate: [TBD]%" / "Verified edges stored: [TBD]"

4. 底部横条（绿边框浅绿底 `#ECFDF5`，左侧标签 "Exp D · Closed-Loop Corridor Benchmark"）：
   - 左侧 3 枚并列指标槽："[Ref-φ Replay: TBD]" | "[Global Model: TBD]" | "[Three-Graph Closed Loop: TBD]"
   - 右上角可切换小徽章（灰底虚线框，便于按实际进度保留或删除）："Status: Completed / Queued on Linux Workstation"

【质量约束】所有 `[TBD]` 文本框独立可编辑，方便工作站跑完后 1 分钟内填数贴图，绝不预绘假折线。
```

### 中文讲稿

*（正常版——工作站已跑完实验 A–C 或 A–D 时使用）*  
基础底盘锁版同步到工作站后，我们按第 9 页的对照方案跑了核心实验。左边是实验 A，在相同试探预算下，可以看到稀疏图响应相对独立模型和稠密模型在留出联合动作上的误差变化以及闭合残差情况；中间是实验 B，多方向重复扰动显著降低了候选功能组在留出动作上的破裂率，证据不足的连接如实保留为未决；右边是实验 C，在多条件对照和回退复测中，我们检验了受限方向响应是否可重复再现。底部给出了接入大脑 FSM 和小脑后的端到端窄道对照结果（若实验 D 尚未完成则改口播：“实验 D 的物理窄道闭环对照将在锁定 A–C 验证边后立即展开”）。

*（降级备用版——若汇报前仅刚推送底盘、工作站实验正在跑）*  
这一页预留给马上在 Linux 工作站拉取最新定版底盘后开展的三图对照实验：左边对应实验 A 的局部响应预算曲线，中间对应实验 B 的功能组留出破裂率，右边对应实验 C 的多条件回退复测，底部对应实验 D 的窄道完整闭环对照。所有脚本、公平预算协议和无网格出图规范已整理完毕，底盘悬挂与舵轮解耦算法已完成冻结定版，工作站拉取即可一键跑出这四组对照。

---

# 六、下一步计划

## 13. 下一步计划：分层验证与真机迁移 · 1:05

### English slide text

**Build evidence in the order required by the control loop**

1. **Freeze chassis & baseline** *(wrapping up)*  
   Finish `φ=2` steering polish, push to Git, and lock canonical states & logs on the Linux workstation.
2. **Complete spinal-graph benchmarks (Exp A–C)**  
   Quantify local prediction, candidate rigid groups, and retreat-retested transition edges.
3. **Connect Brain FSM & Cerebellum (Exp D)**  
   Build the physical narrow-corridor benchmark on `tars_mjcf_grounded` and run the 3-way closed-loop comparison.
4. **Hardware transfer preparation**  
   Align CAN dual-encoder mapping, body geometry envelope, real caster suspension (`k, F₀, c`), and hub-motor sizing check.

### 中文对照

按控制闭环依赖顺序推进：先收尾基础移动并同步工作站冻结基线，随后完成脊髓三图 A–C 定量对照，再在 `tars_mjcf_grounded` 上搭建物理窄道场景完成大脑—脊髓—小脑三方法闭环对照（实验 D），最后对齐真机 CAN 双编码器、外轮廓、悬挂参数并复核轮毂电机选型。

### 插图与版式

用四段递进横向路线图，**不标“第几天”或固定耗时**。前 3 步收在浅蓝底分组框 *Simulation Validation (MuJoCo)* 内，第 1 步高亮 *Wrapping Up*，第 2–3 步标 *Next on Workstation*；第 4 步单独放右侧虚线分组框 *Hardware Transfer (Future)*，视觉区隔仿真与硬件，避免暗示已部署。

#### 可编辑 PPT 插图生成提示词（第 13 页）

```text
请遵循【可编辑 PPT 插图通用视觉规范】，生成第 13 页（Next Steps Roadmap）的原生可编辑 PPTX 与 SVG 矢量路线图，尺寸 12.1 × 4.6 in，图内全部文字使用英文，严禁出现具体日期、天数或周次。

【布局结构：左右两大分区框（左：Simulation Validation 宽 8.9 in；右：Hardware Transfer 宽 2.9 in，中间留 0.3 in 虚线过渡箭头标 "Sim-to-Real Gate"）】
1. 左侧大框（浅蓝淡底，顶栏 "Simulation Validation (Dependency Order)"），内含 3 张横向递进步骤卡（每张宽 2.65 in，高 3.1 in，箭头串联）：
   - Card 1（蓝实线高亮，顶部徽章 "Wrapping Up"）："1. Freeze Chassis & Baseline" — "Finish φ=2 steering polish · Git sync to Linux · lock canonical states & logs (Exp 0)"
   - Card 2（紫虚线框，徽章 "Next"）："2. Spinal Graph Benchmarks" — "Run fair Exp A–C: local prediction, G_mode break rate & G_phase retreat-retest"
   - Card 3（绿虚线框，徽章 "Planned"）："3. Full Closed-Loop Eval" — "Physical narrow corridor on tars_mjcf_grounded · 3-way FSM + Spinal + Swerve test (Exp D)"

2. 右侧独立虚线框（浅灰底，顶栏 "Hardware Transfer"），内含 1 张卡片：
   - Card 4（深灰虚线框，徽章 "Future"）："4. Real-Robot Alignment" — "CAN dual-encoder mapping · geometry envelope · caster spring & hub-motor load check"

【质量约束】4 张卡片横向中心对齐，仿真区与硬件区视觉界限清晰，字号 ≥ 11 pt，全部元素原生可编辑。
```

### 中文讲稿

下一步按依赖顺序收尾：第一步完成 `φ=2` 转向摆幅最后微调后推送到 Git，在工作站拉取并冻结实验 0 基线；第二步跑完或补齐脊髓三图实验 A 到 C 的定量对照；第三步在最新模型上搭建物理窄道场景，跑通大脑 FSM、脊髓三图和小脑舵轮的三方法端到端对照；最后面向真机对齐 CAN 双编码器、包络、悬挂实测参数以及前面发现的轮毂力矩余量问题。

## 14. 讨论与致谢 · 0:30

### English slide text

**Discussion**

1. What evidence should qualify a coupling or transition as verified?
2. Which spinal outputs are essential for cerebellum and FSM?
3. Given the `0.45 N·m` hub cap at `9.26 kg`, what speed and benchmark task should we prioritize first?

**Thank you**

### 中文对照

什么证据足以把耦合或阶段转换称为已验证？脊髓必须向小脑和 FSM 返回哪些信息？结合 9.26 kg 下轮毂 `0.45 N·m` 的速度边界，第一项完整闭环基准任务应优先设定什么速度与场景？谢谢。

### 插图与版式

沿用模板致谢页：左侧放 3 个讨论问题与 *Thank you*，右侧直接缩小复用第 5 页可编辑矢量框图的精简版（保留 Brain–Cerebellum–Spinal 与三图徽章，隐去次要标注），留出充足留白与讨论空间；本页无需单独再生成新图。

### 中文讲稿

最后希望和大家讨论三个问题：第一，耦合和阶段转换需要什么样的复测证据才算可靠；第二，小脑和 FSM 最需要脊髓返回哪些关键信息；第三，结合完整模型 9.26 公斤下轮毂力矩的实际边界，第一项端到端闭环基准任务最适合设在什么速度和场景下。谢谢大家！

---

# 演讲外的实验执行备忘

本节用于后续在 Linux 工作站（`/home/user/桌面/robot-sim`）开展仿真实施与验收，**不作为 PPT 页面，也不描述为已经完成的实验**。具体算法保持开放，优先用可解释基线建立公平对照。每项实验均提供可直接复制给 Codex CLI 或其他编码 Agent 的提示词模板；每次运行前请先确认 Linux 端已同步最新代码与 `tars_mjcf_grounded` 模型，若模型或接口已有变更，以仓库最新定义为准。

## 实验总览与当前 MuJoCo 仿真就绪度

> **当前主线模型提醒**：仓库中并存三套历史/现行模型。后续实验统一基于最新 CAD 落地闭链模型 `data/my_robot/mujoco/tars_mjcf_grounded/scene.xml`（引用同目录 `robot.xml`，对应 `src/envs/tars_grounded.py` 的 `TarsGroundedEnv`），**不要误用**旧 7 关节开链模型 `data/my_robot/mujoco/robot.xml` 或旧推算闭链替身 `data/my_robot/mujoco/robot_closed_loop.xml`。

| 编号 | 实验名称 | 对应架构层级 / 图 | 当前仿真前提就绪度 | 核心产出 |
|---|---|---|---|---|
| **实验 0** | 模型冻结、基础移动框架与标准初态验收 | 仿真底座 + 小脑底层 | **已定版就绪**：8 轴闭链、4 舵轮、8 万向球 6 mm 弹簧悬挂与 `RP¹` 舵轮解耦控制器已全部定版并通过 67 项单元测试；在工作站同步代码与私有 MJCF XML 后即可运行验收并导出标准初态快照与 8-block 映射 | 冻结版模型哈希、基础移动与边走边变形回归报告、`φ=0/1.5/3` 初态快照、8-block 刚体映射表 |
| **实验 A** | 八轴局部响应辨识与留出预测 | 脊髓 `G_state` | **可直接展开**（需先适配 `tars_mjcf_grounded` 试探脚本，注意 `q=0` 对应 `φ=1.5` 且默认 `kp=50` 位置伺服） | 局部响应模型对照结果（对角 / 稠密 / 稀疏图）、留出误差与预算曲线、原始 trial 数据 |
| **实验 B** | 有效拓扑与候选功能组辨识 | 脊髓 `G_mode` | **可直接展开**（需先建立 CAD 刚体名到 8 个逻辑 block 的映射） | 静置底噪阈值、候选功能组与留出破裂率报告、`undecided` 关系清单 |
| **实验 C** | 阶段转换条件与回退复测 | 脊髓 `G_phase` | **可直接展开**（当前模型已在 `φ≈1.434` 附近观察到局部降秩与多分支有限位移，待用反馈与复测独立验证） | 可区分条件对照结果、回退复测复现率、带证据的 `G_phase` 候选/已验证边 |
| **实验 D** | 图查询接入大脑 FSM 与小脑协调的端到端闭环对照 | 大脑 + 脊髓 + 小脑完整闭环 | **存在工程缺口（需先搭桥）**：现有 FSM/MPC/障碍场景仍绑定旧 7 关节模型，`tars_mjcf_grounded` 目前仅有空旷平地与离线窄道投影，需先建物理窄道场景与统一闭环接口 | 物理窄道/任务基准场景、三方法公平对照报告（参考路径 / 全局模型 / 三图查询） |
| **实验 H** | 真机接口对齐与硬件迁移验证 | 仿真 → 真机 | **待真机联调** | CAN 双编码器映射表、静态包络误差、低速形变与悬挂轮载标定记录 |

## 通用运行要求（建议粘贴在每条 Agent Prompt 开头）

```text
【通用执行约束｜每次运行前务必遵守】
1. 仓库与目标模型：
   - 仅使用最新 CAD 落地闭链 8 电机 + 4 舵轮 + 8 万向球弹簧悬挂模型：
     `data/my_robot/mujoco/tars_mjcf_grounded/scene.xml`（含同目录 `robot.xml`）
     及封装环境 `src/envs/tars_grounded.py`（`TarsGroundedEnv`）。
   - 严禁误用旧 7 关节模型 `data/my_robot/mujoco/robot.xml` 或旧推算闭链替身 `data/my_robot/mujoco/robot_closed_loop.xml`。
2. 不可篡改物理前提：
   - 不得为了实验好看擅自修改 XML 闭环拓扑、weld 约束参数、碰撞排除掩码、执行器符号、关节限位、摩擦或悬挂参数。
   - 如发现模型或控制器确有缺陷必须改动，应另存新版本文件并记录完整 diff 与前后对照，不得静默覆盖。
3. 审计与可复现规范：
   - 运行前先打印并记录：git commit（及脏文件清单）、`scene.xml` 与 `robot.xml` 的 SHA-256、Python/MuJoCo 版本、随机种子、物理步长、控制步长、8 轴编号与映射、闭环 weld 约束定义及复位方式。
   - 所有输出写入 `experiments/<实验名>_YYYYMMDD_HHMMSS/` 新目录，保留配置 `config.json`、原始轨迹/试次 `.npz`、汇总指标 `metrics.json`、复现脚本和 `summary.md`，不覆盖或删除既有实验目录。
4. 单位与绘图规范：
   - 所有数据、表格、JSON 字段名与图表必须明确单位（`rad`/`deg`、`m`/`mm`、`N`、`N·m`、`s`、`mm/s` 等）。
   - 绘制曲线图、折线图时**关闭横向和纵向背景网格（包括主、次刻度网格）**，仅保留坐标轴和刻度。
5. 诚实汇报边界：
   - 严格按实测数据报告结果；若出现负结果、数值警告、失载、接触异常或某接口尚未打通，如实记录并收窄结论范围，禁止编造数据或夸大泛化结论。
```

---

## 实验 0 · 模型冻结、基础移动框架与标准初态验收（前置必做）

**目的**　在展开脊髓三图实验前，冻结当前 `tars_mjcf_grounded` 模型与底盘控制版本，验收“基础移动 + 边走边变形 + 闭链稳定”前提是否达标，并导出后续实验 A/B/C/D 共用的标准初态快照与刚体映射表。  
**关键检查**　1) 静置平衡与闭合残差；2) 六向平面行驶（`x± / y± / yaw±`）与 4 mm 单侧凸起越障；3) 边走边变形（`still / moving / lateral / reverse / half`）及舵轮转向控制定版核查；4) 导出 `φ=1.5`（CAD 零位）、`φ=0`、`φ=3` 的闭链可达稳定初态与 8-block 映射表。  
**通过门禁**　无 MuJoCo 数值警告；闭链 weld 间隙 `< 0.2 mm`、相对转角 `< 0.2°`；六向行驶无轮罩/机身刮地、四轮承载率 `≥ 90%`；所有移动变形工况中无持续失载或转向失控。

### 可直接复制的 Agent Prompt（实验 0）

```text
请先遵循【通用执行约束】，在 robot-sim 仓库中完成【实验 0：模型冻结、基础移动框架与标准初态验收】。

一、任务目标
验证当前最新闭链模型 `data/my_robot/mujoco/tars_mjcf_grounded/scene.xml` 与 `TarsGroundedEnv` / `src/controllers/grounded_morph.py` 是否满足展开后续三图与闭环实验的“基础移动框架”前提，并生成冻结基线包供实验 A/B/C/D 复用。

二、必须执行的检查步骤
1. 模型与接口核对：
   - 读取 `data/my_robot/mujoco/tars_mjcf_grounded/scene.xml` 与 `robot.xml`，核对并导出：
     * 总质量、积分器、timestep、control_steps；
     * 8 个闭链形变关节（`dof_1..dof_8`）的 `qposadr`、`dofadr`、限位、正方向、默认执行器（`kp/kv/forcerange`）；
     * 4 个被动 pivot（`dof_9, 12, 15, 18`）、4 个舵轮转向（`dof_10, 13, 16, 19`）、4 个轮毂驱动（`dof_11, 14, 17, 20`）及 8 个万向球滑动悬挂（`caster_w{1..4}{a,b}_slide`）的参数；
     * 闭环 `weld` 约束的两个 site 及其在 XML 零位（`q=0`，即 CAD 装配态 `φ=1.5`）下的位置/姿态残差；
     * 明确建立“8 个逻辑 block（刚体组）”到 MuJoCo body 名称/ID 及物理环状相邻边（含闭环 weld 边）的映射文件 `block_topology.json`。
2. 舵轮控制器版本确认：
   - 检查 `src/controllers/grounded_morph.py` 与 `experiments/grounded_steering_decouple_20261124/` 的状态，确认当前主代码中 `_steer_and_roll` 的精确行为；不要擅自静默替换主控制器，若要比较新旧转向策略请以对照项记录，明确指出实验 A/B/C/D 应锁定的控制器版本。
3. 基础移动框架验收（全部独立复位运行）：
   - 静置稳定性测试（2 s）：记录 8 轴漂移、质心漂移、四胎与八球法向载荷、悬挂行程、weld 间隙与角度残差。
   - 六向平面移动（`TarsGroundedEnv.drive`：`x+, x-, y+, y-, yaw+, yaw-`）与 4 mm 单侧凸起越障：记录实际位移/转角误差、偏航漂移、四轮承载率、轮胎载荷占比、最大悬挂行程、轮罩/机身非法触地力。
   - 边走边变形（`TarsGroundedEnv.drive_morph`：`still / moving / lateral / reverse / half`）：记录上体与整机质心位置误差（peak/rms/end, mm）、四轮承载率、最长单轮失载时长（ms）、悬挂触底比例、最大倾角（deg）、单轮累计转向角与峰值转向速率（deg, deg/s）、8 轴峰值力矩（N·m）与闭链残差（mm, deg）。
4. 导出标准初态快照（供实验 A/B/C/D 严格复用）：
   - 分别稳定停驻在 `φ=1.5`（CAD 零位）、`φ=0`（从 `φ=1.5` 平滑闭链引导并静置稳定）、`φ=3` 及候选低秩点附近（如 `φ≈1.434`），保存完整 `qpos/qvel/ctrl` 与稳态残差到 `canonical_states.npz` 和 `canonical_states.json`。

三、输出文件（写入 `experiments/exp0_baseline_YYYYMMDD_HHMMSS/`）
- `config.json`（含模型哈希、git 状态与全部参数）
- `block_topology.json`（8 block 刚体映射与相邻边）
- `canonical_states.npz` 与 `canonical_states.json`
- `metrics.json`（上述全部定量验收指标与门禁通过/未通过判定）
- `summary.md`（清晰列出已通过项、未通过项或隐患，以及能否直接进入实验 A/B/C/D 的明确结论）
```

---

## 实验 A · `G_state` 八轴局部响应辨识与留出预测

**目的**　检验能否仅通过有界小幅试探与八轴实测反馈，学到当前姿态邻域内的局部耦合响应关系，并准确预测未参与拟合的新联合关节动作。  
**前置条件**　完成实验 0；使用实验 0 导出的 `canonical_states.npz`，避免误把 XML 零位（`φ=1.5`）当成 `φ=0`。  
**关键对照**　在**完全相同的数据预算、相同初态、相同留出联合动作**下比较：1) 对角独立关节模型；2) 稠密线性响应模型；3) 仅保留经重复试探确认稳定边的稀疏图结构响应模型（附“零位移”与“直接等幅执行”参考下界）。同时做控制模式对照：全轴 `kp=50` 位置保持 vs 其余非激励轴低阻抗/零刚度随动，避免把位置保持误判为机械不可动。  
**核心指标**　逐轴及总体留出角度预测误差（RMSE / MAE / Max，单位 `deg` 与 `rad`）、预测动作执行后的闭合残差（`mm`, `deg`）、不同试探预算（如 8 / 16 / 24 / 32 / 48 次）下的误差收敛曲线、重复试次波动（std）与有效响应秩/奇异值谱。

### 可直接复制的 Agent Prompt（实验 A）

```text
请先遵循【通用执行约束】，在 robot-sim 仓库中设计与运行【实验 A：G_state 八轴局部响应辨识与留出预测】。

一、实验对象与初态要求
- 目标模型：`data/my_robot/mujoco/tars_mjcf_grounded/scene.xml`（`TarsGroundedEnv`）。
- 注意：不要直接复用旧 `src/exploration/closed_loop_probe.py`（它绑定了旧 `robot_closed_loop.xml`）；请基于 `TarsGroundedEnv` 编写干净独立的新试探模块（保留原文件不破坏旧测试）。
- 分别在两个标准闭链可达初态邻域运行本实验（若已有实验 0 导出的 `canonical_states.npz` 则直接加载，否则先按闭链路径平滑到达并稳定至少 1 s）：
  1) `phi_0`（`φ=0` 展开态）；
  2) `phi_1p5`（`φ=1.5` CAD 零位态，便于与后续奇异/分叉实验衔接）。
- 在每个初态先记录 1 s 静置漂移（8 轴角度变化、速度、weld 位置/角度残差、四轮载荷）。

二、执行器与控制模式对照（关键防误判）
- 核对 `dof_1..dof_8` 的 `kp=50, kv=0, ±11 N·m` 位置伺服定义。
- 明确设置并记录两种受控试探模式：
  * 模式 M1（默认全轴位置保持）：8 个电机均处于统一限幅 PD/位置伺服，只叠加有界增量指令 `u`；
  * 模式 M2（对照随动模式，用于诊断）：除当前试探激励轴外，其余轴改为极低刚度/零指令增量随动（仅记录为机理对照，不篡改原 XML，在内存模型副本中执行）。
- 说明两种模式下测得响应矩阵和奇异值谱的差异，防止将“其余轴位置保持顶住”误判为“机械不可动”。

三、试探协议与防数据泄漏要求
- 设计限幅、平滑（斜坡上升 + 保持段）的小幅正负试探：覆盖单轴正负试探与多轴联合试探，指令范数严格受限（如 `||u||_2 <= 0.04 rad`，并核查力矩不触限、weld 残差受控、无数值警告）。
- 每次试探均从同一稳定初态出发并复位（记录零输入对照漂移并扣除同起点可重复漂移 `y = Δq(u) - Δq(0)`）；每个试探独立记录完整时序：`qpos`、`qvel`、命令 `u`、实测 `actuator_force`、weld 位置与姿态残差、轮胎与万向球接触载荷及告警标志。
- 严格按**完整 trial** 划分训练集与留出测试集（例如固定 3 个随机种子，留出 ≥16 条未见多轴联合输入 trial，并含重复试次以量化重复性波动），严禁将同一 trial 的相邻帧分入训练和测试集。
- 辨识器只能读取指令 `u`、八轴 `qpos/qvel` 与物理连接拓扑，严禁读取闭环雅可比或 weld 约束力作为输入特征（雅可比仅用于事后离线对照诊断）。

四、模型对照（同初态、同预算、同留出集）
1. `independent_diag`：对角独立关节响应模型；
2. `dense_linear`：带正则化的稠密响应矩阵（如 Ridge）；
3. `sparse_graph_validated`：仅保留经重复试探一致性检验（方向/比例稳定且超过静置底噪）的耦合边构成的图结构局部响应模型；
4. 附参考下界：`zero_pred`（预测零运动）。
- 评估不同训练试探预算（如 `N = 8, 12, 16, 24, 32, 48`）在相同留出集上的误差。

五、输出文件（写入 `experiments/expA_local_response_YYYYMMDD_HHMMSS/`）
- `config.json`、`raw_trials.npz`、`metrics.json`（含逐轴与总体 RMSE/MAE/Max（deg & rad）、奇异值谱、重复性 std、weld 残差、触限次数、各预算下对比表）
- 无背景网格线（关闭主/次横纵网格）的对比图：1) 预算-留出误差曲线；2) 三模型逐轴误差柱状/散点对比；3) 学得响应矩阵热力图与稀疏图结构。
- `summary.md`：明确声明所有结论仅适用于所测初态局部邻域，不外推为全局机械约束。
```

---

## 实验 B · `G_mode` 候选功能组（有效刚体拓扑）辨识与留出检验

**目的**　在给定 8 个 block 的物理连接基础上，仅从多方向局部扰动下的刚体位姿反馈中识别哪些 block 保持相对静止（形成候选功能组），并在未参与构图的留出动作上检验分组是否破裂。  
**前置条件**　完成实验 0（取得 `block_topology.json`）并可复用实验 A 的受控扰动 trial 及新增多方向留出动作。  
**关键对照**　1) 仅基于单方向共动推断的候选分组 vs 2) 基于多方向重复扰动联合推断的候选分组；在相同留出动作集上统计组内相对运动与破裂率。  
**核心指标**　相邻 block 相对平移（`mm`）与相对转角（`deg`）分布、由静置重放确定的噪声阈值、留出动作上的**分组破裂率（%）与破裂次数**、被标记为 `undecided`（证据不足）的连接比例、分组有效姿态范围。

### 可直接复制的 Agent Prompt（实验 B）

```text
请先遵循【通用执行约束】，在 robot-sim 仓库中设计与运行【实验 B：G_mode 候选功能组辨识与留出检验】。

一、物理拓扑与刚体位姿读取
- 目标模型：`data/my_robot/mujoco/tars_mjcf_grounded/scene.xml`（`TarsGroundedEnv`）。
- 注意：新模型中 8 个逻辑 block 的 MuJoCo 刚体名不是 `block1..block8`，而是 CAD 导出的 `link_insert_cd6ae4a9bfcf_part_*` 刚体树。请读取实验 0 生成的 `block_topology.json`（若无则先从 `robot.xml` 准确解析 8 个逻辑 block 的代表刚体与 8 条环状物理相邻边，含闭环 `weld` 边，并保存映射表）。
- 对每条相邻连接边 `(b_i, b_j)`，从 MuJoCo 刚体世界位姿 `(xpos, xmat)` 计算相对位姿变化：
  * 相对平移变化 `Δp_rel`（mm，在 `b_i` 局部系下相对于初态的偏移范数）；
  * 相对转角变化 `Δθ_rel`（deg，由相对旋转矩阵 `R_rel(0)^T R_rel(t)` 的旋转角得到）。

二、阈值标定（禁止拍脑袋定阈值）
- 先在每个测试初态运行静置重放（至少 2 s，多个随机扰动/复位重复），统计每条相邻边在静置状态下的 `Δp_rel` 与 `Δθ_rel` 数值噪声上界（如 99% 分位或 max），结合安全系数确定统计阈值 `τ_p`（mm）与 `τ_θ`（deg），并明确写入报告。
- 不得以“两个 block 在世界坐标系下同时移动”作为合并依据，必须严格按刚体间相对位姿是否保持不变来判定。

三、构图与留出验证对照
- 在选定初态（至少覆盖 `φ=0` 与 `φ=1.5`，以及条件允许时的移动中局部扰动）执行多方向有界扰动试探，并按完整 trial 划分为“构图集”与“留出检验集”（留出集应包含未参与构图的组合动作及跨方向动作）：
  1) 基线方法 `single_dir_inference`：仅使用单一方向（或单一已知路径切向）的扰动样本推断功能组；
  2) 对照方法 `multi_dir_repeated`：要求在多个独立扰动方向及重复试次下相对运动均持续低于阈值，且至少在有激励覆盖的前提下，才合并为候选刚性功能组；激励不足或结果不一致的边严格保留为 `undecided`，不强行合并或拆分。
- 在完全相同的留出动作集上，统计两种方法得到的候选功能组内部所有连接边的相对位移/转角分布、超阈值次数与**留出破裂率**。

四、输出文件（写入 `experiments/expB_gmode_groups_YYYYMMDD_HHMMSS/`）
- `config.json`、`block_topology.json`、`raw_relative_motion.npz`
- `g_mode_candidates.json`（每个候选分组的成员 block、样本数、相对平移/转角分位数、静置阈值依据、留出破裂次数/破裂率、标记为 `undecided` 的边、适用姿态范围）
- 无背景网格线的对比图：1) 各相邻边在静置 vs 单向 vs 多向留出下的相对转角/位移分布；2) 两种方法的留出破裂率对比；3) 8-block 环状拓扑与候选功能组示意图。
- `summary.md`：明确区分“模型给定物理连接”与“反馈支持的候选功能组”，说明证据不足边界。
```

---

## 实验 C · `G_phase` 阶段转换条件与回退复测

**目的**　检验是否存在独立的推进方向/状态条件，使得进入特定状态邻域后，某个原先受限的运动方向响应显著改变；并通过真实回退再进入复测，区分偶然数值现象与可复现的阶段转换边。  
**前置条件**　完成实验 0、实验 A、实验 B；不预设“某组关节共同到 0°”等人工标签或解锁公式。  
**关键对照**　先基于反馈检验是否存在两个可分别推进且可回退的运动方向（或基于实测有效子空间/候选降秩点如 `φ≈1.434` 邻域定义状态条件）。若存在 A/B 两组独立进度，则严格在“**都远 / 仅 A 近 / 仅 B 近 / 两组都近**”四种闭链可达状态下，对同一受限方向施加完全相同的安全探测；若实测组数或条件结构不同，按实测关系重新设计最小可区分条件对照并记录原因。  
**核心指标**　目标转换复现率（多次进入-探测-回退-再进入）、单条件误解锁率（仅 A 近或仅 B 近时的假阳性率）、目标方向实测响应幅度与信噪比、闭合残差（`mm`, `deg`）、执行器峰值力矩（`N·m`）、四轮支撑状态、回退成功率与转换生成机制标签（如 `geometric_branch` / `limit_switch` / `active_contact_switch`）。

### 可直接复制的 Agent Prompt（实验 C）

```text
请先遵循【通用执行约束】，在 robot-sim 仓库中设计与运行【实验 C：G_phase 阶段转换条件与回退复测】。

一、核心原则（严格防先验泄漏）
- 目标模型：`data/my_robot/mujoco/tars_mjcf_grounded/scene.xml`（`TarsGroundedEnv`）。
- 严禁预设“固定关节配对”或“预设解锁角度”作为探索策略输入；严禁在控制/探索器中调用闭环雅可比强制修正动作（雅可比奇异值仅可在实验结束后的离线诊断章节里作为物理对照，且必须明确标注为“离线参考而非在线观测”）。
- 严禁通过添加人工角度锁定公式或修改 XML 来强行制造阶段切换。

二、实验流程
1. 独立推进与可回退方向检验：
   - 结合实验 A/B 测得的局部有效响应方向，先用留出执行测试验证在当前模型中：
     a) 从 `φ=0` 或 `φ=1.5` 邻域出发，是否存在两个能够分别推进且可稳定回退的独立运动方向；
     b) 同时检查已知候选奇异/分叉邻域（此前离线扫描提示 `φ≈1.434` 附近存在秩降与横向分支，如 `dof_1/-dof_5`、`dof_3/-dof_7` 等），检验纯反馈探测能否在进入该邻域前后检测到受限方向响应的显著变化。
   - 按实测结果定义可区分的状态条件：
     * 若实测存在两个独立进度组 A/B：定义各自的进度坐标及近/远区间，构造四格对照条件——`both_far`（都远）、`only_A_near`（仅 A 近）、`only_B_near`（仅 B 近）、`both_near`（两组都近）；
     * 若当前 CAD 模型的实际自由度/组数与双组假设不同：如实记录差异，并按实测方向构造最小可区分的“远离条件 vs 邻域条件 vs 单侧偏离条件”对照矩阵，不硬凑双组标签。
2. 同输入受限方向探测：
   - 在每种对照条件下，从闭链可达且静置稳定的状态出发，对同一个受限/目标方向施加**完全相同幅度、相同斜坡与保持时长**的安全探测输入（执行器力矩、速度、weld 残差均设安全止停阈）。
   - 记录 8 轴 `qpos/qvel` 响应、目标方向投影位移、其余方向泄漏位移、weld 位置与姿态残差、执行器峰值力矩、四轮法向载荷及悬挂行程。
3. 回退与重复进入复测（区分验证边与偶然现象）：
   - 对每个出现显著响应变化的候选转换，执行至少 5 次“进入条件 → 探测 → 回退至源状态 → 再次进入 → 复测”循环。
   - **优先测试真实物理回退**（通过控制指令实际退回源状态）；若同时使用仿真快照重置作辅助对照，必须把两种协议的结果和耗时/失败率分开报告，不得混为一谈。
4. `G_phase` 边准入规则与机制标注：
   - 仅当目标方向响应在重复进入后可稳定复现、误解锁率受控且闭合残差与支撑安全时，才将其写入 `g_phase.json` 并标记 `status: "verified"`，附带：`guard`（进入条件与适用范围）、`action`（已验证动作与速度/幅值上限）、`exit_test`（退出判定）、`failure_evidence`（失败或边界记录）及转换机制标签（`geometric_branch` / `limit_switch` / `active_contact_switch` / `unknown`）。
   - 若当前模型在所测轨迹中不存在可复现阶段转换，如实报告全部证据，输出空 `verified` 边表，严禁构造虚假阶段边。

三、输出文件（写入 `experiments/expC_gphase_transitions_YYYYMMDD_HHMMSS/`）
- `config.json`、`raw_probe_cycles.npz`、`g_phase.json`、`metrics.json`
- 无背景网格线的对比图：1) 四格/多条件下的同一探测响应对比（含多次复测散点与误差带）；2) 闭合残差与峰值力矩安全检查图；3) 复现率与误解锁率汇总图。
- `summary.md`：清晰区分“在线反馈发现的证据”与“事后离线雅可比参考”，如实交代当前 CAD 模型支持与不支持的假设。
```

---

## 实验 D · 图查询接入大脑 FSM 与小脑舵轮协调的端到端闭环对照

**目的**　检验脊髓三图（`G_mode / G_state / G_phase`）提供的可达性、验证动作边、沿途包络与支撑条件，是否能真正改善大脑（FSM）的阶段选择和小脑（舵轮 + 变形）的移动协调，而不仅仅提高单步角度预测精度。  
**前置条件与当前缺口说明**　1) 需先完成实验 0（锁定稳定底盘移动与边走边变形控制器）及实验 A/B/C（获得经过复测标记的图数据）；2) **当前工程缺口**：仓库现有 `MujocoBenchmarkEnv`（A/B/C 场景）、`ScriptedBrain` 与 `MPCController` 仍绑定在旧 7 关节模型上，`tars_mjcf_grounded` 目前仅有空旷平地与 `experiments/grounded_support_navigation` 的离线虚拟墙投影。因此本实验分为两步：先搭建最小可复现的物理窄道任务场景与三层闭环接口，再执行三方法严格对照。  
**关键对照**　保持**同一任务场景、相同起终点、相同模型版本、相同底盘舵轮参数、相同安全限幅与相同探索数据预算**，比较三种控制器：  
1. **基线 1（Reference Replay）**：既有参考 `φ` 路径回放控制，仅按起终点形态规则切换，不读取三图（不查沿途包络与验证阶段边）；  
2. **基线 2（Global Response）**：使用与图方法相同探索数据预算拟合的固定全局响应模型驱动变形与协调；  
3. **本方法（Three-Graph Closed-Loop）**：大脑 FSM 查询 `G_mode / G_state / G_phase`（**仅允许调用实验 C 中 `verified` 边**），结合 `src/controllers/shape_navigation.py` 计算沿途包络净空并接收小脑支撑/跟踪/碰撞反馈，支持进场横移、减速、暂停或重规划。  
**核心指标**　任务成功率（到达目标且无碰撞/无倾覆/无数值告警）、阶段转换成功率与误判率、沿途最小墙体净空与碰撞力（`mm`, `N`）、路径跟踪误差（`mm`）与形态跟踪误差（`deg`）、闭合残差（`mm`, `deg`）、四轮承载率与最长失载时长（`%`, `ms`）、任务耗时（`s`）与中止原因分类。

### 可直接复制的 Agent Prompt（实验 D）

```text
请先遵循【通用执行约束】，在 robot-sim 仓库中搭建并运行【实验 D：图查询接入大脑 FSM 与小脑舵轮协调的端到端闭环对照】。

一、前置检查与最小基准任务搭建（先搭桥，再做对照）
- 目标模型：`data/my_robot/mujoco/tars_mjcf_grounded/scene.xml`（`TarsGroundedEnv`、`src/controllers/grounded_morph.py`、`src/controllers/shape_navigation.py`）。
- 先核查是否存在已通过实验 0 的冻结底盘控制器，以及实验 A/B/C 产出的图文件（`block_topology.json`、`g_mode_candidates.json`、`g_phase.json`）。
  * 若某一层接口或经复测通过的 `G_phase` 边尚未完备，不得伪造验证边，请明确在 `summary.md` 中声明已运行的层级边界（例如先运行“已知验证阶段边 + 沿途包络净空与支撑闭环”层级）。
- 在不修改 `data/my_robot/mujoco/tars_mjcf_grounded/robot.xml` 的前提下，通过独立的场景 XML 或 `mujoco.MjSpec` 在平地上构建**带真实物理碰撞墙体的直窄道穿行基准任务**（例如多组通道宽度：`0.80 m / 0.88 m / 1.00 m` 及不同初始横向偏置/随机种子），同时复用 `ShapeGeometry` 从关节编码器实时计算沿途包络净空。
- 将三层接口打通到 `TarsGroundedEnv`：
  * 大脑（FSM）：发出目标位置、目标阶段与进场策略请求，接收“到达 / 暂停 / 净空不足 / 支撑异常 / 失败重规划”事件；
  * 脊髓：查询三图返回可达性、协调 8 轴参考轨迹、沿途包络边界与转换状态；
  * 小脑：通过 `TarsGroundedEnv.drive` 与 `drive_morph` 协调机身平移、朝向、变形进度、舵角与轮速，并实时回传跟踪误差、闭合残差、墙体接触力与四轮/八球支撑状态。

二、三方法公平对照协议（同任务、同初态、同底盘、同限幅、同探索预算）
在至少 3 个通道宽度 × 多个初始横向偏置（或随机种子）下，对比以下三种控制方式：
1. `baseline_ref_phi`：仅按起终点规则触发既有参考 `φ` 轨迹回放，不读取三图，不做沿途包络净空预检查与重规划；
2. `baseline_global_model`：使用与三图方法相同数量探索数据拟合的固定全局局部响应模型，无阶段图查询与条件门控；
3. `three_graph_closed_loop`：查询 `G_mode / G_state / G_phase`，**严格只允许调用已标记 `verified` 的动作边**（未验证或 `undecided` 关系禁止作为可执行策略），执行前由小脑结合沿途包络做净空预检（必要时先横移对齐再变形），执行中基于关节响应、闭合残差、墙体净空和四轮载荷触发减速、暂停或 FSM 重规划。

三、统一成功判定与记录字段
- 明确定义 `success = goal_reached AND no_wall_collision AND no_tip_over AND closure_ok AND no_sim_warning`（阈值写入 `config.json`，三方法完全一致）。
- 每条运行完整保存：起始条件、随机种子、方法名、目标、FSM 状态转换日志、8 轴 `qpos/qvel/ctrl/force`、闭合残差、舵角与轮速、整机与上体质心轨迹、墙体接触力与最小净空、四轮法向载荷与悬挂行程、完成时间与中止原因。

四、输出文件（写入 `experiments/expD_closed_loop_YYYYMMDD_HHMMSS/`）
- `config.json`、`episodes/*.npz`、`metrics.json`（含按场景分组的三方法对照表）
- 无背景网格线的对比图：1) 典型窄道场景下三方法的俯视轨迹与包络走廊图；2) 关键指标对照图（成功率、最小净空/碰撞率、闭合残差、四轮承载率、路径与形态跟踪误差）。
- `summary.md`：重点回答“脊髓三图相对两条基线在任务闭环上究竟带来了什么收益、在哪些工况失效、以及结论覆盖的实际层级”。
```

---

## 后续硬件接口验证（真机联调备忘）

**目的**　在仿真闭环稳定后，分步完成仿真接口到实物的传感、几何、动力学与安全联调，避免把仿真成功直接等同于真机可用。

### 可直接复制的 Agent Prompt（真机接口核对与离线对齐）

```text
请先遵循【通用执行约束】，编写或核对 TARS 仿真到真机的接口映射与标定检查脚本（仅做数据对齐与离线验证，不直接向实物发未经确认的高幅运动指令）。

一、必须核对的五项接口对齐
1. 关节与编码器映射：
   - 将仿真 `dof_1..dof_8`（`qpos/qvel`）与真机 CAN 双编码器输出轴绝对位置逐一对应，核对关节编号、零位定义（明确对应 `φ=1.5` 还是 `φ=0`）、正方向、限位与单位（`rad`）。
   - 电流与反馈力矩作为可用时的负载辅助信号，角度和速度作为主要闭环反馈。
2. 静态姿态与包络重建验收：
   - 使用实测静态编码器读数输入 `src/controllers/shape_navigation.py`（`ShapeGeometry`），与实物实测外轮廓尺寸对比，记录最大包络误差（mm）。
   - 明确区分：局部机身形状由关节编码器与几何决定；世界坐标净空额外依赖外部/机载定位误差，需将定位误差单独记账。
3. 低速单条形变验证：
   - 先验证低速、小幅、可随时急停的单条已知闭链形变动作，比较仿真与真机在相同指令下的 8 轴跟踪误差、电机电流/力矩峰值及机械回差/滞回。
4. 万向球悬挂与轮载标定：
   - 核对单万向球实测行程（mm）、力-位移曲线（预紧力 `F_0` 与刚度 `k`）和回弹阻尼，将实测值与仿真 `SuspensionConfig`（`k=2400 N/m`, `F_0=0.2 N`, `c=28 N·s/m`, `6 mm`）逐项对比；标定通过悬挂压缩量或独立传感估计轮载的误差范围。
5. 移动中舵轮协调验证：
   - 在空旷平地低速验证纯平面行驶与缓慢边走边变形，单独记录打滑、转向跟随滞后、定位漂移及所有未观测量。

二、输出
- 生成 `hardware_interface_checklist.json` 与 `hardware_alignment_report.md`，明确列出“已实测对齐项 / 存在误差项 / 尚未观测项”，不得以仿真指标代替真机验收。
```

---

## 论文核对摘记

### Adaptive hierarchical origami-based metastructures · Y. Li et al., *Nature Communications*, 2024, 15(1), 6247

论文作者为 Yanbin Li, Antonio Di Lallo, Junxi Zhu, Yinding Chi, Hao Su, Jie Yin（发表于 *Nature Communications* 2024，DOI: `10.1038/s41467-024-50497-5`）。论文结合厚板折纸（thick origami）与自然界多层级结构，构建了基于多面体的空间闭环分层折纸超结构，仅依靠 `<3` 个主动自由度与刚性折叠运动学即可在单一结构上实现超过 `10³` 种三维体态重构。对 TARS 的启发在于直观展示了“厚块空间闭链机构”所具备的多态形态空间与少驱动潜力；差异在于该工作聚焦静止超结构的机构构造与解析折叠路径，不涉及移动底盘协同以及由反馈扰动、回退复测驱动的阶段约束辨识。[原文](https://www.nature.com/articles/s41467-024-50497-5)

### BRUCE · Y. Tanaka et al., *IEEE-RAS Humanoids*, 2025 (`arXiv:2507.00273`)

论文作者为 Yusuke Tanaka, Alvin Zhu, Quanyou Wang, Yilun Liu, Dennis Hong（UCLA RoMeLa），发表于 *2025 IEEE-RAS 24th International Conference on Humanoid Robots (Humanoids 2025)*。论文在 MuJoCo MJX 中显式保留差动滑轮、五杆和四杆等并联闭链传动（而非串联近似），采用课程强化学习训练人形机器人行走策略，并实现零样本真机部署且泛化优于传统 MPC。可迁移的是闭链等式约束在 MuJoCo 中的保真建模与仿真到真机的验证路线；差异在于其执行器闭链结构与运行模式在行走中保持固定，端到端策略不向上层暴露可查询的形变阶段接口，未涉及 TARS 的阶段相关耦合发现与舵轮协同变形。[原文](https://arxiv.org/abs/2507.00273) · [项目页](https://alvister88.github.io/BRUCE_RL_Website/)

### Design and Motion Planning for a Reconfigurable Robotic Base · J. Pankert et al., *IEEE RA-L / IROS*, 2022

论文作者为 Johannes Pankert, Giorgio Valsecchi, Davide Baret, Jon Zehnder, Lukasz L. Pietrasik, Marko Bjelonic, Marco Hutter（ETH Zurich Robotic Systems Lab），发表于 *IEEE Robotics and Automation Letters (RA-L)*, vol. 7, no. 4, pp. 9012–9019, 2022（同步入选 *IROS 2022*，DOI: `10.1109/LRA.2022.3189166`）。论文提出带主动舵轮的可重构移动底盘，通过统一 MPC 同步规划导航、底盘轮廓变形与操作，在窄通道收缩轮廓通过、在重负载操作时展开增大支撑多边形并抑制振动。对 TARS 极具参考价值的是“主动舵轮 + 轮廓重构 + 窄道/支撑兼顾”的移动控制框架；核心差异在于该底盘为单一模式、平滑解析参数化的三角形结构，而 TARS 为 8 块闭链多阶段可变形机身，其耦合与可行运动随阶段变化，需要引入反馈验证的脊髓三图与三层闭环接口。[原文 arXiv](https://arxiv.org/abs/2206.15298) · [DOI](https://doi.org/10.1109/LRA.2022.3189166)

