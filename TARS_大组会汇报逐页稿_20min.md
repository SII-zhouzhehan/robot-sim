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

**三图实验口径（最新）**　一次完整的 Exp A 活动包含多轮试探、复位、图更新、冻结和独立复测，不能用一次动作或一次连续运行同时证明“发现”和“复用”。`G_struct` 由 XML 读取并在整个实验中固定，不属于学习结果；`G_local(s)` 由阶段内反馈建立，`G_transition` 根据可重复的局部响应和复位重测建立。只有冻结图后的新 episode 仍能执行成功，相关关系才进入 `verified`。沿途包络、轮位和支撑记录是动作边的轨迹证据与小脑接口，不是第四张图。

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

| Main Challenges | Why They Matter |
|---|---|
| Which joints move together in each shape? | Predicting which joints will follow helps the controller choose a feasible motor command. |
| When does a new motion become possible? | Recognizing the transition condition helps the robot reach the next stage again after a reset. |
| How should the wheels steer as the body changes shape? | Wheel control needs the current wheel positions. Navigation needs the clearance required along the path. |

### 三张独立插图的提示词

三张图均为 16:9 可编辑矢量示意图，沿用前页配色与机器人轮廓。每张图对应表格中的一行。图中只保留必要的英文短标签；图注统一用 *Conceptual illustration*。

**图 1 · 同一条电机指令，哪些关节跟着动？**

> 并排画同一台八模块闭链机器人处于两种姿态，保持模块次序一致。在同一电机处画相同的蓝色指令箭头，用琥珀色虚线弧箭头标出两种姿态下可能随动的关节。将两侧指令用一个短标签 “Same command” 连起来，虚线图例写 “Possible response”。让读者一眼看到需要通过反馈比较什么。

**图 2 · 从发现一个新动作，到重置后再次做出来**

> 画上下两条路径，两行都从全关节归零的同一轮廓出发。上行画探索过程，在两个相邻姿态施加相同的小幅试探，用浅灰短线和蓝色位移弧分别表示一种假设情形下该方向受限和可动的反馈。下行画重置、重放探索中记录的动作、再次试探三个画面，末端加问号。只标 “Explore”“Reset & replay”“Same response?”。保留闭链轮廓，用反馈差异表达候选转换条件。

**图 3 · 变窄过门时，轮子和机身各需要什么信息？**

> 用俯视图画一道窄门。在同一条路径上叠画宽姿态、收窄中、窄姿态三个轮廓，四个舵轮随各自模块移动。用蓝箭头表示轮子的滚动方向，用半透明琥珀色带表示机身沿途扫过的范围，在门框内侧标出最小间隙。只标 “Wheel direction” 和 “Clearance”，让同一幅图同时体现舵轮控制和导航各自关心的信息。

### 中文讲稿

这一页只讲三个必须解决的问题。第一，闭链机器人在不同姿态下，哪些关节会一起动可能不同，所以要从反馈中学出当前可用的运动关系。第二，机器人不能依赖 `phi` 或人工写好的阶段条件，而要从同一个初始姿态出发，通过试探发现什么时候出现新的运动能力，并在重置后复现这个转换。第三，身体变形会同时改变舵轮的位置、朝向和通过宽度，因此脊髓层需要把当前形态和可动状态传给小脑层，帮助它选择合适的移动动作。三张图分别对应这三个问题，后续实验要验证这些信息能否被三层图保存并被控制程序重新使用。

---

# 三、已有工作

## 4. 相关工作与本项目的差异 · 1:50

### English slide text

**Prior work covers closed-loop morphing, parallel-actuated control, and reconfigurable swerve bases**

#### Hierarchical Origami Metastructures · Closed-loop morphing *(Y. Li et al., Nature Communications 2024)*
Thick origami panels fold into over 1,000 shapes.

**Remaining Limitations**

The stationary structure follows predefined folding paths.

#### BRUCE · Closed-chain locomotion control *(Y. Tanaka et al., IEEE-RAS Humanoids 2025)*
A walking policy trained in MuJoCo transfers directly to BRUCE.

**Remaining Limitations**

The walking controller uses a fixed mechanism mode.

#### Reconfigurable Robotic Base · Morphing + swerve MPC *(J. Pankert et al., IEEE RA-L / IROS 2022)*
One controller coordinates wheel steering and chassis width.

**Remaining Limitations**

The base model assumes a single mode of deformation.

### 中文对照

三项顶刊/顶会代表性工作分别从“闭链多态变形机构”“MuJoCo 闭链仿真控制”“可重构舵轮底盘运动规划”三个维度对应本课题核心边界：
1. **Hierarchical Origami Metastructures（Y. Li 等，*Nature Communications* 2024）**：采用厚板多面体空间闭环折纸机构，仅靠 `<3` 个主动自由度与解析刚性折叠运动学实现 `>10³` 种三维形态切换，奠定了厚块闭链多态变形的机构基础；但其对象为**静止超结构**且依赖离线解析折叠路径，不涉及移动底盘与反馈驱动的阶段条件复测。
2. **BRUCE（Y. Tanaka 等，*IEEE-RAS Humanoids* 2025，UCLA RoMeLa）**：在 MuJoCo MJX 中原生保留差动滑轮、五杆、四杆闭链传动并实现课程 RL 零样本真机行走迁移，验证了闭链仿真与控制路线；但其构型模式在行走中**固定不变**，端到端策略不向上层暴露可查询的形变阶段接口。
3. **Reconfigurable Robotic Base（J. Pankert & M. Hutter 等，*IEEE RA-L / IROS* 2022，ETH Zurich RSL）**：用统一 MPC 协同主动舵轮转向与底盘轮廓收缩/展开，兼顾窄通道穿行与操作稳定性；但其底盘为**单一模式、平滑解析参数化的三角形结构**，未涉及多阶段闭链耦合与反馈验证的阶段转换图。

### 插图与版式

严格按模板相关工作页排三张等宽卡片，每张包含“顶部示意图（自绘矢量或预留论文原图贴图框）+ 核心做法 + 蓝色/绿色高亮 `Remaining Limitations` 局限栏”，底部加一条通栏定位条 *TARS bridges all three: Closed-loop multi-stage morphing + native MuJoCo closed-chain control + swerve-coordinated mobile navigation*：
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
  (3) "What it solves" 摘要区（浅灰底，一句简短英文）；
  (4) "Remaining Limitations" 局限栏（浅蓝/浅绿底，一句简短英文）。

三张卡片具体内容：
- Card 1: "Origami Metastructures" | Tag: "Y. Li et al., Nat. Commun. 2024"
  * 示意图：厚板闭环多面体小块经折叠箭头变成 2–3 种不同三维轮廓，标签 "Closed-loop thick blocks → >10³ shapes (stationary, analytical)"。
  * What it solves: "Thick origami panels fold into over 1,000 shapes."
  * Remaining Limitations: "The stationary structure follows predefined folding paths."

- Card 2: "BRUCE · Parallel Humanoid" | Tag: "Y. Tanaka et al., Humanoids 2025"
  * 示意图：简化人形下肢旁画并联四杆/五杆与滑轮小图标，箭头指向 "MuJoCo MJX + Curriculum RL → Hardware"，标签 "Native closed-chain sim → fixed-mode locomotion"。
  * What it solves: "A walking policy trained in MuJoCo transfers directly to BRUCE."
  * Remaining Limitations: "The walking controller uses a fixed mechanism mode."

- Card 3: "Reconfigurable Swerve Base" | Tag: "J. Pankert et al., RA-L / IROS 2022"
  * 示意图：带 3 组舵轮的三角形底盘在 "Narrow pass" 与 "Wide stance" 之间切换，旁标 "Unified MPC"，标签 "Active swerve + footprint morph (single-mode kinematics)"。
  * What it solves: "One controller coordinates wheel steering and chassis width."
  * Remaining Limitations: "The base model assumes a single mode of deformation."

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

**Feedback closes the loop:** measured joint response → spinal layer; clearance, support and tracking → cerebellum; task outcome → brain. Closure residual is logged by the safety/evaluation monitor unless a real sensor explicitly exposes it.

### 中文对照

大脑用 FSM 导航并选择目标阶段，PPO 作为计划中的策略探索任务层选择。脊髓利用三图表征阶段相关的可行运动，输出协调关节轨迹和转换状态。小脑协调形变与机身平移、朝向、舵角和轮速。实测关节响应回脊髓，净空、支撑和跟踪误差回小脑，任务结果回大脑。闭链残差默认由安全/评测记录保存，除非真实传感器明确提供给控制器。

### 插图与版式

三层控制闭环框图，**视觉重心放在“三层分工 + 双向接口 + 三路反馈闭合”**（脊髓模块内只放紧凑的 `G_struct / G_transition / G_local(s)` 三图徽章，把三图细节留到第 6 页展开，避免信息过载；注：仓库旧图 `outputs/three_layer_three_graph_20261114/three_layer_three_graph_framework.pptx` 术语与版面偏满，建议按下方提示词重生成与本稿术语完全一致的简洁新版）。PPO 用虚线框标为 *Planned option*。

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
     * 内部 3 个紧凑徽章：`G_struct`、`G_transition`、`G_local(s)`
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
  * Plant → Spinal："8-axis measured joint response"
  * Safety monitor → Evaluation log: "closure residual / contacts (not learner input)"
  * Plant → Cerebellum："Clearance, 4-wheel support & tracking error"
  * Spinal & Cerebellum → Brain："Reached / Pause / Failure → Replan"

【图例】底部一行极简图例：蓝实线 = Control / query；绿线 = Execution feedback；虚线框 = Planned extension (PPO)。
```

### 中文讲稿

大脑负责任务目标和导航，FSM 请求目标阶段，PPO 是后续探索任务层选择的方向。脊髓根据当前反馈给出是否可达以及可执行的协调变形轨迹。小脑把机身形态变化与世界坐标中的移动、朝向、舵角和轮速协调起来。执行中的关节和闭合反馈更新脊髓状态；净空、支撑和跟踪情况交给小脑；到达、暂停或失败的结果回到大脑，供 FSM 继续或重新规划。

## 6. 脊髓三图：硬件关系已知，运动规律靠探索 · 1:25

### English slide text

**One graph is fixed from XML; two are learned and then reused**

| Graph | Source and role | Information returned to the controller |
|---|---|---|
| **G_struct · Physical structure** | Fixed XML hardware context | Blocks, joints, actuators, sensors and loop-closure links; never treated as a learned motion graph |
| **G_local(s) · Motion within a stage** | Learned from repeated feedback | Tested single- and multi-joint relations, valid range, confidence and evidence |
| **G_transition · Discovered stages** | Learned from local evidence and reset tests | Behavior modes, directed actions, observable conditions, outcomes and replay status |

### 中文对照

`G_struct` 只从 XML 读取物理连接，不提前写入运动规律，也不需要被“训练”。脊髓先用试探反馈建立 `G_local(s)`，再把反复出现且可区分的运动模式组织进 `G_transition`。XML 中的父子树用于坐标计算，闭环连接仍作为物理图上的一条边。候选联动和转换边要记录证据，冻结图后在新的复位试次中重测成功才可交给上层执行。一个完整实验活动可以得到三图接口，但单个 episode 不能证明三图已经可复用。

### 插图与版式

主图画三个独立但互相关联的对象：XML 闭环硬件图、由行为模式节点组成的有向转换图、某个模式对应的局部关节响应图。用虚线表示候选关系，用实线表示复测验证关系。图 2 的单独绘图提示见 [三图示意图插图清单](THREE_LAYER_THREE_GRAPH_PPT_PROMPT.md)。不要画成三个神经网络层，也不要用姿态进度变量给阶段命名。

### 中文讲稿

第一张图只描述硬件怎么连接。机器人自己试探后，脊髓逐步学到不同运动状态下电机和关节的响应关系，并把反复出现的状态和可执行转换连接起来。XML 的父节点只是文件组织形式，在闭环机器人里不能把它当成唯一的机械父子关系。

## 7. Exp A：从零位开始反复探索和更新三图 · 1:15

### English slide text

**The learner discovers motion modes from commands and feedback**

1. Reset to measured joint zero and check drift.
2. Apply safe positive/negative and combined joint probes.
3. Read permitted feedback and update `G_local(s)` evidence.
4. Detect repeatable response changes and propose `G_transition` candidates.
5. Reset and retest candidates, then freeze the graphs.
6. Replay frozen actions on independent reset episodes.

*XML connectivity and permitted feedback are the learner’s only information sources.*

### 中文对照

每轮都从八个驱动关节的实测零位开始。探索器只读 XML 硬件连接、事先声明的传感器反馈和自己发出的动作。先做小幅正反向单轴试探，再逐步加入多轴组合，先更新 `G_local(s)` 的响应证据。只有当局部响应模式和可执行动作在复位后仍可区分，才提出 `G_transition` 候选。候选图冻结后，使用新的独立 episode 做重放，不能把复测数据继续用于这次冻结图的更新。闭链误差等隐藏仿真量只供安全停止和事后评分，不进入学习器。

### 插图与版式

画独立的循环插图：*Reset at measured zero → Safe probe → Observe permitted feedback → Update G_local / G_transition → Choose next uncertain action → Reset and retest*。在循环外单独画安全监控和事后评价框，避免隐藏仿真数据连到学习器。候选关系用虚线，已验证关系用实线。单独插图提示见 [三图示意图插图清单](THREE_LAYER_THREE_GRAPH_PPT_PROMPT.md)。

### 中文讲稿

Exp A 不再从预选工作点开始拟合。每个试次都回到同一个实测零位，探索器只知道机械连接和允许读取的反馈。它先试探各个电机，再尝试组合动作，观察反馈规律是否稳定。随后它根据不确定性选择新的试探方向。只有在新复位试次中能够重复执行并通过完成检查的关系，才会升为已验证。每轮的图和原始数据都会保留，用于分析它如何逐步改进。

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
   - Col 4（Badge: "Robot Feedback", 绿）：标题 "4. Observe" — "Measured joint q/q̇ and declared sensors"
   - Col 5（Badge: "Brain / Graphs", 深蓝）：标题 "5. Decision" — 3 枚极简状态标签："Reached (green)"、"Pause (amber)"、"Replan (gray/red)"
   - 时序带底部画两条折返反馈线（绿色虚线/实线）：
     * 从 Col 4 折返到 Col 2："Update from permitted feedback"
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

## 9. Exp A 如何验证三图结构 · 1:10

### English slide text

**Build the graphs, freeze them, and test reuse**

- **Discovery:** keep `G_struct` fixed from XML; learn `G_local(s)` and candidate `G_transition` from permitted feedback across repeated reset episodes.
- **Freeze and replay:** stop graph updates, then test unseen joint combinations and learned transitions on independent reset episodes.
- **Fair controls:** compare random safe probing, flat trial memory, a global response model, and the three-graph method under equal budgets.

**Metrics:** held-out joint-response error · repeatability · false transition rate · probes to discovery · closure and torque safety

### 中文对照

主问题是：机器人能否从零位开始，只凭 XML 连接关系和允许的反馈，先建立阶段内运动关系，再建立可重复的转换边，并在图冻结后用新的复位试次重放这些动作。这里的“跑一次实验”指一整个多轮实验活动，不是一次连续动作。`G_struct` 只是固定的 XML 硬件图，真正需要验证的是 `G_local(s)` 和 `G_transition` 是否能被发现、冻结和复用。对照方法使用同一模型、复位、观测、控制限幅和动作预算。探索期不提供阶段标签、目标姿态库或仿真器内部约束量。

### 插图与版式

画三行简洁实验关系图：第一行呈现零位到多轮自主探索；第二行呈现冻结图后的未见多轴动作和转换复测；第三行并列展示公平对照方法。不要画预测性柱状图或虚构数值。绘图素材与独立插图提示见 [三图示意图插图清单](THREE_LAYER_THREE_GRAPH_PPT_PROMPT.md)。

### 中文讲稿

判断实验成功要看学到的部分能不能重用，不是只看最终有没有三张图。`G_struct` 只需核对 XML 解析正确；`G_local(s)` 要通过未见过的联合动作测试；`G_transition` 要在图冻结后的新复位试次中重做，并在失败时停止或回退。还要用相同交互预算与随机探索、平坦记录和全局响应模型比较。这样才能判断三图表示是否真的提供了帮助。

## 10. 完整 CAD 模型导出与电机选型对比（预计 vs 实测） · 1:25

### English slide text

**Full CAD MuJoCo export reveals a 44% heavier robot, lower body peaks, and a tighter hub-drive cap**

| Item | Early Surrogate Estimate *(Oct 2026)* | Full Exported CAD Model *(Nov 2026)* | Engineering Takeaway |
|---|---|---|---|
| **Closed-chain & support** | Inferred 8th hinge (~5.18 mm gap, ~14.6° offset); rigid caster proxy (`R=25.5 mm`) | Native 8-joint CAD + exact 6-DoF `weld` (`<1e-10`); **8 explicit 6 mm spring casters** (`R=28 mm`) | Eliminates artificial closure fighting; realistic tire/caster load sharing |
| **Total mass** | `6.27–6.42 kg` (assumed +0.15 kg closure) | **`9.26 kg`** full CAD mass (**+44%**) | Higher inertial & rolling load on drive modules |
| **8 body joints (6-s morphing sweep)** | RMS `0.39–1.40 N·m`; peak **`8.31–9.02 N·m`** *(1.5× spec ≥ 2.5 / 14 N·m)* | CAD sweep peak **`4.28 N·m`** (RMS `0.61–1.38`); w/ ramps & wheel coord: **`3.56 N·m`** (RMS `0.38–1.18`) | **Peak dropped ~50–60%**; zero saturation under `11 N·m` cap *(flat ground, no payload)* |
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
   - Row 2: "Body Torque (6-s morphing sweep)" | "Peak 8.31–9.02 N·m · RMS 0.39–1.40" | "Peak 3.56–4.28 N·m · RMS 0.38–1.38" | "Peak -50–60%; 0 s at 11 N·m cap"
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

**Grounded chassis and basic motion are verified; autonomous graph discovery is next**

- **Explicit 8-caster spring suspension (`k=2400 N/m`, 6 mm):** raises coast-morph 4-wheel support from **`50.8% → 90.3%`** (`82.1 → 23.6 mm` peak error) and halves tire penetration (`3.24 → 1.71 mm`) with **0 N** housing scrape.
- **Planar drive & 4 mm bump (`TarsGroundedEnv`):** 6-direction travel (`x± / y± / yaw±`) achieves **`96.5–98.9%`** 4-wheel support and **`≤ 0.28°`** yaw drift; single-caster 4 mm bump recovers **`100%`** support.
- **Moving-while-morphing & steering decoupling (`RP¹` scrub-gradient + block-yaw FF):** four-frame decoupling (`{W}/{B}/{R₀}/{Mᵢ}`) + urgency-gated projective-circle (`RP¹`) steering eliminates a 180° steering flip during morphing (`lateral` max `|q|` `530° → 102.5°`, all 4 wheels return to `0°`; `moving` transition-region swing `200.6° → 127.6°`), reaching **`100%`** (`lateral`), **`99.9%`** (`still`), **`97.2–98.6%`** (`moving/reverse/half`) 4-wheel support, COM error **`11.5–15.2 mm`**, weld gap **`< 0.09 mm`**.
- **Offline model reference only:** geometry/constraint diagnostics and swept-envelope calculations can be used after exploration to score results; they are hidden from the learner and cannot guide its action choices.

### 中文对照

近两日在最新 `tars_mjcf_grounded` 模型上完成了基础移动框架与底层物理前提的系统验证并已冻结定版：
1. **8 万向球显式弹簧悬挂**：相比旧刚性接触代理，滑行变形四轮着地率由 `50.8%` 提升至 `90.3%`（峰值偏移 `82.1 → 23.6 mm`），轮胎最大压入减半（`3.24 → 1.71 mm`），全程无轮罩/机身刮地；
2. **六向平面行驶与 4 mm 越障**：`x± / y± / yaw±` 四轮承载率 `96.5%–98.9%`、偏航漂移 `≤ 0.28°`，单侧 4 mm 凸起越障后恢复 `100%` 四轮承载；
3. **边走边变形与四坐标系舵轮解耦（已定版）**：建立 `{W}/{B}/{R₀}/{Mᵢ}` 四坐标系与无根块脊髓变形接口 `spinal_morph_kinematics()`，并采用紧迫度门控射影圆（$\mathbb{R}P^1$）侧滑能量梯度 + 自适应安装块偏航前馈，彻底消除 变形过程中的 180° 急转/回弹（`lateral` 单轮最大转角从 `530°` 压至 **`102.5°`** 且四轮归零、`100%` 接地、倾角 `6.0°`；`moving` 变形区间内摆幅从 `200.6°` 降至 `127.6°`；5 种主动工况四轮承载率 **`97.2%–100%`**，闭链 `weld` 间隙 `< 0.09 mm`）；
4. **离线评测参考**：闭链约束与形态包络的仿真内部量只用于安全审计和探索后的结果评分，不提供给新的自主探索器。沿途包络也可在后续导航实验中用于比较进场位置。

### 插图与版式

采用 **2×2 实测证据网格**，直接嵌入仓库已生成的真实可视化与动图（每格配一行醒目英文量化结论）：
- **左上（显式悬挂对照）**：贴入 `experiments/grounded_caster_suspension_20261124/suspension_closeup_demo.gif`（单舵轮双万向球 6 mm 弹簧压缩特写）或 `baseline_vs_suspension_comparison.svg`。
- **右上（舵轮解耦与边走边变形）**：并排播放 `experiments/grounded_steering_decouple_20261124/lateral_steering_comparison.gif`（最直观：`±530°` → `±102.5°` 归零），配合无网格对比曲线 `steering_angles_before_after.svg`（或 `moving_steering_comparison.gif`、`steering_metrics_summary.svg`）。
- **左下（六向驱动与 4 mm 越障）**：贴入 `experiments/grounded_caster_suspension_20261124/swerve_drive_and_bump.gif` 关键帧 + 实测指标角标。
- **右下（离线评测参考）**：贴入 `experiments/grounded_support_navigation/tire_loads.png`，并标注 *Post-hoc / safety evaluation only; hidden from Exp A learner*。

#### 可编辑 PPT 插图生成提示词（第 11 页 · 2×2 实测证据画框）

```text
请遵循【可编辑 PPT 插图通用视觉规范】，生成第 11 页（Recent Grounded Chassis & Substrate Validation）的原生可编辑 PPTX 与 SVG 矢量版式框架，尺寸 12.1 × 5.2 in，图内全部文字使用英文。

【布局结构：2×2 等大证据卡片网格（每卡宽 5.9 in，高 2.48 in，横纵间距 0.25 in），每卡分为“顶部标题+结果徽章条（0.45 in）” + “中部真实图像/GIF 贴图框（1.58 in）” + “底部一行量化实测数据（0.42 in）”】
- Card 1 (左上，绿标 "Done"): 标题 "1. Explicit 8-Caster Spring Suspension"
  * 贴图框提示："Insert suspension_closeup_demo.gif / baseline_vs_suspension_comparison.svg"
  * 底部实测："Coast 4-wheel support: 50.8% → 90.3%  |  Tire penetration: 3.24 → 1.71 mm  |  Housing scrape: 0 N"

- Card 2 (右上，绿标 "Done"): 标题 "2. Moving-Morph & RP¹ Steering Decoupling"
  * 贴图框提示："Insert lateral_steering_comparison.gif / steering_angles_before_after.svg"
  * 底部实测："Lat |q|max: 530° → 102.5° (0° return)  |  4-wheel: 100% (lat), 97.2–98.6% (mov/rev/half)  |  morphing flip fixed"

- Card 3 (左下，绿标 "Done"): 标题 "3. 6-Dir Planar Drive & 4 mm Bump"
  * 贴图框提示："Insert swerve_drive_and_bump.gif keyframe"
  * 底部实测："6-dir 4-wheel support: 96.5–98.9%  |  Yaw drift ≤ 0.28°  |  4 mm bump recovery: 100%"

- Card 4 (右下，灰标 "Evaluation Only"): 标题 "4. Offline Geometry & Safety Reference"
  * 贴图框提示："Insert grounded_phi_local / grounded_support_navigation figure"
  * 底部说明："Post-hoc diagnostics only  |  Not available to the exploration learner"

【质量约束】所有贴图区留白居中并带虚线框，方便一键替换为仓库已有 SVG/PNG/GIF；实测数字全部原生文本框可编辑。
```

### 中文讲稿

这一页总结近两天在新模型上已经完成并冻结定版的底层验证，证明开展脊髓三图探索和三层耦合闭环的基础移动与物理前提已经完全到位。

左上角是显式 8 万向球 6 毫米弹簧悬挂的特写与前后对照：相比旧刚性代理，滑行变形的四轮着地率从 50.8% 提到 90.3%，轮胎最大压入减半，全程零轮罩刮地。左下角是六向平面行驶和 4 毫米单侧越障测试，四轮承载率保持在 96.5% 到 98.9%，偏航漂移不到 0.3 度，越障后完全恢复四轮支撑。

右上角是边变形边移动与四坐标系舵轮解耦的实测对比：我们在脊髓层抽出了与根块无关的内部变形运动学接口，在小脑层结合安装块偏航前馈和射影圆侧滑能量梯度，彻底消除了旧版舵轮甩到 ±530 度再急转回弹，以及变形中出现的 180 度突转问题。可以看到横移边变形中四轮全程稳定在 ±102.5 度以内并对称归零，四轮接地率达到 100%，5 种主动移动变形工况接地率全部在 97.2% 到 100%，67 项单元测试全部通过。右下角这项只代表仿真内部的事后评测能力，不会提供给新的探索器。当前底盘已满足开展自主探索的基础条件，接下来要通过全新的 Exp A 检验探索器能否仅凭 XML 连接和允许反馈建立脊髓三图。

## 12. 已完成的预实验与新的 Exp A · 1:35

### English slide text

**Completed: basic model checks and local response prediction**

- **Exp 0:** the current closed-chain MuJoCo model passed the basic chassis and motion gates.
- **Previous local-response pilot:** 1,676 probes compared several predictors around preselected configurations.
- **What remains untested:** autonomous discovery of motion modes and transition actions from the zero configuration.

**Next Exp A:** XML connectivity + permitted feedback → iterative exploration → three-graph snapshots → frozen replay on new episodes.

### 中文对照

实验 0 已完成新模型和底盘基础运动验收。旧 Exp A 做了 1,676 次局部响应试探和模型预测对照，能作为“局部动作响应可学习”的先导结果。它从预选工作点开始，没有验证从零位自主发现阶段和转换条件。新的 Exp A 将从实测关节零位开始，只把 XML 硬件连接提供给探索器，其余运动关系和阶段都通过反馈学习。

### 插图与版式

- 已有结果图可从 `experiments/expA_gstate_deep_20261125_021500/` 复制 `expA_budget_convergence.png` 或 `expA_model_and_closedloop_eval.png`。图旁必须标注“preselected local-response pilot”，避免将它解释为自主图发现。
- 下一步方案图单独绘制为“固定零位 → 安全试探 → 反馈更新三图 → 新动作探索 → 独立复测”的循环，并在学习器边界标出“XML connectivity + permitted feedback only”。绘图提示见 [三图示意图插图清单](THREE_LAYER_THREE_GRAPH_PPT_PROMPT.md)。
- 不要画没有实验数据支持的阶段节点数量、转换成功率或性能柱状图。

### 中文讲稿

到目前为止，我们完成了两层前置工作。第一，完整闭链模型和底盘基本移动已经通过实验 0 的验收。第二，旧 Exp A 做了局部关节响应预测，说明在给定工作点附近，反馈数据可以帮助预测新动作。但旧试验从预先选好的配置开始，所以还没有回答我们的核心问题。下一轮 Exp A 会让探索器从实测零位重新开始，只读 XML 连接关系和允许的传感器反馈，反复选择安全动作，自己建立阶段内运动图和阶段转换图，再在全新试次里检验学到的动作能否重放。

## 13. 下一步计划：先验证自主发现，再接入导航 · 1:05

### English slide text

**Build evidence in the order required by the control loop**

1. **Exp 0 · Complete** — freeze the current model, chassis controller, and measured zero reset.
2. **New Exp A · Next** — iteratively discover and retest the spinal graphs from XML connectivity and permitted feedback only.
3. **Brain–Spinal–Cerebellum integration · Later** — connect verified morphing paths to navigation, swerve coordination, and a physical corridor task.
4. **Hardware transfer · Future** — align encoders, geometry, suspension, and motor limits with the real robot.

### 中文对照

实验 0 已完成。下一步先用新的 Exp A 验证探索器能否从零位只凭 XML 连接关系和允许的反馈，逐轮学出并复测三张脊髓图。三图复测通过后，再接入大脑任务决策和小脑舵轮协调，最后做真机接口对齐。

### 插图与版式

用四个独立步骤组成横向路线图。突出“Exp A · Next”，把端到端导航和真机迁移标为后续阶段。插图只画流程节点和箭头，不生成整页 PPT，也不放虚构指标。单独绘图提示见 [三图示意图插图清单](THREE_LAYER_THREE_GRAPH_PPT_PROMPT.md)。

### 中文讲稿

底盘和模型基础验收已经完成。接下来不直接跳到窄道任务，而是先让机器人从零位自己探索并建立三张图，检查未见动作和转换能不能复测。这个阶段通过后，再把验证过的变形路径交给上层任务规划和小脑运动协调。最后才开展真机传感器、几何与动力学对齐。

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
| **实验 0** | 模型与底盘基础验收 | 仿真底座 + 小脑底层 | **已完成**：现行闭链模型、控制器、零位静置和基础移动已记录 | `experiments/exp0_baseline_20261125_015900/` |
| **旧 Exp A** | 预选工作点的局部响应预测 | 局部辨识先导 | **已完成但范围有限**：没有从零位自主发现阶段或三图 | `experiments/expA_gstate_deep_20261125_021500/` |
| **新 Exp A** | 只凭 XML 连接和允许反馈，从零位多轮发现三图 | 脊髓探索与执行 | **下一步**：先做接口/信息泄漏审计，再按安全门禁逐轮探索和复测 | 三图快照、原始 episode、预算学习曲线、冻结边新试次重放结果 |
| **Exp D** | 三图接入大脑 FSM 与小脑舵轮协调 | 完整闭环 | **后续**：新 Exp A 得到可复测动作后，再补齐物理任务场景和统一接口 | 物理窄道任务、任务成功率与运动安全对照 |
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

## 实验 0 · 模型与底盘基础验收（已完成）

实验 0 已在当前完整 CAD 闭链 MuJoCo 模型上完成模型冻结、零位静置、六向移动、越障、移动变形和八块刚体映射检查。详细数值、模型哈希与安全门槛见 [`experiments/exp0_baseline_20261125_015900/summary.md`](experiments/exp0_baseline_20261125_015900/summary.md)。

后续 Exp A 每个试次只从实测关节零位 `q=0` 复位。历史标准姿态库只用于旧模型验收，不输入新探索器，也不用于给阶段命名。

## 实验 A · 从零位自主发现脊髓三图

**目的**　验证机器人能否从固定零位出发，只凭 XML 中的硬件连接和公开的反馈，自主试探并逐步建立阶段内运动图与阶段转换图，再用冻结图规划和复测可执行的变形动作。

**11 月 26 日实测**　两轮共 518 个复位试次。机器人从单电机正反试探学到了零位附近的联动。冻结后，在同一批 96 个未见联合动作上，八关节位移预测误差为 0.192°，完整矩阵也是 0.192°，只考虑各电机自身响应时为 1.159°。放慢变形后，12 个进场目标中能完成诊断的从 9 个增至 11 个。已测区域的响应均归入起始模式，仍需找到能跨复位重现的第二种响应，才能检验 FSM 重放变形。[两轮简报](experiments/expA_encoder_discovery/RESULTS.md)

**当前结果边界**　`experiments/expA_gstate_deep_20261125_021500/` 已做 1,676 次仿真试次，包含预选工作点附近的局部响应预测、闭环执行和记忆读写。它是局部辨识试验，不是自主三图发现。该历史结果可作为局部响应预测的先导数据，不能用来声称 Exp A 的核心目标已经通过。

**初态与学习器可见信息**　每个 episode 从八个驱动关节的实测零位开始。学习器只读冻结 XML 解析的物理连接、事先列出的传感器反馈和自己的命令/执行记录。不得读取人工阶段标签、预选姿态库、姿态进度变量、参考变形路径、闭链雅可比或 MuJoCo 隐藏约束量。隐藏闭合误差、接触、载荷和力矩可以用于安全停机与事后评分，必须与算法可见数据分开保存。

**三图职责**

| 图 | 初始来源 | 探索中学到的内容 |
|---|---|---|
| `G_struct` | XML 唯一先验 | 模块、关节、执行器、传感器和闭环连接，不包含运动规律或阶段答案 |
| `G_local(s)` | 空图开始 | 每种可重复运动模式下，单轴/多轴命令对各关节的响应关系、适用范围、置信度和试验依据 |
| `G_transition` | 空图开始 | 可重复行为模式节点；有向动作边、可观察的进入/完成条件、成功/失败证据与回退方式 |

**多轮设计**　A0 审计零位复位、传感器底噪和数据白名单；A1 采集单轴正负响应和零输入对照；A2 用多轴联合动作学局部耦合并留出未见组合；A3 根据可重复的响应改变提出、合并或拆分行为模式；A4 学习转换条件和动作，复位后重复执行并独立检验反向回退；A5 每隔相同动作预算保存三图快照，继续探索不确定前沿；A6 冻结图，在全新 reset episode 中测试预测和动作重放。失败会降级或撤销候选边并作为下一轮训练数据。

**公平对照**　相同模型版本、零位、传感器、控制器、限幅、复位、动作预算和留出 episode 下比较随机安全试探、平坦试验记录、全局响应模型和三图方法。训练/测试按完整 episode 和随机种子隔离。神经网络或图神经网络仅作为可选额外对照，不是图结构定义的一部分。

**核心指标**　未见多轴动作响应误差；可重复发现的行为模式与转换数；每个可靠关系所需试探数；冻结边在新试次中的成功率及置信区间；错误阶段判断/错误触发率；回退成功率；闭链残差、力矩限幅、异常停止和到达目标模式所需时间。

**结论门槛**　至少发现两个可复现行为模式/转换（若模型不支持则如实报告）；冻结图能在新试次重放经验证动作；安全指标通过；并且三图方法相对至少一个同预算基线在预先指定指标上稳定改善。图比矩阵更强不是预设结论。

详细字段、轮次、更新规则及数据文件要求见 [Exp A 三图自主发现计划](experiments/EXP_A_THREE_GRAPH_DISCOVERY_PLAN.md)。独立示意图提示见 [三图示意图插图清单](THREE_LAYER_THREE_GRAPH_PPT_PROMPT.md)。

## 实验 B 与 C · 并入新的 Exp A

旧版把局部响应、功能组识别和转换条件验证拆成三个实验，并以预先指定的模式/工作点安排测试。这个拆分与“探索器从零位自己发现三图”的新目标不一致，因此不再按旧 B/C 提示词单独运行。

原 B 的多方向耦合与功能组检验，现纳入新 Exp A 的 A1–A3；原 C 的转换条件、正反复测和失败记录，现纳入 A4–A6。具体轮次与字段以 [Exp A 三图自主发现计划](experiments/EXP_A_THREE_GRAPH_DISCOVERY_PLAN.md) 为准。

## 实验 D · 图查询接入大脑 FSM 与小脑舵轮协调的端到端闭环对照

**目的**　检验脊髓三图（`G_struct / G_local(s) / G_transition`）提供的可达性、验证动作边、沿途包络与支撑条件，是否能真正改善大脑（FSM）的阶段选择和小脑（舵轮 + 变形）的移动协调，而不仅仅提高单步角度预测精度。
**前置条件与当前缺口说明**　1) 需先完成实验 0（锁定稳定底盘移动与边走边变形控制器）及新 Exp A（获得经过复测标记的图数据）；2) **当前工程缺口**：仓库现有 `MujocoBenchmarkEnv`（A/B/C 场景）、`ScriptedBrain` 与 `MPCController` 仍绑定在旧 7 关节模型上，`tars_mjcf_grounded` 目前仅有空旷平地与 `experiments/grounded_support_navigation` 的离线虚拟墙投影。因此本实验分为两步：先搭建最小可复现的物理窄道任务场景与三层闭环接口，再执行三方法严格对照。
**关键对照**　保持**同一任务场景、相同起终点、相同模型版本、相同底盘舵轮参数、相同安全限幅与相同探索数据预算**，比较三种控制器：  
1. **基线 1（Scripted Morph Replay）**：预先记录的参考变形轨迹回放控制，仅按起终点形态规则切换，不读取三图（不查沿途包络与验证阶段边）；
2. **基线 2（Global Response）**：使用与图方法相同探索数据预算拟合的固定全局响应模型驱动变形与协调；  
3. **本方法（Three-Graph Closed-Loop）**：大脑 FSM 查询 `G_struct / G_local(s) / G_transition`（**仅允许调用新 Exp A 中 `verified` 边**），结合 `src/controllers/shape_navigation.py` 计算沿途包络净空并接收小脑支撑/跟踪/碰撞反馈，支持进场横移、减速、暂停或重规划。
**核心指标**　任务成功率（到达目标且无碰撞/无倾覆/无数值告警）、阶段转换成功率与误判率、沿途最小墙体净空与碰撞力（`mm`, `N`）、路径跟踪误差（`mm`）与形态跟踪误差（`deg`）、闭合残差（`mm`, `deg`）、四轮承载率与最长失载时长（`%`, `ms`）、任务耗时（`s`）与中止原因分类。

### 可直接复制的 Agent Prompt（实验 D）

```text
请先遵循【通用执行约束】，在 robot-sim 仓库中搭建并运行【实验 D：图查询接入大脑 FSM 与小脑舵轮协调的端到端闭环对照】。

一、前置检查与最小基准任务搭建（先搭桥，再做对照）
- 目标模型：`data/my_robot/mujoco/tars_mjcf_grounded/scene.xml`（`TarsGroundedEnv`、`src/controllers/grounded_morph.py`、`src/controllers/shape_navigation.py`）。
- 先核查是否存在已通过实验 0 的冻结底盘控制器，以及新 Exp A 产出的图文件（`g_struct.json`、`g_local/`、`g_transition.json`）。
  * 若某一层接口或经复测通过的 `G_transition` 边尚未完备，不得伪造验证边，请明确在 `summary.md` 中声明已运行的层级边界（例如先运行“已知验证阶段边 + 沿途包络净空与支撑闭环”层级）。
- 在不修改 `data/my_robot/mujoco/tars_mjcf_grounded/robot.xml` 的前提下，通过独立的场景 XML 或 `mujoco.MjSpec` 在平地上构建**带真实物理碰撞墙体的直窄道穿行基准任务**（例如多组通道宽度：`0.80 m / 0.88 m / 1.00 m` 及不同初始横向偏置/随机种子），同时复用 `ShapeGeometry` 从关节编码器实时计算沿途包络净空。
- 将三层接口打通到 `TarsGroundedEnv`：
  * 大脑（FSM）：发出目标位置、目标阶段与进场策略请求，接收“到达 / 暂停 / 净空不足 / 支撑异常 / 失败重规划”事件；
  * 脊髓：查询三图返回可达性、协调 8 轴参考轨迹、沿途包络边界与转换状态；
  * 小脑：通过 `TarsGroundedEnv.drive` 与 `drive_morph` 协调机身平移、朝向、变形进度、舵角与轮速，并实时回传跟踪误差、闭合残差、墙体接触力与四轮/八球支撑状态。

二、三方法公平对照协议（同任务、同初态、同底盘、同限幅、同探索预算）
在至少 3 个通道宽度 × 多个初始横向偏置（或随机种子）下，对比以下三种控制方式：
1. `baseline_scripted_morph`：仅按起终点规则触发预先记录的参考变形轨迹回放，不读取三图，不做沿途包络净空预检查与重规划；
2. `baseline_global_model`：使用与三图方法相同数量探索数据拟合的固定全局局部响应模型，无阶段图查询与条件门控；
3. `three_graph_closed_loop`：查询 `G_struct / G_local(s) / G_transition`，**严格只允许调用已标记 `verified` 的动作边**（未验证或 `undecided` 关系禁止作为可执行策略），执行前由小脑结合沿途包络做净空预检（必要时先横移对齐再变形），执行中基于关节响应、闭合残差、墙体净空和四轮载荷触发减速、暂停或 FSM 重规划。

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
   - 将仿真 `dof_1..dof_8`（`qpos/qvel`）与真机 CAN 双编码器输出轴绝对位置逐一对应，核对关节编号、零位定义（以实测关节零位和编码器映射为准）、正方向、限位与单位（`rad`）。
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

