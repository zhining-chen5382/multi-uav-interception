# 多机协同拦截 MAPPO Baseline v1 技术规格

## 1. 目的

本文档定义多机协同拦截项目的第一版 MAPPO（多智能体PPO）baseline（基线）。

Baseline v1 的主要目标是：

- 搭建一个最小可运行的 3D（三维）多无人机协同拦截环境。
- 跑通 MAPPO（多智能体PPO）的完整训练流程。
- 明确定义 observation（观测）、action（动作）、reward（奖励）、dynamics（动力学）和 evaluation metrics（评估指标）。
- 为后续研究改进提供一个稳定、可比较的 baseline（基线）。

Baseline v1 仅用于初步实验验证。

本文档中的所有具体数值参数均视为 provisional defaults（暂定默认值），后续可以根据实验结果、文献调研或项目要求进行修改。

---

## 2. 基础场景

Baseline v1 采用：

- 3 架 pursuers（追击机）
- 1 架 evader（逃逸机）
- 3D continuous environment（三维连续环境）
- 无 obstacles（障碍物）
- evader 使用 rule-based policy（规则型策略）
- pursuers 使用 MAPPO（多智能体PPO）
- 3 架 pursuers 为 homogeneous agents（同构智能体）

当前基础场景称为：

`3A1C`

即 3 架 A 型追击机协同拦截 1 架 C 型逃逸机。

---

## 3. Environment（环境）

环境采用 bounded 3D space（有限三维空间）。

pursuers 和 evader 的初始位置应支持配置。

Baseline v1 初期应优先使用相对简单的初始条件，确保任务具备可学习性。

后续可以逐步加入：

- 随机初始队形
- 更大的初始距离
- obstacles（障碍物）
- unseen layouts（未见环境布局）
- 不同类型的 evader strategy（逃逸策略）

环境尺寸和初始化范围必须通过 YAML configuration（YAML配置文件）设置，不允许直接写死在 Python 代码中。

---

## 4. Dynamics Model（动力学模型）

Baseline v1 使用简化的 3D point-mass dynamics（三维质点动力学模型）。

每架飞行器状态定义为：

```text
state = [x, y, z, vx, vy, vz]
```

其中：

- `x, y, z` 为三维位置
- `vx, vy, vz` 为三维速度

MAPPO policy（策略）输出 acceleration command（加速度指令）：

```text
action = [ax, ay, az]
```

仿真器使用固定时间步长 `dt` 更新速度和位置。

基本更新关系：

```text
v(t+1) = v(t) + a(t) * dt
p(t+1) = p(t) + v(t+1) * dt
```

Baseline v1 暂时不建立完整 quadrotor attitude dynamics（四旋翼姿态动力学）。

---

## 5. Physical Constraints（物理约束）

仿真器应支持配置以下物理约束：

- 最大水平速度
- 最大爬升速度
- 最大下降速度
- 最大水平加速度
- 最大垂直加速度
- 最大 jerk（加加速度）

Baseline v1 暂定默认值：

```text
max_horizontal_speed = 20.0 m/s
max_climb_speed = 8.0 m/s
max_descent_speed = 6.0 m/s
max_horizontal_acceleration = 3.0 m/s^2
max_vertical_acceleration = 2.0 m/s^2
max_jerk = 5.0 m/s^3
```

这些参数不得 hard-code（硬编码）在 environment logic（环境逻辑）中。

必须通过 configuration（配置文件）读取。

水平加速度限制应采用模长约束，而不是分别对 x、y 方向单独限幅。

即：

```text
sqrt(ax^2 + ay^2) <= max_horizontal_acceleration
```

Jerk limiting（加加速度限幅）应限制连续两个 simulation step（仿真步）之间的加速度变化。

---

## 6. Pursuer Observation（追击机观测）

每架 pursuer 使用 decentralized observation（分散式观测）。

Baseline v1 优先使用 relative state information（相对状态信息），避免过度依赖绝对坐标。

对于第 `i` 架 pursuer，本地 observation（观测）包括以下部分。

### 6.1 自身速度

```text
[vx_i, vy_i, vz_i]
```

维度：3

### 6.2 相对 evader 位置

```text
[p_e - p_i]
```

维度：3

### 6.3 相对 evader 速度

```text
[v_e - v_i]
```

维度：3

### 6.4 相对 teammate（队友）位置

对于另外两架 pursuers：

```text
[p_j - p_i, p_k - p_i]
```

维度：6

### 6.5 相对 teammate 速度

```text
[v_j - v_i, v_k - v_i]
```

维度：6

因此 Baseline v1 的本地 observation dimension（观测维度）为：

```text
3 + 3 + 3 + 6 + 6 = 21
```

即：

```text
observation_dim = 21
```

所有 observation 的排列顺序必须保持固定并在代码中明确记录。

后续应支持 observation normalization（观测归一化）。

---

## 7. Global State（全局状态）与 MAPPO Critic

MAPPO 采用 CTDE（Centralized Training, Decentralized Execution，集中训练、分散执行）。

Actor（演员网络）只接收各自 local observation（局部观测）。

Centralized Critic（集中式评论家）可以在训练阶段访问完整 global state（全局状态）。

每架飞行器状态为：

```text
[x, y, z, vx, vy, vz]
```

每架 6 维。

4 架飞行器总共：

```text
4 × 6 = 24
```

因此：

```text
global_state_dim = 24
```

Critic 初始接收 24 维 global state。

---

## 8. Action Space（动作空间）

每架 pursuer 输出连续三维加速度指令：

```text
[ax, ay, az]
```

Policy 原始输出范围暂定为：

```text
[-1, 1]^3
```

Environment 负责将 normalized action（归一化动作）映射为真实物理加速度。

完整 action pipeline（动作处理流程）：

```text
policy output
→ acceleration scaling（加速度缩放）
→ horizontal / vertical acceleration constraint（水平/垂直加速度约束）
→ jerk limiting（加加速度限幅）
→ velocity update（速度更新）
→ speed constraint（速度约束）
→ position update（位置更新）
```

---

## 9. Evader Policy（逃逸机策略）

Baseline v1 中，evader 暂时不使用 reinforcement learning（强化学习）。

采用 deterministic / rule-based policy（确定性 / 规则型策略）。

初始推荐使用 repulsive potential-field policy（排斥势场策略）。

核心思想：

> evader 根据所有 pursuers 的相对位置，沿远离威胁的方向移动。

Evader dynamics（逃逸机动力学）同样需要遵守可配置的物理约束。

Evader policy 应独立实现为单独模块，便于后续替换为：

- 不同 scripted strategies（脚本化策略）
- 随机逃逸策略
- trajectory-based strategy（基于轨迹的策略）
- adversarial RL policy（对抗强化学习策略）

---

## 10. Capture Condition（捕获条件）

Baseline v1 使用基于距离的捕获条件。

当任意一架 pursuer 满足：

```text
distance(pursuer_i, evader) < capture_radius
```

则认为团队成功完成 interception（拦截）。

暂定：

```text
capture_radius = 5.0 m
```

一旦成功捕获，episode（回合）立即结束。

更严格的 physical-contact threshold（物理接触阈值）可以在后续实验中单独加入。

---

## 11. Reward Function（奖励函数）

Baseline v1 的 reward 设计应保持简单。

第一版暂时不加入 explicit encirclement reward（显式包围奖励）。

团队 reward 定义为：

```text
R =
    capture_reward
  + progress_reward
  + time_penalty
  + collision_penalty
```

### 11.1 Capture Reward（捕获奖励）

捕获成功时：

```text
capture_reward = +100
```

所有 pursuers 共享同一个 team reward（团队奖励）。

### 11.2 Progress Reward（进展奖励）

使用最近 pursuer 与 evader 的距离变化：

```text
progress_reward =
    k_progress *
    (previous_min_distance - current_min_distance)
```

如果 pursuers 整体正在逼近 evader，则得到正奖励。

该 reward 用于提供 dense feedback（稠密反馈）。

### 11.3 Time Penalty（时间惩罚）

每一个 simulation step：

```text
time_penalty = -0.01
```

用于鼓励更快完成拦截。

### 11.4 Pursuer Collision Penalty（追击机碰撞惩罚）

如果两架 pursuers 之间距离小于 configurable safety distance（可配置安全距离），则施加负奖励。

暂定：

```text
collision_penalty = -5
```

所有 reward coefficient（奖励系数）必须通过配置文件设置。

---

## 12. Episode Termination（回合终止条件）

Episode 在以下情况下结束。

### 12.1 Success（成功）

任意 pursuer 满足 capture condition（捕获条件）。

### 12.2 Timeout（超时）

Episode 达到最大 simulation steps（仿真步数）。

### 12.3 Optional Safety Termination（可选安全终止）

后续可以考虑在以下情况终止：

- 无效状态
- 严重越界
- 数值异常

Baseline v1 不应加入过多复杂终止条件。

---

## 13. MAPPO Architecture（MAPPO网络结构）

Baseline v1 使用 parameter sharing（参数共享）。

3 架 pursuers 共享同一个 Actor network（演员网络）。

### 13.1 Actor

输入：

```text
21-dimensional local observation
```

初始网络结构：

```text
21
→ 128
→ 128
→ continuous 3D action distribution
```

连续动作可以使用 Gaussian policy（高斯策略）。

### 13.2 Critic

输入：

```text
24-dimensional global state
```

初始网络结构：

```text
24
→ 256
→ 256
→ V(s)
```

以上网络规模均为 provisional defaults（暂定默认值）。

---

## 14. Evaluation Metrics（评估指标）

不能只观察 training reward（训练奖励）。

至少记录以下指标：

- Capture Rate（捕获率）
- Average Capture Time（平均捕获时间）
- Minimum Pursuer-Evader Distance（最近敌我距离）
- Pursuer Collision Rate（追击机碰撞率）
- Average Pairwise Pursuer Distance（追击机平均两两距离）
- Episode Reward（回合奖励）

Evaluation（评估）阶段应保存有代表性的 3D trajectories（三维轨迹）。

---

## 15. Behavior Analysis（行为分析）

Baseline v1 需要观察是否出现 emergent cooperative behavior（涌现式协同行为）。

重点关注：

- 三架 pursuers 是否全部从同一方向追击
- 是否只有一架主要追击，另外两架贡献很小
- 是否自发出现 flanking（夹击）
- 是否自发出现 encirclement（包围）
- pursuers 是否频繁相互碰撞
- 是否存在特定 evader trajectory（逃逸轨迹）下必然失败的情况

这些 failure modes（失败模式）将用于后续确定 research direction（研究切入点）。

---

## 16. Baseline Success Criteria（基线成功标准）

Baseline v1 初步成功标准：

1. 3D environment 可以稳定运行。
2. 可以生成 random policy（随机策略）rollout。
3. MAPPO training（训练）能够完整运行，没有实现层面错误。
4. Training reward 出现有意义的学习趋势。
5. Capture Rate 明显优于 random policy baseline。
6. 可以输出并可视化 3D pursuit trajectories（三维追击轨迹）。
7. 可以导出 evaluation metrics（评估指标）。
8. 物理参数可以通过配置文件修改，而无需修改 environment 核心代码。

Baseline v1 不要求达到接近 100% 的拦截成功率。

第一版的核心目标是建立一个稳定、清晰、可复现的 reference system（参考系统）。

---

## 17. Baseline v1 暂不包含的内容

以下内容第一版明确不做：

- obstacles（障碍物）
- 完整四旋翼姿态动力学
- CTBR（总推力与机体角速度）控制
- visual perception（视觉感知）
- observation noise（观测噪声）
- evader trajectory prediction（逃逸轨迹预测）
- explicit encirclement reward（显式包围奖励）
- adversarial RL evader（对抗强化学习逃逸机）
- curriculum learning（课程学习）
- adaptive environment generation（自适应环境生成）
- sim-to-real（仿真到实机）

这些内容应在 baseline 稳定后逐步加入。

---

## 18. 后续潜在扩展方向

可能的研究方向包括：

1. Explicit Encirclement Prior（显式包围先验）
2. Unknown / Adaptive Evader Strategy（未知 / 自适应逃逸策略）
3. Adversarial Evader Learning（对抗逃逸者学习）
4. 3D Obstacle Environment（三维障碍环境）
5. Curriculum Learning（课程学习）
6. Prediction under Occlusion（遮挡条件下的轨迹预测）
7. Observation Noise and Uncertainty（观测噪声与不确定性）
8. More Realistic Flight Dynamics（更真实的飞行动力学）
9. Sim-to-Real Transfer（仿真到实机迁移）

---

## 19. Development Principle（开发原则）

Baseline v1 应始终保持：

- minimal（最小化）
- understandable（容易理解）
- configurable（可配置）
- reproducible（可复现）

除非某项复杂功能是验证 baseline 所必需，或后续实验明确要求，否则不要提前加入。