# 多机协同拦截 MAPPO Baseline v1 技术规格

## 1. 当前任务与目的

**Three cooperative UAV interceptors defend a fixed point target against one attacking UAV.**

当前 Baseline v1 是 3D point-defense（点防御）任务：三架我方拦截机保护固定基地，
一架敌机从距基地约 500 m 的固定位置被发现，并尝试进入基地的有效攻击区域。
我方必须在敌机攻击成功之前完成拦截。当前默认任务不再是追逐无限逃跑的 evader。

Baseline 的目标是建立简单、清晰、可配置、可复现的环境、MAPPO、评估和可视化流程，
并记录失败模式。除导师确认的任务定义外，具体控制增益、奖励与物理数值仍是
provisional defaults（暂定默认值）；本次只将攻击机制导改为 Direct Attacker，
不改变既定点防御任务，不进行性能调参或正式 100k 训练。

## 2. 场景、基地与固定初始化

场景保留 `3A1C`：三架同构 pursuers（我方拦截机）与一架 rule-based attacker（敌机），
三维连续空间，无障碍物。我方使用共享 actor 的 MAPPO；敌机不使用强化学习。

所有参数来自 `configs/baseline_3a1c.yaml`：

```yaml
mission:
  base_position: [0.0, 0.0, 0.0]
  detection_radius: 500.0
  attack_radius: 10.0
initialization:
  pursuer_positions:
    - [1.5, 0.0, 1.0]
    - [-0.75, 1.25, 1.0]
    - [-0.75, -1.25, 1.0]
  attacker_position: [497.49371855331, 0.0, 50.0]
```

基地是三维点，攻击区域是以基地为球心、半径 10 m 的三维球体。
`detection_radius = 500 m` 是导师确认的本轮初始探测距离，不是雷达/传感器模型，
也不额外形成观测裁剪或任务终止边界。当前固定敌机初始方向、零初速度；所有 reset
重置加速度历史为零，不引入随机初始化。

三架我方机距原点均不超过 2 m，最大约 1.8028 m；最小两两距离为 2.5 m，
大于保留的 `pursuer_safety_distance = 2 m`。队形不重合、关于 x-z 平面对称，
无需修改安全距离。敌机距基地约 500 m，高度 50 m。

## 3. 数值工作空间与边界

```yaml
environment:
  dt: 0.1
  max_episode_steps: 500
  bounds:
    low: [-750.0, -750.0, -750.0]
    high: [750.0, 750.0, 750.0]
```

工作空间仅是 numerical safety（数值保护），覆盖 500 m 探测球并留有余量，
不是“战场墙”，也不是地面/高度模型。允许负 z 不表示增加了地形或地面碰撞。
所有飞行器仍在积分后 hard position clip，保留速度，不反弹，不增加边界奖励或终止。

默认 `attacker.policy.mode: direct` **关闭 soft boundary steering**，显式提供任何
`boundary_*` 参数会报错，避免叠加任务未授权的制导项。Pursuer 不使用 boundary-aware steering。
此前的连续 soft boundary 模块保留在旧 GoalDirected/evader 实现中，仅供无 `mode` 的历史
配置/快照复现；不是当前 Direct Attacker 的可选制导功能。

`boundary_clip_count` 按“某架飞机某一步实际越界并裁剪”计数；同帧多个轴只记一次。
`boundary_clip_counts` 顺序为 `[P0,P1,P2,attacker]`，`boundary_clipped` 是当前步布尔数组，
reset 清零；返回数组均为副本。停留在 exact boundary 而没有再次越界不增加计数。

## 4. Dynamics 与物理约束

保留简化 3D point-mass dynamics，每机状态为 `[x,y,z,vx,vy,vz]`：

```text
v(t+1) = v(t) + a(t) * dt
p(t+1) = p(t) + v(t+1) * dt
```

Pursuer 仍先将归一化动作缩放为物理加速度；Direct Attacker 则直接产生物理加速度指令，
不经过 `scale_action`。两者之后均使用原 dynamics：水平模长/垂直加速度约束 →
三维模长 jerk 限制 → 速度更新 → 水平/爬升/下降速度约束 → 位置更新 → 数值 hard clip。

双方现有物理性能均保持不变，分别从 `pursuer` / `attacker` YAML 节读取：

| 参数 | 默认值 |
| --- | --- |
| 最大水平速度 | 20 m/s |
| 最大爬升 / 下降速度 | 8 / 6 m/s |
| 最大水平 / 垂直加速度 | 3 / 2 m/s² |
| 最大三维 jerk | 5 m/s³ |
| `dt` / 最大回合步数 | 0.1 s / 500 |

水平加速度满足 `sqrt(ax²+ay²) <= max_horizontal_acceleration`，不是按 x/y 分别裁剪。
Jerk 约束针对前后两步 applied acceleration command；限速时实际 `dv/dt` 可与其不同。
不建立完整四旋翼姿态动力学。

## 5. Rule-based Direct Attacker policy

当前默认使用 `envs/attacker_policy.py` 的 `DirectAttackerPolicy`：从约 500 m 固定
探测位置出发，以基地为唯一目标，不规避我方机。使用 desired-velocity tracking，
而非把位置吸引方向直接作为加速度。令敌机位置为 `p_a`，速度为 `v_a`：

```text
goal_vector = base_position - p_a
goal_direction = goal_vector / ||goal_vector||     if ||goal_vector|| > distance_epsilon
goal_direction = 0                                otherwise
desired_velocity = attack_speed * goal_direction
physical_acceleration_command = velocity_gain * (desired_velocity - v_a)
```

Policy 返回形状 `(3,)` 的原始物理加速度指令，不再归一化，不调用 `scale_action`，
不自行实现加速度、jerk 或速度限幅；全部由原 `envs/dynamics.py` 统一处理。
因此速度偏离目标方向时，指令会主动修正横向速度。输入和配置必须有限且 shape 正确。

```yaml
attacker:
  policy:
    mode: direct
    attack_speed: 20.0
    velocity_gain: 1.0
    distance_epsilon: 0.000001
```

`attack_speed` 是期望三维速度模长，单位 m/s；`velocity_gain` 单位 s^-1。
本轮选取 `1 s^-1`，对应未受限一阶速度跟踪时间常数 1 s，易于解释；实际加速时间仍受
原加速度和 jerk 限制。三个数值参数都要求有限且为正，均从 YAML 读取。
这是单组 provisional defaults，未搜索多组速度或增益。

Direct Attacker 不使用 pursuer repulsion、soft boundary steering、随机机动或预测。
既有 pursuer 状态参数仅为环境调用接口兼容，不影响制导结果。
目标是穿入 10 m 球，不是停在基地；不增加 arrival braking、stop-at-origin、终端悬停或 PID
位置保持。距离不超过 epsilon 时的零目标方向仅为分母保护，正常回合已在 `<10 m` 时结束。

历史 `GoalDirectedAttackerPolicy` 使用目标方向加速度与规避项。历史无近威胁对照曾出现
最近距基地约 19.28 m 后过冲绕行；该现象属于旧策略，不能当作当前 Direct Attacker 的结果。
旧模块仅为无 `mode` 历史配置/快照复现而保留，本轮没有新增 `evasive` mode。

## 6. Pursuer observation 与 global state

Local observation 保持 21D，不追加基地位置、预测、噪声或任务标志：

| 顺序 | 内容 | 维度 |
| --- | --- | --- |
| 1 | 自身速度 | 3 |
| 2 | 相对 attacker 位置 | 3 |
| 3 | 相对 attacker 速度 | 3 |
| 4 | 两名 teammate 的相对位置 | 6 |
| 5 | 两名 teammate 的相对速度 | 6 |

队友按固定编号升序排列，两个相对位置在两个相对速度之前。环境输出物理量，不静默归一化。

Centralized global state 保持 24D：`[P0_state,P1_state,P2_state,attacker_state]`，
每机 6D。基地固定为原点，critic 可从 attacker 绝对位置获得其相对基地位置。
Actor 仅使用自身 21D local observation，不读取 global state。

已知信息限制：local observation 只有速度和相对状态，对所有飞机共同平移不敏感，
因此不能唯一恢复当前位置相对基地的方位/距离。固定初始化可能提供先验，但不解决一般
可观测性问题；本轮明确保留该限制，不扩维。Jerk 所需的加速度历史也未进入既定
21D/24D 输入，因此这些输入不是完整 Markov 状态。

## 7. Action

每架 pursuer 输出 `[-1,1]^3` 的 normalized acceleration，环境映射为物理加速度。
团队 action shape 为 `(3,3)`，有限越界值先裁剪，NaN/Inf 报错。没有增加航向、角色或
拦截点动作，也未修改 pursuer 的动作处理方式。

## 8. 终止条件与同时触发规则

四机从同一个旧状态同步积分后，用更新后的同一时刻位置判定：

1. **Interception Success**：任一我方机与 attacker 距离严格 `< capture_radius = 5 m`。
   `captured = interception_success = True`，`terminated = True`。
2. **Enemy Attack Success**：未捕获且 attacker 到基地距离严格 `< attack_radius = 10 m`。
   `attacker_reached_base = enemy_attack_success = interception_failure = True`，终止。
3. **Timeout**：上述两种 terminal outcome 都未发生且达到 500 步，`truncated = True`。

同一步进入两种距离范围时，**capture 优先**：不同时判敌方进攻成功或施加失败惩罚。
`attacker_in_attack_region` 独立记录几何进入攻击球的事实，即使该步捕获优先也可为 True。
Exactly 5 m / 10 m 不触发对应严格小于条件。Terminal outcome 优先于同一步超时。
当前没有步内连续接触检测；结束后必须 reset 才能继续 step。

`info` 至少包含：`captured`、`interception_success`、`attacker_reached_base`、
`enemy_attack_success`、`interception_failure`、`attacker_in_attack_region`、
`distance_to_base`、`mission_enabled`、`min_distance`、`collision_pairs`、`step_count`、
`global_state` 和边界诊断。`collision_pairs` 是当步不安全追击机对的数量。

## 9. Reward：保留既有点防御定义

本轮保持共享团队回报及所有现有点防御奖励项和系数不变：

```text
R = capture + attack_failure + progress + time + collision
progress = progress_coefficient * (previous_min_distance - current_min_distance)
```

| YAML 系数 | 当前默认值 / 触发 |
| --- | --- |
| `capture_reward` | +100，仅拦截成功 |
| `attack_failure_penalty` | -100，仅敌方攻击成功；必须为负 |
| `progress_coefficient` | 1，原最近敌我距离进展 |
| `time_penalty` | -0.01，每步 |
| `collision_penalty` | -5，任意我方机对间距 `< 2 m` 时每步团队扣一次 |

三个 pursuer 返回相同 reward，团队回报累计一次，不乘三。终止步仍计算原有进展、时间与
碰撞项，所以终止步完整 reward 不一定恰好为 ±100。
`info["reward_components"]` 在 step 中给出各项。没有新增包围、角度、角色、预测或拦截点奖励。

## 10. MAPPO 与训练

保持 CTDE、parameter sharing、网络和 PPO 设定：

- Actor：21 → 128(Tanh) → 128(Tanh) → 3D Gaussian mean，可学习共享 log_std，
  使用 tanh 压缩动作；确定性评估使用 tanh(mean)。
- Critic：24 → 256(Tanh) → 256(Tanh) → 一个团队 value。
- 超参数统一来自 `configs/mappo_baseline.yaml`，本次不为新任务调优。
- 两种 terminal outcome 均不 bootstrap；timeout 从 reset 前最终状态 bootstrap，
  但切断跨回合 GAE。没有修改 rollout buffer 或网络结构。

本轮只执行极短 MAPPO smoke training 验证兼容性，不执行 100k 正式实验。
流程运行成功不能证明学会任务或学到协同。

## 11. Evaluation 与 3D visualization

评估必须分别报告三种任务结局，而非只报告 captured：

- `interception_success_rate`；
- `enemy_attack_success_rate`；
- `timeout_rate`；
- 平均 attacker-base terminal distance；
- 成功回合的平均 interception time（无成功时 null）；
- episode reward/length、最小敌我距离、我方碰撞统计、平均两两距离；
- hard boundary clipping 诊断。

保留原 capture rate/time 名称作为兼容别名，不能用它们代替完整任务结果。
最小敌我距离包括初始状态；碰撞步数从 step 后开始累计。
固定初始化下 deterministic evaluation 重复相同轨迹，改变 seed 不构成泛化测试。

3D 图绘制基地、10 m attack sphere、三架 pursuer 与 attacker 的轨迹、起点与终点；
标题显示 Interception Success / Enemy Attack Success / Timeout、步数、最小敌我距离与
最终敌机到基地距离。原始轨迹保存初始状态和全部更新后的状态，便于独立分析。

另提供 `scripts.sanity_direct_attacker` 单独积分攻击机的无 pursuer 干扰诊断。
使用当前 YAML 固定初态、原 dynamics、原 hard clip 和严格 `<10 m` 判定，检查完整回合
基地距离下降、进入攻击球、无绕圈/长振荡/timeout、无 NaN/Inf；记录初始/最小基地距离、
进入时间、进入速度和 hard clipping，保存原始轨迹、距离曲线和只含攻击机的 3D 图。
运行顺序为全量 pytest → sanity check → 检查结果与 3D 图 → random rollout →
短 MAPPO smoke training → checkpoint 加载、evaluate 与 visualization；不运行正式长训练。

## 12. 历史兼容与验证

历史无 `mission`、使用 `evader` 字段的配置/旧 checkpoint 仍可按原任务加载，
保留 `RepulsiveEvaderPolicy` 与已有 boundary 实现，不将纯逃跑设为当前默认行为。
含 mission 但没有 `attacker.policy.mode` 的旧配置继续使用 `GoalDirectedAttackerPolicy`，
用于复现上一版点防御快照；当前 YAML 必须显式选择 `mode: direct`。
代码继续使用 `PursuitEnv` 类名和部分 `evader` 兼容别名，避免无关的大规模重命名。

历史追逃环境的边界修复曾将 workspace 扩为 `[-300,-300,0]` 至 `[300,300,150]`，
并启用 margin 50/20、strength 2 的 steering。该历史对照中的硬裁剪减少不代表新任务成绩，
详见 [BOUNDARY_MECHANISM_REPORT.md](experiments/BOUNDARY_MECHANISM_REPORT.md)。
旧 100k 训练和视频也属于旧任务，见
[MAPPO_100K_BASELINE_REPORT.md](experiments/MAPPO_100K_BASELINE_REPORT.md)，
不能与新点防御任务直接比较学习效果。

上一版点防御任务与旧 GoalDirected 制导的实际结果见
[POINT_DEFENSE_TASK_REPORT.md](experiments/POINT_DEFENSE_TASK_REPORT.md)。
当前 Direct Attacker 的 pytest、无干扰 sanity check、random rollout、3D 图与 smoke training
实际结果见 [DIRECT_ATTACKER_BASELINE_REPORT.md](experiments/DIRECT_ATTACKER_BASELINE_REPORT.md)。
默认 `pytest.ini` 只收集 `tests/`，避免历史输出中的源码快照被重复收集。

## 13. 当前不包含与建模限制

本阶段不加入随机初始化、障碍物、观测噪声、显式包围奖励、trajectory prediction、
curriculum、对抗 RL attacker、真实姿态/地面碰撞模型或 sim-to-real。

当前主要限制包括：固定方向与固定初始化、actor 的基地关系不可完全观测、
加速度历史未纳入输入、离散步接触检测、未建模空气阻力的质点动力学、数值裁剪仍保留速度、
Direct Attacker 不覆盖敌机规避/机动能力且跟踪增益仍是暂定值。默认初态 sanity check
不能证明任意初速度、任意几何都能有限时间到达。保留原 progress reward 也意味着敌机自行接近会产生正进展，
不能仅据 reward 上升判断我方学会保护基地。

后续研究应先根据真实任务结局与轨迹分析问题，再明确授权修改研究设定。
