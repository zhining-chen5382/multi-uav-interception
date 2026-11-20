# 多机协同拦截 MAPPO Baseline v1 技术规格

## 独立单机正式实验入口，2026-11-16

用户确认的1v1单机PPO实验采用 [正式训练规范 v1.0](FORMAL_TRAINING_SPEC_V1.md)。环境配置为 `configs/interception_1v1.yaml`，初始300k入口为 `scripts/run_formal_experiment.py`。用户后续要求各Seed累计500k，续训采用 `configs/ppo_formal_v1_500k.yaml` 与 `scripts/run_formal_500k.py`，完整恢复各自300k final，原300k学习率履历保留，随后维持终值。

其16维字段依次为自身速度3、applied acceleration3、敌机相对位置3、敌机相对速度3、敌机相对基地位置3、剩余时间比例1，固定除数依次为 `[20,20,8,3,3,2,1500,1500,1500,40,40,14,500,500,500,1]`。加速度读取经过jerk限制的环境加速度状态，并作为下一步jerk限制的历史状态。训练、验证和测试使用同一参数，仅做逐维除法，越界记录日志。

下文记录3A1C MAPPO参考系统及历史实验定义。

独立1v1 Reward A/B实验使用`configs/ppo_reward_ab.yaml`，B组在原四项上增加λ=20m、β=100的历史最小距离差分项，1001–1003各从零300k。其正式状态、恢复及对照范围见[正式训练规范第9节](FORMAL_TRAINING_SPEC_V1.md)。默认3A1C和原1v1 YAML继续分别作为已有参考配置。

用户指定的独立k=0.1线段成本实验使用`configs/ppo_line_cost_k0p1.yaml`，在B组奖励基础上累加三维基地—敌机闭线段距离时间成本，采用同三个Seed各300k。定义与交付见[正式训练规范第10节](FORMAL_TRAINING_SPEC_V1.md)。

2026-11-18授权独立主动接近near配比训练，采用冻结我方本步旧位置的反事实归因。β100/k0.01、β200/k0.01两组各三个Seed从零300k，其余正式协议沿用。配置为`configs/ppo_active_near_beta100.yaml`与`configs/ppo_active_near_beta200.yaml`，定义见[正式训练规范第11节](FORMAL_TRAINING_SPEC_V1.md)。

## 1. 当前任务与目的

**Three cooperative UAV interceptors defend a fixed point target against one attacking UAV.**

当前 Baseline v1 是 3D point-defense（点防御）任务：三架我方拦截机保护固定基地，
一架敌机从距基地约 500 m 的固定位置被发现，并尝试进入基地的有效攻击区域。
我方必须在敌机攻击成功之前完成拦截。当前默认任务不再是追逐无限逃跑的 evader。

Baseline 的目标是建立简单、清晰、可配置、可复现的环境、MAPPO、评估和可视化流程，
并记录失败模式。除导师确认的任务定义外，具体控制增益、奖励与物理数值仍是
provisional defaults（暂定默认值）。当前默认奖励已修复距离进展的被动接近收益，并使用按秒计的
存活惩罚；Direct Attacker、初始化、21D/24D 输入、动力学和 MAPPO 配置保持原定义。
实验方案编号为 v05：v05-r01 完成 20k 诊断，后续独立的 v05-r02 从零完成 100k 并续至 200k。
后续 v06-r01 在相同环境、奖励与 PPO 参数下，只开启 Actor/Critic 固定输入缩放，完成独立 20k 对照。
随后经用户授权，从其20k final完整恢复并补训至总计100k，仍属同一run，原20k产物保持冻结。
最新统一对比已将v05-r02从200k、v06-r01从100k分别续至总300k，均保留原任务、网络与PPO参数；
区别仍为Actor/Critic输入缩放。四组best/final各100回合确定性评估均未捕获。
实际训练、行为差异和调度限制见
[V05_V06_300K_COMPARISON_REPORT.md](experiments/V05_V06_300K_COMPARISON_REPORT.md)。
原始 `mappo_baseline.yaml` 仍是未缩放参考；新实验配置为 `mappo_normalized.yaml`。
最新独立v07-r01按用户确认α=0.01将active-progress替换成最近距离时间成本，关闭输入缩放，
从零实际完成300k；训练捕获9/1066，best/final确定性均0/100，60个保存策略筛查均未捕获。
独立配置与数学定义见第10节，完整结果见
[ABSOLUTE_DISTANCE_300K_REPORT.md](experiments/ABSOLUTE_DISTANCE_300K_REPORT.md)。默认v05参考YAML保持不变。
最新v08仅将v07捕获奖励从100提高至500，从零完成300k；92160步模型固定初态确定性100/100捕获，
17.3s、最近3.525m，但原生best与300k final均失败，出现后期退化。
成功模型独立登记，详见[CAPTURE500_300K_REPORT.md](experiments/CAPTURE500_300K_REPORT.md)，
100次固定初态重复不代表泛化。v08配置与公式见第10节。
v09 新增独立的来袭方位课程入口：±10° 从零训练，依次向 ±30°、±60°、±90° 扩展，
只改变每回合敌机初始方位分布，其余物理任务、奖励、输入、网络和 PPO 沿用 v08。
本节不将流程实现或工程测试视为课程成功；测试和正式运行状态见
[V09_DIRECTION_CURRICULUM_IMPLEMENTATION.md](experiments/V09_DIRECTION_CURRICULUM_IMPLEMENTATION.md)。
版本、运行时间及模型选择标准见 [实验索引](experiments/EXPERIMENT_INDEX.md)。这些实验方案编号与本规格的 Baseline v1 名称独立。

历史输出已按实验版本整理；本规格的路径示例使用迁移后位置。冻结快照、metadata、日志及 checkpoint 中的旧路径不改写，现址见 [目录迁移表](experiments/output_path_migration_20261108.json)。

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
`detection_radius = 500 m` 是保留的固定初始探测距离，不是雷达/传感器模型，
也不额外形成观测裁剪或任务终止边界。默认参考配置固定敌机初始方向、零初速度；所有 reset
重置加速度历史为零。v09 仅通过独立子类改变敌机初始方位，不改默认环境或其他初始化量。

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
保留的 `1 s^-1` 对应未受限一阶速度跟踪时间常数 1 s，易于解释；实际加速时间仍受
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

v06 的归一化不改变上述环境输出或信息内容，仅在 Actor/Critic 的 forward 内除以固定尺度：
Actor 相对位置用三轴 workspace extent（当前均 1500 m），Critic 绝对位置用三轴最大绝对边界
（当前均 750 m）；自身速度用对应速度上限，相对速度用双方分量速度上限之和。
尺度由环境配置推导，作为 checkpoint buffer 保存；没有运行统计、输入裁剪或 reward 缩放。
`configs/mappo_normalized.yaml` 显式开启两个开关；原始 `configs/mappo_baseline.yaml` 保持关闭。

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

## 9. Reward：Active Progress 与物理时间存活惩罚

原 progress 只计算最近敌我距离缩短，敌机自行飞向基地也会给 pursuers 正反馈。
当前按指定公式扣除敌机同期向基地的净进展，其他 terminal/collision 定义保持不变：

```text
pursuer_attacker_progress = d_PE_previous - d_PE_current
attacker_base_progress = d_EB_previous - d_EB_current
active_progress = pursuer_attacker_progress - attacker_base_progress
active_progress_reward = active_progress_coefficient * active_progress
survival = survival_penalty_per_second * dt
R = active_progress_reward + survival + collision + capture + attack_failure
```

| YAML 系数 | 当前默认值 / 触发 |
| --- | --- |
| `capture_reward` | +100，仅拦截成功 |
| `attack_failure_penalty` | -100，仅敌方攻击成功；必须为负 |
| `active_progress_coefficient` | 1，乘上述 active 距离差分 |
| `survival_penalty_per_second` | -0.2 / s，每步乘实际 `dt` |
| `collision_penalty` | -5，任意我方机对间距 `< 2 m` 时每步团队扣一次 |

距离均为三维 Euclidean distance；`d_PE` 取三个 pursuer 中最近者，`d_EB` 为敌机到固定基地。
前后值对应同一步同步积分前与积分后的状态。敌机向基地前进 2m、敌我距离也只缩短 2m
时 active=0；敌我距离缩短 5m 时 active=+3m，只缩短 1m 时 active=-1m。
该定义是距离差分代理，不是在任意几何下对我方自主运动贡献的精确归因。

默认 YAML 删除旧 `progress_coefficient` / `time_penalty`，不叠加两套项。
`dt=0.1s` 时 survival 每步 -0.02，28.2s 累计 -5.64；终止步仍代表已过去的 `dt`，同样计费。
三个 pursuer 返回相同 reward，团队回报累计一次，不乘三；终止步还包括 active/survival/
collision，所以完整 reward 不一定恰好为 ±100。没有新增包围、角度、角色、预测或拦截点奖励。

`info["reward_components"]` 的原始诊断为 `pursuer_attacker_progress`、
`attacker_base_progress`、`active_progress`；已加权项为 `active_progress_reward`、`survival`、
`collision`、`capture`、`attack_failure`。**总 reward 只累计五个已加权项，不加原始诊断值。**
`info["episode_reward_components"]` 返回这五项的回合累计副本，reset 清零。

训练和评估逐回合 metrics 增加 `cumulative_active_progress_reward`、
`cumulative_survival_penalty`、`cumulative_collision_penalty`、`terminal_reward`，后者为
capture 与 attack_failure 的和。Evaluation summary 保存对应 `average_` 均值。
轨迹 NPZ 保存 `reward_component_names` 与 `(episode_length, num_components)` 的
`reward_components` 矩阵；该矩阵每行对应一次 transition，位置/time 另外包含 reset 初态。

Active 项回合累计可写为 `(d_PE0-d_PET)-(d_EB0-d_EBT)`。在当前默认固定初态下
`d_PE0<d_EB0=500m`；敌方攻击成功时 `d_EBT<10m` 且 `d_PET>=5m`，故 active 累计小于 5，
默认系数为1，加上终局 -100 及非正 survival/collision，失败回报小于 -95。
这是当前初态和系数下的界，不是改变初态、半径、系数后的普遍符号保证，也不证明所有
reward hacking 均已消除。仍必须检查任务成功、轨迹和各分量，而不能仅凭回报判断学习。

## 10. MAPPO 与训练

### v07 独立奖励对照：按物理时间累计最近绝对距离成本

用户确认的本轮实验使用 `configs/baseline_absolute_distance.yaml` 与
`configs/mappo_absolute_distance.yaml`。原 `baseline_3a1c.yaml` 的 v05 active-progress
参考配置保留，不会自动把旧 checkpoint 改成新奖励。

令 `d_(t+1) = min_i ||p_i(t+1)-p_a(t+1)||_2`，采用更新后距离的右端点矩形积分：

```text
distance_time_integral = d_(t+1) * dt
absolute_distance_cost = -0.01 * distance_time_integral
R_t = absolute_distance_cost - 0.2 * dt
      - 5 * I(any pursuer pair distance < 2 m)
      + 100 * I(captured)
      - 100 * I(enemy attack success and not captured)
```

该项替换 active-progress，不与旧 progress 或 active-progress 叠加；捕获优先规则、
终止、碰撞、初始化、动作、21D/24D 输入与动力学不变。系数来自 YAML 的
`reward.distance_cost_coefficient: 0.01`，单位为分/(m·s)，即距离100m持续1s扣1分。
每步使用三机中的最近者，未增加三机平均距离或包围项。

Actor/Critic 输入缩放均关闭，`reward_scale=1`，原 PPO advantage normalization 保留。
新奖励需要从零训练，不能把旧 value/optimizer 的恢复当作同一奖励实验。
正式预算为单次 seed0、300000环境步；训练 best 的窗口选取规则保持原样。

`info["reward_components"]` 新增两项原始诊断 `distance_to_nearest_pursuer` 和
`distance_time_integral`，不参与奖励求和；五个加权项为 `absolute_distance_cost`、
`survival`、`collision`、`capture`、`attack_failure`。累计与 reset 仍按既有接口处理。
CSV 新增 `cumulative_absolute_distance_cost`，旧奖励快照该字段为 null；v07 的
`cumulative_active_progress_reward` 为 null。评估 summary 保存对应 `average_` 均值。

该目标对整个接近过程计费，不只看端点，但最近距离代理仍不能直接保证团队分工，
也不能保证某条最近距离更小的失败轨迹回报一定最高。新旧奖励的原始回报不可直接比较；
应比较相同任务的捕获率、距离和轨迹，或明确标注对旧轨迹按新公式离线重算。

### v08 独立对照：捕获奖励 +500

按用户明确要求，`configs/baseline_capture500.yaml`从v07环境配置复制，唯一研究参数变化为
`reward.capture_reward: 100.0 → 500.0`。实际公式为：

```text
R_t = -0.01 * d_(t+1) * dt - 0.2 * dt
      - 5 * I(any pursuer pair distance < 2 m)
      + 500 * I(captured)
      - 100 * I(enemy attack success and not captured)
```

`configs/mappo_capture500.yaml`保持v07的seed0、从零连续300k预算、网络、初始化、PPO、
关闭Actor/Critic输入缩放和reward_scale=1；仅指定新环境配置和`outputs/v08_捕获奖励500/`输出目录。
旧v07配置、权重和结果保留。捕获条件仍为更新后最近距离严格小于5m，未降低任务难度。
提高捕获项扩大同一成功轨迹的无折扣回报400分，但不保证任意成功回合总回报都为正，
也不保证确定性策略能成功；需分别检查真实训练捕获、确定性评估和轨迹。

保持 CTDE、parameter sharing、网络和 PPO 设定：

- Actor：21 → 128(Tanh) → 128(Tanh) → 3D Gaussian mean，可学习共享 log_std，
  使用 tanh 压缩动作；确定性评估使用 tanh(mean)。
- Critic：24 → 256(Tanh) → 256(Tanh) → 一个团队 value。
- 超参数统一来自 `configs/mappo_baseline.yaml`，本次不为新任务调优。
- 两种 terminal outcome 均不 bootstrap；timeout 从 reset 前最终状态 bootstrap，
  但切断跨回合 GAE。没有修改 rollout buffer 或网络结构。

历史 v05-r01 在 reward sanity 通过后，从零执行一次 20000 环境步诊断；仅通过 CLI 指定步数
和专用输出目录，不修改网络、学习率、normalization 或 PPO 参数。后续 v05-r02 的 100k→200k
运行另有完整记录。两次实验均保留原 PPO/GAE 与训练 best 选取规则。
流程运行成功或回报改善不能证明学会任务或学到协同。

v06-r01 采用 v05-r01 相同的 20000 步、seed 0，从零执行一次输入缩放联合对照。
实跑 40 updates、70 完整回合，训练捕获 0；best/final 各 10 回合确定性评估均无捕获。
缩放缓解初始 Tanh 饱和，但本次没有显示拦截能力改善；不能据此分别归因 Actor 或 Critic，
也不能把单种子固定场景结果推广为一般结论。详见
[NORMALIZATION_20K_REPORT.md](experiments/NORMALIZATION_20K_REPORT.md)。

后续v06-r01从20k final续至总计100000步，没有改变上述研究参数；使用独立
`configs/mappo_normalized_100k.yaml`，累计197次更新、354完整回合、2次随机训练捕获。
100k best/final各50回合确定性评估仍0%捕获；final最近42.891m，较20k final改善，但尚未学会任务。
20k最后32步已更新，续训最后128步更新，因此与一次不间断100k的196次更新日程不同。
实现恢复和保留历史结果的封装仅在本次输出目录内，未给通用训练CLI添加resume功能。
数据和限制见 [NORMALIZATION_100K_REPORT.md](experiments/NORMALIZATION_100K_REPORT.md)。

### v09 独立来袭方位课程

配置 `configs/mappo_direction_curriculum.yaml` 仍引用 `baseline_capture500.yaml`，
网络、PPO、观测/状态缩放关闭与 `reward_scale=1` 均与 v08 相同。
独立入口 `scripts/train_direction_curriculum.py` 复用原 `MAPPOAgent`、`RolloutBuffer`、
训练回合统计和确定性 `run_episode`；原 `scripts/train_mappo.py` 的行为不变。

令基地为 `b`、原始敌机位置为 `p0`，每回合采样 `θ ~ U[-A,A]`：

```text
p_attacker_initial.xy = b.xy + Rz(θ) * (p0.xy - b.xy)
p_attacker_initial.z = p0.z
A ∈ [10°, 30°, 60°, 90°]，按顺序执行
```

`utils.direction_curriculum.DirectionCurriculumEnv` 每次均从原始配置和参考初态计算，
不累计旋转；保持水平半径、三维基地距离、高度及全部初速度、我方初始位置不变。
训练采用独立 NumPy Generator 连续均匀采样，阶段 seed 为 `seed + stage_index*10000`
（stage_index 从 0 开始）。Direct Attacker 继续直接飞向原基地，不增加规避策略。

每阶段最多 100000 实际训练环境步，在每 10000 步完成 PPO 更新后，对固定 100 个等间距、
包含两端的阶段角度做确定性评估。该数量读取 `curriculum.evaluation_episodes=100`；新配置的
通用 `evaluation.episodes` 也记为 100，普通固定场景评估不能代替课程评估。评估使用独立环境，结束后恢复网络 train/eval 模式
与 Python/NumPy/Torch/CUDA RNG，不推进训练环境、训练角度采样器或优化器。

连续两次当前阶段捕获率 ≥80% 才晋级；任何未达标评估会清零连续次数。
best-eval 按以下顺序严格字典序比较：捕获率高、敌方攻击成功率低、成功回合平均捕获时间短、
平均回合最小敌我距离小、碰撞回合率低。完全并列保留较早策略，不使用 reward 排名。
无捕获时平均捕获时间保留 null；内部比较使用有限占位值，不输出 Inf/NaN。
后续阶段每个评估点另测前一阶段的同一固定角度集，仅监控遗忘，不影响 best 或晋级。

±10° 从零构造 Actor/Critic 和 Adam。后续阶段加载前阶段历史 `best_eval.pt`，
恢复 Actor、Critic、两套 Adam 全部状态（含 step、exp_avg、exp_avg_sq）及所选权重的
全局 RNG；不加载阶段末 `last.pt`。阶段升级重建环境、buffer、partial episode 与 gate，
不跨阶段续接 GAE。实际消耗总步数、updates 和完整回合计数不因选中较早 best 而回退，
另记录父 checkpoint 的 SHA256、来源步数和继承关系。

为精确在 10k 评估，512 容量 rollout 在边界切短（每完整 10k 区间为 19×512+272）。
正常计算下一状态 value 并 bootstrap；真实 terminal 不 bootstrap，timeout 仍按原规则处理。
切分只影响调度，没有修改 PPO/GAE、网络或物理任务定义。评估权重立即保存为
`evaluated_*.pt`，改进时原样复制为 `best_eval.pt`，确保指标与权重对应。

本阶段用完 100k 仍未连续两次达标即停止，不跳过阶段；若第二次达标恰在 100k，则允许晋级。
四阶段总预算上限 400k，实际消耗由 gate 决定。每阶段保存 `initial.pt`、周期 checkpoint、
`evaluated_*.pt`、`best_eval.pt` 和 `last.pt`，以及训练/评估 CSV、固定角度和阶段摘要。
默认输出 `outputs/v09_来袭方向课程/v09-r01_20261109_seed000/`；非空目录拒绝覆盖。
已有阶段间继承不等于任意中断自动恢复。新增的预算结束续训入口说明如下。
具体命令、文件职责和验证状态见 [v09 实现记录](experiments/V09_DIRECTION_CURRICULUM_IMPLEMENTATION.md)。

### v09 同阶段续训至累计 300k

在原 ±10° 阶段因 100k 预算停止后，用户授权从该阶段末 `last.pt` 继续到累计 300000 步，
不从 30k 历史 best 重启，也不重新初始化。独立配置为 `configs/mappo_direction_curriculum_300k.yaml`，
输出为同一 v09-r01 的 `continuation_300k/`，原 100k 文件保持冻结。

`continuation.source_output` 指向已正常因预算结束的源输出。只允许预算、输出路径和续训来源字段变化，
环境、网络、PPO、观测/状态缩放、reward scale、种子和评估协议均需与源快照一致。
本次 `curriculum.stage_step_budgets=[300000,100000,100000,100000]` 覆盖阶段预算默认值，
`training.total_env_steps=300000` 为所有已执行阶段共享的实际累计上限；提前晋级也不能超训。
未达标阶段预算耗尽仍停止；连续两次 80% 的晋级要求、历史 best-eval 排序及 previous 监控不变。

同阶段续训恢复 last 的 Actor、Critic、双 Adam、全局 RNG、环境状态、加速度历史、
未结束回合及加权奖励累计、最近窗口、当前角度、双采样 RNG、streak 与计数。
历史 best 作为选模候选原样保留。新 `resume_initial.pt` 便于核验恢复来源；累计 CSV 先复制历史再追加，
不覆盖原记录。新版 checkpoint 保存完整环境内部快照。
旧 checkpoint 缺少的奖励分量从 partial episode 恢复，缺失的边界计数只有在速度位移界和
Direct Attacker 单机重放能够证明从未 clipping 时才恢复为零，否则拒绝。

当前只支持具有结束 summary 的预算停止续训，尚未实现任意进程崩溃/任意 update 的自动恢复。
本次短测确认分段与连续训练的模型、双 Adam、RNG 和环境状态逐项一致；完整测试 354 项通过。

## 11. Evaluation 与 3D visualization

### 新独立 v09-r02：从固定 0° 起步的课程

当前默认课程入口使用 `configs/mappo_direction_curriculum_zero_start.yaml`，按
`0° → ±2° → ±5° → ±10° → ±30° → ±60° → ±90°` 顺序执行，原 v09-r01 路线和结果保持冻结。
0° 第一阶段从零创建 Actor、Critic 和 Adam，每回合固定原始敌机初位，不消耗角度采样 RNG。
每 10000 步完成 PPO 更新后评估 1 个固定 0° 场景；连续两个评估 checkpoint 均捕获才晋级。
配置显式记录 `zero_angle_evaluation_episodes=1`、`zero_angle_promotion_capture_rate=1.0`。
这是一条固定轨迹的成功复现判定，不是 100 角度或 100 回合泛化结果。

正角阶段保持连续均匀抽样与固定 100 个含端点角度评估，阈值 0.8、连续次数 2。
所有阶段最多 100000 步，用尽本阶段预算未达标则终止整个实验，不自动跳过。
总上限 700000 仅为七阶段预算之和，不保证会全部执行。
每次晋级继承当前阶段历史 best-eval 的 Actor、Critic 和两个完整 Adam 状态，排序标准仍为
高捕获率、低失守率、短成功时间、低平均最小距离、低碰撞回合率；不以 reward 选 best。
previous 监控仍与 gate 隔离，其中 ±2° 阶段对前一 0° 的监控只有一个固定场景。
其余物理、奖励、观测、动作、MAPPO 超参数及网络初始化方法不变。

新独立输出 `outputs/v09_来袭方向课程/v09-r02_20261109_seed000/`，不从 r01 的 best/last 或成功旧模型 warm start。
旧 YAML 的正角首阶段行为保持兼容；现有阶段终止后视频和 XY 轨迹演化分析入口继续使用。

在 r02 的 0° 阶段达到原 100k 预算后，用户授权继续至累计 200k。
独立续训配置为 `configs/mappo_direction_curriculum_zero_start_200k.yaml`，同 run 的新输出为 `continuation_200k/`，原 100k 全部保留。
从原 100k last 恢复双网络、双 Adam、全部 RNG、未结束回合和完整环境，历史 70k best 仅作为累计选模候选。
预算向量为 `[200000,100000,100000,100000,100000,100000,100000]`，全局实际上限 200000；
0° 两次固定捕获及正角度两次 ≥80% 规则不变，提前晋级也不超本次总预算。
该授权只延长训练预算，不改变 reward、观测、动作、动力学、Direct Attacker 或 MAPPO 超参数。

**0° 更新调度复现核对：** 原 v09 课程入口为恰好每 10000 步评估，会截短临近评估点的 rollout；
这与 v08 普通入口持续收满 512 步再更新的调度不同。例如第 20 次更新，前者使用 272 步到 10000，后者使用 512 步到 10240。
配置中的 `rollout_steps: 512` 相同不表示实际每次更新的数据长度相同。
独立 `scripts.verify_zero_direction_reproduction` 使用 v09 的固定 0° 环境和未修改的普通 trainer 验证该差异，
配置为 `configs/mappo_zero_direction_reproduction.yaml`、独立输出为 `v09-r03_20261109_seed000`。
其 99840 步完整前缀与 v08 精确一致并重现 92160 步成功策略；100k 最后 160 步的短批次单独标注，不与原 300k 的后续完整批次作等价比较。
本验证不修改原课程入口的评估时点、晋级规则或后续阶段；实际结果见 [复现报告](experiments/V09_ZERO_DIRECTION_REPRODUCTION_REPORT.md)。

**v09-r04 已授权的 pretrained 课程：** 按用户后续要求，直接从 r03 的 92160 步成功模型进入
`±2° → ±5° → ±10° → ±30° → ±60° → ±90°`，不再训练 0°。
独立配置为 `configs/mappo_direction_curriculum_warm_start.yaml`，保留旧 YAML、模型和结果。
`warm_start` 显式给出源 checkpoint 及 SHA256；恢复双网络、双 Adam 与全局 RNG，新回合重新初始化，
不继续源文件中未结束的 237 步回合。新 run 的 `env_steps` 只计新增工作，原始 92160 步单独记录；
源步数加新增步数仅表示总训练投入，阶段回滚到历史 best 后不等于连续优化路径长度。

此入口显式选择 `curriculum.evaluation_timing: after_full_rollout`：不为评估截短 512 步 rollout，
跨过每个 10k 阈值后的首次更新进行评估并记录实际步数（例如 10240、20480、30208），
阶段/总训练硬预算末尾可有不足一批的尾部，仍保存并评估最终模型。
无该字段的旧配置默认 `exact_steps`，保留既有复现行为。
各阶段新增上限 100k；固定 100 角度集连续两次捕获率 ≥80% 才晋级，否则到预算停止。
首次 warm 初始评估只作为当前范围的历史 best 候选，不计连续达标。
第一 ±2° 阶段另用固定 0° 场景监控遗忘；后续阶段监控前一范围，监控结果不影响晋级或当前 best 排序。
阶段晋级仍继承历史 best 的双网络、双 Adam 和全局 RNG。
本轮未改变 reward、观测、动作、动力学、Direct Attacker 或网络/PPO 参数。
当前 warm-start 入口不与旧 `continuation` 同时使用；旧非 warm run 的续训支持保持不变。

**v10 独立策略保持诊断：** 使用原始 v08 92160 步成功 checkpoint，仅训练原始固定 0° 场景。
A 继承双网络与原双 Adam，B 继承相同双网络、清空双 Adam 的矩和计数；参数组、学习率、
环境、奖励、观测、动作、网络和 PPO 更新实现均不改变，也不继续课程。
两组恢复同一源全局 RNG、从同一固定初态的新回合开始，各新增 20k。
独立入口与配置为 `scripts.train_policy_retention`、`configs/mappo_policy_retention.yaml`；
每 2k 精确暂停采样做一次确定性完整回合，保留原训练环境、未结束回合、rollout 数据和全局 RNG。
若评估点处于 batch 中间，记录 `last_update_env_steps` 与 `pending_rollout_steps`，不提前 PPO 更新；
若恰在完整 batch 或硬预算末尾，则先更新再评估。20k 的最后 32 步尾批单独记录。
中间评估 checkpoint 是推理分析快照，未序列化 pending buffer，不承诺可直接恢复完整训练。
源文件中的 237 步 partial episode 两组共同丢弃，不将本实验称为历史 v08 同回合连续复现。
评估的首次失败只定位于相邻已检查策略之间，不能推断未检查更新点始终成功或从未恢复。
实际对照结果见 [策略保持报告](experiments/POLICY_RETENTION_ADAM_REPORT.md)。

**v11 独立 Actor LR 诊断：** 同一原始 v08 92160 步 checkpoint，继承双网络、双 Adam 的原矩和计数，
只比较 Actor LR=3e-4/1e-4/3e-5，Critic LR=1e-3，其余物理、奖励、观测、网络与 PPO 参数不变。
配置为 `configs/mappo_actor_lr_retention.yaml`，新入口 `scripts.train_actor_lr_retention` 复用 v10 的采样/PPO/评估循环。
源加载后只修改 Actor optimizer 的参数组 LR，避免 checkpoint 将实际新 LR 覆盖为原值；每次更新记录实际生效 LR。
三组仍从同一 source RNG 和同一固定0°新回合开始，各新增20k，采用同样的精确2k评估、不截512步批次和末尾32步更新规则。
不设置scheduler，不进行课程；v10未指定分组Actor LR的旧行为保留。A应复现上一轮v10 A的共同控制结果。
所有轨迹与同状态漂移比较均区分新增步数、源92160步和实际最近完整更新步数；固定单场景结果不能作为泛化统计。
实际结果见 [Actor LR 对照报告](experiments/ACTOR_LR_RETENTION_REPORT.md)。

**v13 独立固定std保持诊断。** 原始v08 92160步成功模型，固定0°，三组继承Actor/Critic和双Adam原状态。A=1×、B=0.15×、C=0.10×先将log_std设为source+ln(倍率)，再requires_grad=False，保留原Adam参数列表和状态，grad=None使Adam跳过log_std而均值网络继续训练。采样、PPO logprob和entropy使用同一个冻结std，其他LR/PPO参数不变。旧可学习std A不符合此冻结条件，本轮另跑固定1× A。各20k、完整512步批及精确2k评估规则沿用v10，末尾32步单独更新。冻结配置由本轮训练入口启用，普通from_checkpoint重建推理数值。完整续训还需显式恢复requires_grad处理及pending buffer。实际结果见[固定std保持报告](experiments/FIXED_STD_RETENTION_REPORT.md)。

训练结束后，标准 `scripts.train_mappo` 与 `scripts.train_direction_curriculum` 在最终模型和摘要落盘后调用独立的 `utils.post_training.run_post_training_analysis`。
默认按 `configs/trajectory_evolution.yaml` 抽样真实 checkpoint，在共同固定初态下进行确定性完整回合，生成以训练步数渐深着色的 XY 主图、逐 checkpoint 小图、指标 CSV、完整 NPZ 和来源 manifest。
普通 trainer 从此保存真实 `initial.pt`，不会为历史 v08 重造缺失的 0 步模型。
分析恢复外部 RNG、线程和 cuDNN 标志，不参与 PPO 更新、best 选择或课程晋级。
分析失败会发出 warning 并保存 `trajectory_evolution_error.json`，已经完成的训练权重保持可用。
分析参数单独配置，不改变物理、奖励、动作或观测；目前面向有 mission/base 的点防御快照。
生成方法和当前 v08/v09 结果见 [轨迹演化报告](experiments/TRAJECTORY_EVOLUTION_REPORT.md)。

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
当前验证顺序为全量 pytest → Direct Attacker sanity → reward sanity（stationary 与至少
10回合 random）→ 单次 20k 诊断训练 → final/best 评价和 3D 对照；reward sanity 未通过
时先定位工程问题，不开始训练。命令示例：

```powershell
.\.venv\Scripts\python.exe -m scripts.sanity_direct_attacker --config configs/baseline_3a1c.yaml --output-dir outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/direct_attacker_sanity
.\.venv\Scripts\python.exe -m scripts.sanity_reward --config configs/baseline_3a1c.yaml --episodes 10 --seed 0 --output-dir outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/sanity
.\.venv\Scripts\python.exe -m scripts.train_mappo --config configs/mappo_baseline.yaml --total-steps 20000 --output-dir outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000
.\.venv\Scripts\python.exe -m scripts.evaluate_mappo --checkpoint outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/checkpoints/final.pt --episodes 10 --output-dir outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/evaluation/final
.\.venv\Scripts\python.exe -m scripts.evaluate_mappo --checkpoint outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/checkpoints/best.pt --episodes 10 --output-dir outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/evaluation/best
.\.venv\Scripts\python.exe -m scripts.visualize_mappo --checkpoint outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/checkpoints/best.pt --output-dir outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/visualization/best
```

`scripts.sanity_reward` 保留 stationary 完整回合和 random 每回合原始 NPZ、CSV、summary，
独立从状态重算距离差分与分量求和。Stationary 采用原零初速和全零加速度，不改初始化、
动力学或敌机策略。失败回报符号结论仅适用于实际检查的配置。

## 12. 历史兼容与验证

历史无 `mission`、使用 `evader` 字段的配置/旧 checkpoint 仍可按原任务加载，
保留 `RepulsiveEvaderPolicy` 与已有 boundary 实现，不将纯逃跑设为当前默认行为。
含 mission 但没有 `attacker.policy.mode` 的旧配置继续使用 `GoalDirectedAttackerPolicy`，
用于复现上一版点防御快照；当前 YAML 必须显式选择 `mode: direct`。
代码继续使用 `PursuitEnv` 类名和部分 `evader` 兼容别名，避免无关的大规模重命名。

Reward schema 也保持历史快照可复现：完整旧 `progress_coefficient` / `time_penalty` 配置
使用原 reward，完整 `active_progress_coefficient` / `survival_penalty_per_second` 配置
使用active/survival；完整`distance_cost_coefficient` / `survival_penalty_per_second`配置使用v07距离时间成本。
三种schema互斥，禁止混用和不完整的新schema；active与绝对距离schema要求存在mission/base。
默认新 YAML 不保留旧项。Evaluation 默认读 checkpoint 的环境快照，旧模型不会因当前
YAML 变更而静默换奖励；历史不存在的 active/survival 累计 metrics 记为 null，不填造0。

历史追逃环境的边界修复曾将 workspace 扩为 `[-300,-300,0]` 至 `[300,300,150]`，
并启用 margin 50/20、strength 2 的 steering。该历史对照中的硬裁剪减少不代表新任务成绩，
详见 [BOUNDARY_MECHANISM_REPORT.md](experiments/BOUNDARY_MECHANISM_REPORT.md)。
旧 100k 训练和视频也属于旧任务，见
[MAPPO_100K_BASELINE_REPORT.md](experiments/MAPPO_100K_BASELINE_REPORT.md)，
不能与新点防御任务直接比较学习效果。

上一版点防御任务与旧 GoalDirected 制导的实际结果见
[POINT_DEFENSE_TASK_REPORT.md](experiments/POINT_DEFENSE_TASK_REPORT.md)。
Direct Attacker 最初的 pytest、无干扰 sanity check、random rollout、3D 图与 smoke training
实际结果见 [DIRECT_ATTACKER_BASELINE_REPORT.md](experiments/DIRECT_ATTACKER_BASELINE_REPORT.md)。
同一 Direct Attacker、旧 progress/time reward 的正式 100k 失败结果见
[DIRECT_ATTACKER_MAPPO_100K_REPORT.md](experiments/DIRECT_ATTACKER_MAPPO_100K_REPORT.md)。
当前 reward 修复、sanity、20k诊断与新旧 best 行为对照见
[ACTIVE_PROGRESS_REWARD_REPORT.md](experiments/ACTIVE_PROGRESS_REWARD_REPORT.md)。
默认 `pytest.ini` 只收集 `tests/`，避免历史输出中的源码快照被重复收集。

## 13. 当前不包含与建模限制

默认参考环境保持固定初始化；v09 已授权在独立入口加入敌机初始方位随机采样和课程。
本阶段不加入随机我方队形/初速度、随机敌机高度/基地距离、障碍物、观测噪声、显式包围奖励、
trajectory prediction、对抗 RL attacker、真实姿态/地面碰撞模型或 sim-to-real。

当前主要限制包括：默认参考环境固定方向与固定初始化（v09 仅覆盖配置角区间）、actor 的基地关系不可完全观测、
加速度历史未纳入输入、离散步接触检测、未建模空气阻力的质点动力学、数值裁剪仍保留速度、
Direct Attacker 不覆盖敌机规避/机动能力且跟踪增益仍是暂定值。默认初态 sanity check
不能证明任意初速度、任意几何都能有限时间到达。Active progress 已扣除敌机朝基地的同期
净进展，但仍是最近距离代理，不能仅据 reward 上升判断我方学会保护基地或协同。

后续研究应先根据真实任务结局与轨迹分析问题，再明确授权修改研究设定。
