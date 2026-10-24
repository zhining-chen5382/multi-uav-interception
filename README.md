# Multi-UAV Cooperative Interception

## 当前任务

**Three cooperative UAV interceptors defend a fixed point target against one attacking UAV.**

当前 Baseline v1 是 3D point-defense（点防御）任务：三架我方拦截无人机保护位于
原点的基地，一架敌机从约 500 m 外被发现并试图进入半径 10 m 的攻击球。
我方需要在敌机攻击成功前，以保留的 5 m capture radius 完成拦截。

当前默认任务已经从历史 pursuit-evasion（单纯追逃）改为点防御。三架我方机使用
共享 actor 的 MAPPO，敌机使用朝基地飞行、不主动规避我方机的 Direct Attacker 规则策略。场景仍为 3A1C、
固定初始化、无障碍物的三维连续质点环境。具体定义见 [BASELINE_SPEC.md](BASELINE_SPEC.md)，
本轮实际验证见 [DIRECT_ATTACKER_BASELINE_REPORT.md](experiments/DIRECT_ATTACKER_BASELINE_REPORT.md)。

本轮只修改攻击机制导并验证到达行为与 pipeline 兼容性，没有执行新的 100k 正式训练，
也不能将 smoke training 或 reward 上升解释为已学会保护基地。

## 任务配置与初始化

默认环境配置为 `configs/baseline_3a1c.yaml`，核心参数均从 YAML 读取：

| 项目 | 当前配置 |
| --- | --- |
| 基地 / detection radius / attack radius | `[0,0,0]` / 500 m / 10 m |
| 我方初始位置 | `[1.5,0,1]`, `[-0.75,1.25,1]`, `[-0.75,-1.25,1]` m |
| 敌机初始位置 | `[497.49371855331,0,50]` m，距基地约 500 m |
| 初始速度 / 初始化方式 | 全零 / 固定，无随机队形 |
| Capture radius / pursuer safety distance | 5 m / 2 m，均不变 |
| `dt` / 最大回合步数 | 0.1 s / 500，均不变 |
| 数值 workspace | 三轴 `[-750,750]` m |

我方三机距原点最大约 1.8028 m，最小两两距离 2.5 m，满足原点 2 m 内与原安全距离，
不会在 reset 时直接触发碰撞，无需降低 safety distance。
500 m 是本轮导师确认的固定探测/初始化距离，并未实现雷达、传感器探测或距离观测裁剪。

双方物理性能保持原值：水平最大速度 20 m/s、爬升/下降 8/6 m/s、水平/垂直加速度
3/2 m/s²、最大 jerk 5 m/s³。数值 workspace 留有显著余量，hard position clip 仅作
最后保护，保留速度，不作为战场墙或地面高度模型；没有加入地形或地面碰撞。

## Attacker policy 与 boundary

`envs/attacker_policy.py` 的 `DirectAttackerPolicy` 使用 desired-velocity tracking：

```text
goal_vector = base_position - attacker_position
goal_direction = goal_vector / norm(goal_vector)   if norm(goal_vector) > distance_epsilon
goal_direction = 0                               otherwise
desired_velocity = attack_speed * goal_direction
physical_acceleration_command = velocity_gain * (desired_velocity - current_velocity)
```

配置全部来自 YAML：

```yaml
attacker:
  policy:
    mode: direct
    attack_speed: 20.0
    velocity_gain: 1.0
    distance_epsilon: 0.000001
```

`attack_speed=20 m/s` 是期望三维速度的模长。`velocity_gain=1 s^-1` 对应未受限的
一阶速度跟踪时间常数 1 s；实际起步仍受原加速度和 jerk 限制，不能解释成一秒内到达
20 m/s。这是一组 provisional defaults，没有超参数搜索。

Policy 不对加速度再次归一化，不调用 `scale_action`，不自行限幅；输出原始物理加速度指令，
由现有 dynamics 统一施加水平/垂直加速度、jerk 和速度限制。横向速度与目标方向不一致时，
速度误差会产生修正指令。Pursuer 位置不影响 Direct Attacker；没有 avoidance、随机机动或预测。

攻击机持续飞向攻击球，不加入 arrival braking、原点停车或悬停。距离不超过 epsilon 时
目标方向置零仅用于分母保护；正常任务在严格进入 10 m 球时已终止，不运行原点停车阶段。

`mode: direct` **关闭 soft boundary steering，并拒绝 `boundary_*` 配置**，避免无意引入
其他制导力；pursuer 也不使用 boundary-aware steering。此前的 GoalDirected、repulsion 和
边界模块保留用于无 `mode` 的历史配置/快照，不影响当前默认策略，也没有新增 evasive mode。

`info` 保留裁剪诊断：`boundary_clip_count` 为本回合累计“飞行器-步”事件数，
`boundary_clip_counts` 按 `[P0,P1,P2,attacker]` 分别计数，`boundary_clipped` 为当步布尔数组。
一架飞机一帧多轴越界计一次，恰好在边界而未越界不新增计数；reset 清零，返回数组为副本。

## Termination、reward 与接口

每步四机同步更新后，用同一时刻状态判断，顺序明确：

1. **Interception Success**：最近我方机-敌机距离严格 `<5 m`，`terminated=True`。
2. **Enemy Attack Success**：未捕获且敌机-基地距离严格 `<10 m`，`terminated=True`。
3. **Timeout**：未发生两种 terminal outcome 且达到 500 步，`truncated=True`。

同一步同时满足两个距离范围时 capture 优先；不会同时施加攻击失败惩罚。
Exactly 5 m / 10 m 不触发对应终止。Terminal outcome 也优先于同一步 timeout。
按更新后的离散位置判定，不做步内连续接触检测。

团队 reward 保留 `progress + time + collision + capture + attack_failure`，本轮没有修改。
原 `attack_failure_penalty=-100`、成功 capture 项 +100、进展系数 1、每步时间项 -0.01、
发生任意不安全我方机对时每步团队扣 -5 均不变。终止步也计算原有其他项，所以完整终止步
reward 不一定恰好为 ±100。没有新增 encirclement、angle、role 或 prediction reward。

```python
from envs import PursuitEnv

env = PursuitEnv("configs/baseline_3a1c.yaml")
observations, info = env.reset(seed=0)
terminated = truncated = False
while not (terminated or truncated):
    observations, rewards, terminated, truncated, info = env.step(env.sample_actions())
```

- Action shape `(3,3)`、范围 `[-1,1]`；有限越界动作裁剪，NaN/Inf 报错。
- Observation shape `(3,21)` 不变：自身速度、相对敌机位置/速度、两名队友相对位置、
  两名队友相对速度。队友按固定编号升序排列，无噪声、无新增基地输入。
- Global state `(24,)` 不变：`[P0_state,P1_state,P2_state,attacker_state]`，每机
  `[x,y,z,vx,vy,vz]`。`env.get_state()` 与 `info["global_state"]` 返回副本。
- Reward shape `(3,)`，三个值相同，团队累计一次。
- `info` 包含 `captured`、`interception_success`、`attacker_reached_base`、
  `enemy_attack_success`、`interception_failure`、`distance_to_base`、`min_distance`、
  `collision_pairs`、`step_count`、`global_state`、`mission_enabled` 与边界诊断。
  `attacker_in_attack_region` 独立记录几何进入攻击球的事实；capture 优先时也可能为 True。
- `step()` 的 `reward_components` 包含 `capture/progress/time/collision/attack_failure`。
  结束后必须 reset，才能继续 step。

类名仍为 `PursuitEnv`，并保留部分 evader 兼容别名，避免无关大规模重命名。
普通 NumPy 接口未适配 Gymnasium/PettingZoo。

## 安装、测试与 random rollout

所有命令在仓库根目录执行（Python 3.8+）。推荐独立虚拟环境，避免全局数值库冲突：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
.\.venv\Scripts\python.exe -m pytest -q
```

第三方 pytest 插件自动加载可按上例关闭，本项目测试不依赖额外插件。
`pytest.ini` 仅收集 `tests/`，不收集历史 `outputs/` 源码快照。
测试覆盖任务初始化、Direct Attacker 期望速度和侧向速度修正、不使用 avoidance、完整到达行为、
数值安全、两种 terminal outcome 和优先级、失败奖励、
原物理约束及 21D/24D 接口、MAPPO 更新/checkpoint/CLI 评估和绘图。

先单独运行无 pursuer 干扰的 Direct Attacker sanity check：

```powershell
.\.venv\Scripts\python.exe -m scripts.sanity_direct_attacker --config configs/baseline_3a1c.yaml --output-dir outputs/direct_attacker_validation/sanity
```

该入口仅积分攻击机，使用当前固定初态、原动力学和严格 `<10 m` 判定，记录初始/最小基地距离、
进入攻击球时间、进入时速度与 hard clipping，并保存原始轨迹、距离曲线和 3D 轨迹图。
检查整个无干扰回合的距离下降、有限性和是否按时进入攻击球，不能只检查前几秒的进攻方向。
验证与图像完成后再运行普通 point-defense random rollout：

Random rollout 默认读取 `rollout.seed` / `rollout.episodes`，也可显式指定：

```powershell
.\.venv\Scripts\python.exe -m scripts.random_rollout --config configs/baseline_3a1c.yaml --seed 0 --episodes 3
```

每回合输出 JSON，包含三种任务结局、reward、步数、最近敌我距离、敌机到基地距离、
碰撞与 hard clipping 诊断。该入口是环境检查，不自动生成正式训练记录或模型。
本轮 random/rule-based 的实际行为和 limitations 见当前任务报告。

## MAPPO baseline

### 网络与更新流程

三个 pursuer 共享一个 `SharedActor`，每机仅输入自身 21D local observation。
默认 `21 → 128(Tanh) → 128(Tanh) → 3D Gaussian mean`，共享可学习 3D log_std。
采样 `z ~ Normal(mean,std)` 后执行 tanh(z)；确定性评估使用 tanh(mean)。
Critic 为 `24 → 256(Tanh) → 256(Tanh) → 1`，只预测一次团队 value。
网络使用原有正交初始化与零 bias，没有因任务重构修改架构或初始化。

单环境顺序 rollout 保存局部观测、全局状态、压缩前动作、log probability 与团队
reward/value。Log probability 包含 tanh Jacobian 修正；entropy 用新采样的
`-log_probability` 估计并保留梯度，可能为负。

GAE 对拦截成功与敌方攻击成功均不 bootstrap；timeout 使用 reset 前最后状态的价值
bootstrap，同时切断跨回合 GAE 链。rollout 中途截断也使用下一状态价值。
按配置标准化 advantage，再做 clipped PPO、value loss、entropy bonus 与梯度裁剪。
一个 minibatch 的单位是团队时间步，64 步对应 192 个 actor 样本与 64 个 critic 样本。
Value loss 为 `0.5*mean((V-return)^2)`，再乘配置权重。

实现位于 `algorithms/mappo/`。本轮保留既有 PPO 逻辑、网络、训练配置和区分点防御任务
结局的训练日志。超参数全部来自 `configs/mappo_baseline.yaml`：

| 配置 | 默认值 |
| --- | --- |
| 环境 / seed / device / CPU threads | `baseline_3a1c.yaml` / 0 / cpu / 1 |
| 总环境步数 / rollout steps | 100000 / 512（本轮通过 CLI 仅运行极短 smoke） |
| 日志 / checkpoint 间隔，统计窗口 | 1 / 10 updates，50 episodes |
| Actor / critic hidden sizes | `[128,128]` / `[256,256]` |
| 初始 / 最小 / 最大 log_std | -0.5 / -5 / 2 |
| 隐藏层 / actor 输出 / critic 输出初始化增益 | sqrt(2) / 0.01 / 1 |
| Actor / critic learning rate，Adam epsilon | 0.0003 / 0.001，0.00001 |
| Gamma / GAE lambda / clip ratio | 0.99 / 0.95 / 0.2 |
| Update epochs / minibatch size | 4 / 64 |
| Entropy / value loss coefficient / max grad norm | 0.01 / 0.5 / 0.5 |
| Advantage normalization / epsilon | true / 1e-8 |
| Observation / global state normalization | false / false |
| Reward scale | 1.0 |
| Evaluation episodes / seed | 10 / 1000 |
| Visualization seed / DPI / figure size | 1000 / 160 / `[10,8]` |

可选输入缩放由数值 bounds 与速度限制推导，作为 network buffer 存入 checkpoint；
默认关闭，不改变环境原始 21D/24D 输出。Reward scale 是显式常数，CSV 保留原始回报。

### 极短训练、评估与 3D demo

本轮只做兼容性验证，示例运行 512 个环境步，不运行默认 100k。先完成上述 pytest、
无干扰 sanity check、轨迹图与 random rollout，再执行：

```powershell
.\.venv\Scripts\python.exe -m scripts.train_mappo --config configs/mappo_baseline.yaml --total-steps 512 --output-dir outputs/direct_attacker_validation/smoke_train
.\.venv\Scripts\python.exe -m scripts.evaluate_mappo --checkpoint outputs/direct_attacker_validation/smoke_train/checkpoints/final.pt --episodes 3 --output-dir outputs/direct_attacker_validation/evaluation_final
.\.venv\Scripts\python.exe -m scripts.evaluate_mappo --checkpoint outputs/direct_attacker_validation/smoke_train/checkpoints/best.pt --episodes 3 --output-dir outputs/direct_attacker_validation/evaluation_best
.\.venv\Scripts\python.exe -m scripts.visualize_mappo --checkpoint outputs/direct_attacker_validation/smoke_train/checkpoints/final.pt --output-dir outputs/direct_attacker_validation/visualization
```

训练支持 `--config`、`--total-steps`、`--output-dir`、`--seed`。
`env_config` 相对训练 YAML 所在目录解析，相对输出目录相对仓库根目录解析。
同目录已有训练指标、配置或 checkpoint 时拒绝覆盖，重复实验应使用新目录。
步数按环境 step 计数，不乘 pursuer 数量；不完整回合不会伪装为完整 episode。

评估支持 `--episodes`、`--seed`、`--output-dir`、`--device`；绘图支持后三项。
两者默认使用 checkpoint 的环境快照，只有显式 `--env-config` 才覆盖并记录来源。
输出三种结局对应的 `interception_success_rate`、`enemy_attack_success_rate`、`timeout_rate`，
以及 `average_attacker_base_terminal_distance` 和仅成功回合的 `average_interception_time`。
保留 `capture_rate` / `average_capture_time` 兼容字段；无成功时 time 为 null，不用 0/NaN 冒充。
同时保留 episode reward/length、最小距离、碰撞率和我方平均两两距离。

3D 静态图使用无头 Agg 后端，绘制基地、10 m attack sphere、四机轨迹及起终点，
标题显示任务结局、步数、最近敌我距离与最终敌机-基地距离。
同时保存初始状态和完整轨迹，无需桌面窗口。

### 输出文件与复现

| 文件 | 内容 |
| --- | --- |
| `training_config.yaml`, `environment_config.yaml` | 实际训练配置与独立环境快照 |
| `metadata.json` | Seed、Git 版本/dirty 状态、依赖版本、输入来源与输出目录 |
| `training_metrics.csv`, `updates.csv` | 完整回合 reward/mission metrics；逐更新 loss/entropy/统计窗口 |
| `checkpoints/update_*.pt`, `final.pt`, `best.pt` | 周期、最终与训练窗口启发式最优 checkpoint |
| `evaluation_metrics.csv`, `evaluation_summary.json` | 逐回合指标与三种任务结局汇总 |
| `first_episode.npz` | 首个评估回合完整轨迹 |
| `trajectory_3d.png`, `trajectory.npz` | 独立 3D 图及完整原始轨迹 |
| `evaluation_*`, `visualization_*` 配置与 metadata | 有效环境快照、seed 与 checkpoint 来源 |

Checkpoint 包含 actor/critic、optimizer、训练计数、随机状态和环境状态；暂无 resume CLI。
`best.pt` 依据训练窗口捕获率、再平均回报选择，不保证确定性评估最优。
极短训练未完成任何回合时回退为最终策略并记录这一事实。
生成输出、模型、cache 与虚拟环境均不默认提交 Git。

## 当前建模限制

- Actor 只有相对状态与自身速度，对共同平移不敏感，不能唯一恢复基地相对方位或距离。
  基地在 critic 的绝对 global state 中可知，但本轮不向 actor 扩维。
- Jerk 依赖的加速度历史不在既定 21D/24D 输入内，因此输入不是完整 Markov 状态。
- 固定初始化下 deterministic policy 会重复同一轨迹，多次评估不能当作泛化证据。
- Direct Attacker 不规避我方机，也不覆盖机动敌机情形；速度跟踪增益仍是未经搜索的暂定值。
  默认初态的 sanity check 不能证明任意初速度、任意目标几何下都能在有限时间进入攻击球。
- 原 progress reward 会因敌机自己朝我方接近而增加；回报改善不等于拦截能力改善，需看任务结局。
- 数值空间没有地面模型；hard clip 仍保留速度。接触判定只有离散更新时刻。
- 当前仅单环境、前馈网络与 CPU 验证，没有分布式采样或长期新任务性能结论。

## 历史实现与报告

历史 pursuit-evasion 环境保留兼容支持：无 `mission` 且使用 `evader` 的旧 YAML/历史
checkpoint 继续按旧规则运行，不设为当前默认任务。

上一版点防御使用 `GoalDirectedAttackerPolicy` 的“位置目标方向加速度 + 我方机规避”。
历史无近威胁对照最近距基地约 19.28 m，随后过冲绕行，未进入 10 m 球，见
[POINT_DEFENSE_TASK_REPORT.md](experiments/POINT_DEFENSE_TASK_REPORT.md)。该观察属于旧制导，
本轮以 desired-velocity tracking 修复；旧 policy 模块只为无 `mode` 的历史快照保留。

此前边界修复使用 `[-300,-300,0]` 到 `[300,300,150]` 的 workspace，evader soft steering
margin 50/20、strength 2。在相同 10 组随机种子的历史对照中，旧边界、仅扩大空间、
扩大空间并 steering 的敌机硬裁剪比例依次为 83% / 63% / 13%；仍可能连续数秒贴边。
这些数据属于旧追逃任务，仅用于解释历史边界 artifact，不能作为当前点防御任务成绩。
详见 [BOUNDARY_MECHANISM_REPORT.md](experiments/BOUNDARY_MECHANISM_REPORT.md)。

[MAPPO_DEMO_REPORT.md](experiments/MAPPO_DEMO_REPORT.md) 与
[MAPPO_100K_BASELINE_REPORT.md](experiments/MAPPO_100K_BASELINE_REPORT.md) 记录旧任务实验。
旧 100k 模型和视频使用其各自环境快照，不能与新配置混为同一训练环境，
也不能直接比较新旧任务的 reward/capture 学习效果。

当前 Direct Attacker 的实际 pytest、无干扰 sanity check、rollout、3D visualization 与
smoke training 结果，统一见 [DIRECT_ATTACKER_BASELINE_REPORT.md](experiments/DIRECT_ATTACKER_BASELINE_REPORT.md)。
后续研究扩展应在任务验证基础上另行明确，不提前加入包围奖励、随机初始化、障碍物、
观测噪声、curriculum、轨迹预测、对抗 RL attacker 或更复杂动力学。
