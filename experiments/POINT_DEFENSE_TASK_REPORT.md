# Baseline v1 点防御任务定义重构与验证

日期：2026-10-24。

**任务定义、终止/奖励、评估、绘图和 MAPPO 兼容流程已接通；但当前给定的攻击机加速度势场还不能保证无威胁时持续接近直至进入攻击球。**
真实诊断中，攻击机先接近基地，再过冲绕行；本轮保留这一结果，没有改权重、降低难度或额外加入阻尼。
因此本报告证明的是任务接口和工程流程可运行，不能证明攻击机到达控制已经充分、也不能证明 MAPPO 学会了任务。

## 1. 当前任务与代码版本

**Three cooperative UAV interceptors defend a fixed point target against one attacking UAV.**

三架我方拦截机保护位于 `[0,0,0]` 的三维点基地；一架规则攻击机从约 500 m 处被发现，
向基地的三维攻击球推进。我方需在敌机攻击成功前拦截，不再以追逐无限逃逸目标为默认任务。

只读 Git HEAD：`23bf55576cca27944b4ee2ae5c53a70295c7dac3`。
实际运行包含未提交的工作区改动，不能仅凭此 HEAD 重建本次代码；有效配置与源码 SHA256
记录在 `outputs/point_defense_validation/manifest.json`，训练另有配置快照和 metadata。
没有执行 `git add`、`commit`、`merge` 或 `push`，没有运行新任务的 100k 正式训练。
此前的边界报告、100k 结果、checkpoint 和视频保持为旧任务历史产物。

## 2. 任务参数与固定初始化

全部来自 `configs/baseline_3a1c.yaml`：

| 参数 | 本次实际值 |
| --- | --- |
| `mission.base_position` | `[0,0,0]` m，三维点 |
| `mission.detection_radius` | **500 m**，导师确认的发现距离 |
| `mission.attack_radius` | **10 m**，严格三维球形区域 |
| `capture.radius` | 5 m，原值不变 |
| P0 | `[1.5,0,1]` m |
| P1 | `[-0.75,1.25,1]` m |
| P2 | `[-0.75,-1.25,1]` m |
| Attacker | `[497.49371855331,0,50]` m，距基地 500 m |
| 初速度 / 初始加速度历史 | 四机均为零 |
| `dt` / 最大回合长度 | 原值 0.1 s / 500 steps，即 50 s |
| 数值 workspace | 三轴均为 `[-750,750]` m |

编队关于 x-z 平面对称、不重合。我方距原点分别约 1.8028、1.7678、1.7678 m，均小于 2 m；
两两距离约 2.5739、2.5739、2.5 m，大于原 `pursuer_safety_distance=2 m`，reset 无碰撞。
示例中的 y 间距 1.5 m 会小于安全距离，因此采用上述对称三角形，不修改安全距离。
导师的“原点 2 m 内”要求与当前安全距离不存在几何冲突。

`detection_radius` 定义本轮固定初始发现时刻/距离，不是新增传感器、观测范围或动态发现过程。
所有 reset 都使用同一初态，seed 只影响随机动作/训练随机数。

## 3. Attacker policy 与权重选择

新增 `GoalDirectedAttackerPolicy`，默认替换纯逃逸行为：

```text
goal_direction = normalize(base_position - attacker_position)
away_i = attacker_position - pursuer_position_i
pursuer_repulsion = sum_i(away_i / max(norm(away_i), distance_epsilon)^3)
combined = goal_weight * goal_direction + avoidance_weight * pursuer_repulsion
command = scale_action(normalize(combined), attacker_limits)
```

基地距离不超过 epsilon 时目标项为零；合力为零时返回零，不随机破平局。
原始 inverse-square pursuer repulsion 公式不变，不先将其归一化。
总方向归一化后使用原有物理加速度映射，再经过同一积分、jerk 和速度限制。
输入和配置校验防止非法 shape、NaN/Inf；强近距离合力在归一化前缩放，避免求模溢出。

实际配置 `goal_weight=1.0`、`avoidance_weight=25.0`、`distance_epsilon=1e-6`。
理由是单位进攻项下，一架我方机在 5 m 的加权斥力为 1，在 10 m 为 0.25，远处贡献迅速减小。
这是一次可解释的暂定选择，没有比较多组权重再择优。多个近邻仍可合力压过目标项。

此公式指定加速度指令，不是速度跟踪或稳定到达控制器；其实际限制见第 10 节。

## 4. Termination 与 info

四机从同一旧状态同步更新，hard clip 后以同一时刻状态判断，优先级为：

1. 最近 pursuer-attacker 距离严格 `<5 m`：`captured=interception_success=True`，终止。
2. 未捕获且 attacker-base 距离严格 `<10 m`：
   `attacker_reached_base=enemy_attack_success=interception_failure=True`，终止。
3. 两类 terminal 均未发生且达到 500 步：仅 `truncated=True`。

同一步同时满足 capture/attack 时只判我方成功，不给敌方攻击成功标志或失败惩罚。
`attacker_in_attack_region` 单独保留原始几何事实，因此 capture 优先时它仍可为 True。
Exactly 5 m / 10 m 不触发对应条件。两类 terminal 均优先于同一步 timeout。

`info` 增加上述任务标志、`distance_to_base`、`mission_enabled`，保留 `min_distance`、
`collision_pairs`、`step_count`、`global_state` 和逐机/逐步/累计 boundary clipping 诊断。
两类 terminal 均禁止 GAE bootstrap；timeout 仍从最终状态 bootstrap，并切断跨回合 GAE。
修正训练和 random 日志中原先把所有 `terminated` 等同于 `captured` 的假设。

## 5. Reward 最小修改

原进展、时间、碰撞、捕获系数及计算方式保持不变，只新增 YAML 的失败项：

| 项 | 配置 / 行为 |
| --- | --- |
| Capture | +100，仅我方拦截成功 |
| Attack failure | **-100**，仅敌方攻击成功 |
| Progress | `1 * (previous_min_distance-current_min_distance)` |
| Time | 每步 -0.01 |
| Collision | 任意我方机对间距 `<2 m` 时，每步团队扣一次 -5 |

同一团队 reward 返回三个相同值，日志只累计一次。终止步仍计算其他原有项，完整值未必恰好 ±100。
未新增包围、角度、角色、拦截点或预测奖励。
进展项会因攻击机主动靠近而增大，所以无捕获也可能有较高累计回报，不能仅看 reward 认定学会任务。

## 6. Boundary 与物理约束

三轴 ±750 m 数值空间覆盖 500 m 探测球，并留 250 m 余量，不是战场墙。
仍保留 hard clip、速度不重置；默认 attacker 不提供三个 `boundary_*` 键，soft steering 关闭。
旧 `RepulsiveEvaderPolicy` 和连续边界力实现保留；显式提供完整三参数时可选启用，不加入 pursuer steering。

双方物理约束维持水平速度 20 m/s、爬升/下降 8/6 m/s、水平/垂直加速度 3/2 m/s²、
jerk 5 m/s³。`envs/dynamics.py` 与此前正式实验快照逐字节一致。
原捕获半径、安全距离、dt、episode 上限均未修改。

数值空间允许负 z，因为基地按三维点定义、没有地面或地形模型；这不是现实高度约束。
默认验证的全部回合均为 **0 hard clipping**，没有借助边界限制 attacker。

## 7. Observation、state 与 MAPPO 保护项

- Local observation 仍为 `(3,21)`：自身速度、相对敌机位置/速度、两名队友相对位置/速度，顺序不变。
- Global state 仍为 `(24,)`：P0、P1、P2、attacker 的位置与速度；没有追加基地或任务标志。
- Critic 从绝对坐标知道基地关系；21D actor 仅有相对状态，对共同平移不敏感，不能唯一确定基地相对方位。
- Jerk 所需加速度历史仍未纳入输入，这些输入不构成完整 Markov 状态。
- Actor/critic 架构、初始化、PPO、buffer 以及 `configs/mappo_baseline.yaml` 逐字节保持不变。
- `normalization.py` 仅兼容读取新 `attacker` / 历史 `evader` 物理配置，缩放公式和默认关闭开关不变。

`PursuitEnv` 类名、`num_evaders` 配置键和少量旧 evader 属性别名保留，避免无关大规模改名。
无 mission 的历史 evader 配置仍运行旧任务；独立审查以旧源码和相同旧 YAML 运行完整 500 步，
状态、加速度、观测、奖励和终止结果逐元素一致。旧任务不作为新默认。

## 8. 实际 pytest

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
$env:MPLBACKEND = "Agg"
.\.venv\Scripts\python.exe -m pytest -q
```

**156 passed in 12.70s**。

覆盖初始编队/500 m 距离、初始无碰撞、goal/avoidance 原始合成、有限性和物理限制、
严格 5/10 m 阈值、真正三维球、同步更新后判断、capture 优先、失败奖励、任务 info、
21D/24D、完整随机回合、旧 dynamics/evader/boundary 回归，以及短训练、checkpoint、评价和绘图。
历史边界测试使用显式 legacy fixtures，不依赖被忽略的 outputs 目录，也未删除原覆盖。

## 9. 实际 rollout 与受控终局验证

先执行全量 pytest，再实际执行默认 random 入口，随后保存完整轨迹、绘图与额外诊断：

```powershell
.\.venv\Scripts\python.exe -m scripts.random_rollout --config configs/baseline_3a1c.yaml --seed 0 --episodes 3
.\.venv\Scripts\python.exe outputs/point_defense_validation/validate_point_defense.py
```

| Random seed | Steps / 结局 | Team reward | 最小敌我距离 m | 最小基地距离 m | 终点基地距离 m | 碰撞步数 | Hard clips |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 500 / Timeout | 404.4987 | 14.0233 | 19.3345 | 79.4003 | 0 | 0 |
| 1 | 500 / Timeout | 371.4125 | 33.2101 | 19.3141 | 78.7405 | 0 | 0 |
| 2 | 500 / Timeout | 347.3975 | 27.1556 | 19.4078 | 78.7325 | 12 | 0 |

三个默认随机回合均无拦截、无敌方攻击成功，timeout rate=100%。这不是训练 MAPPO 的结果。
另运行默认初态下零加速度 pursuers，以及将 pursuers 移远的无近威胁受控对照，各 500 步。
五个完整回合与四个受控终局步共 **2504 steps**，独立审计形状、有限性、速度、加速度、jerk、
半隐式积分、clip masks/计数与终止标志，全部通过。

四个独立受控 YAML 只为验证逻辑调整初态和一步 horizon，保持原物理限制：

| 受控情况 | 实际结果 | Capture 项 | Attack failure 项 |
| --- | --- | ---: | ---: |
| 敌我约 4 m、远离基地 | Interception Success | +100 | 0 |
| 敌机距基地约 9 m、我方远离 | Enemy Attack Success | 0 | -100 |
| 两阈值均未触发且到测试上限 | Timeout | 0 | 0 |
| Capture / attack 同步触发 | Interception Success，raw attack region=True | +100 | 0 |

这些是受控工程验证，不能混入默认任务成功率。有效 YAML 与原始结果均已保存。

## 10. Attacker 推进与 3D 行为：关键未解决限制

默认随机回合均在第 284 步、28.4 s 达到约 19.3 m 的基地最近距离，然后过冲绕行，
最终回到约 79 m；距离增大的步数分别为 141、141、140。攻击机没有无限向外逃跑，
但也没有稳定抵达 10 m 攻击区，不能把“目标项朝基地”当作完整到达控制已验证。

无近威胁对照中，最近基地距离仍为 **19.2824 m**，终距 **78.7369 m**，同样 timeout。
最近点位置约 `[0.4480,0,-19.2772]` m，速度约 `[-20,0,-0.4956]` m/s；敌机从基地下方掠过。
默认零动作 pursuers 的最近基地距离为 19.2897 m，结果也相似。
这支持“惯性、jerk、水平/垂直加速度映射不同导致过冲”的解释，而非归咎于近距离规避或边界。

**因此“无威胁时持续向基地接近，直到攻击成功”的强行为要求尚未完全满足。**
前 120/150 步单调接近的测试只证明初期推进，不能替代完整 500 步验证。
本轮按照给定合成公式与原映射保留真实结果，没有私自加速度阻尼、速度跟踪、预测或换初态来掩盖问题。
正式训练前应先确认 attacker 的到达控制定义；本报告不建议立即用当前模型启动新 100k。

![完整随机回合 3D 轨迹](../outputs/point_defense_validation/default_random_seed0_trajectory_3d.png)

![基地距离原始曲线，无平滑](../outputs/point_defense_validation/attacker_base_distance_raw.png)

3D 图含基地、10 m 球、四机轨迹、起止点及真实结局、步数和两类距离，均已实际打开检查。
后续 MAPPO smoke 的独立绘图入口也已运行，见 `smoke_visualization/trajectory_3d.png`。

## 11. 实际 MAPPO smoke training 与独立评价

完成上述验证后，运行一次 **512 steps** 的短训练，除 CLI 总步数和输出目录外保留默认全部配置：

```powershell
.\.venv\Scripts\python.exe -m scripts.train_mappo --config configs/mappo_baseline.yaml --total-steps 512 --output-dir outputs/point_defense_validation/smoke_train
.\.venv\Scripts\python.exe -m scripts.evaluate_mappo --checkpoint outputs/point_defense_validation/smoke_train/checkpoints/final.pt --episodes 3 --output-dir outputs/point_defense_validation/smoke_evaluation_final
.\.venv\Scripts\python.exe -m scripts.evaluate_mappo --checkpoint outputs/point_defense_validation/smoke_train/checkpoints/best.pt --episodes 3 --output-dir outputs/point_defense_validation/smoke_evaluation_best
.\.venv\Scripts\python.exe -m scripts.visualize_mappo --checkpoint outputs/point_defense_validation/smoke_train/checkpoints/final.pt --output-dir outputs/point_defense_validation/smoke_visualization
```

CPU、seed 0；actor `[128,128]`、critic `[256,256]`、rollout 512、epochs 4、minibatch 64、
actor/critic LR 0.0003/0.001、gamma 0.99、lambda 0.95、clip 0.2、entropy coefficient 0.01、
value coefficient 0.5、max grad norm 0.5。输入归一化仍关闭，reward scale=1。
全部其余默认参数见保存的 `smoke_train/training_config.yaml`，无静默覆盖。

实际完成：512/512 steps、1 update、1 个完整 500 步 timeout、下一回合 12 步未完成。
完整回合回报 327.2139、min distance 34.9441 m、碰撞 16 步。
Actor loss=-0.0056552、value loss=128.5047531、entropy=1.7884906；损失、梯度指标与模型参数均有限。
`final.pt` 和 `best.pt` 均保存并重新加载通过；仅一个 update，二者 actor 权重相同。
Checkpoint 的未完成回合长度为 12，没有伪报为完整回合。
周期存档间隔原为 10 updates，本次只有 1 update，因此不应期待 `update_000010.pt`。

Final/best 各实际做 3 回合 deterministic evaluation，结果相同：

| 指标 | Final / Best |
| --- | ---: |
| `interception_success_rate` | 0% |
| `enemy_attack_success_rate` | 0% |
| `timeout_rate` | 100% |
| 平均回报 | -163.1296 |
| 平均长度 | 500 steps |
| 平均 attacker-base 终距 | 78.6004 m |
| 平均最小敌我距离 | 13.1745 m |
| 平均成功拦截时间 | null，无成功 |
| 碰撞步比例 | 25.4%，每回合 127 步 |

固定初始化、规则 attacker 和确定性 actor 下，同一模型三个回合重复相同轨迹；
这些结果仅验证模型消费、任务指标、CSV/NPZ/JSON 和绘图兼容，不能当作泛化或有效学习证据。
本轮没有进一步训练、改变配置或挑选更好 seed。

## 12. 修改文件、产物与后续确认

| 文件 | 目的 |
| --- | --- |
| `configs/baseline_3a1c.yaml` | Mission、固定初态、attacker 配置、数值 workspace、失败惩罚 |
| `envs/attacker_policy.py`（新增）、`envs/__init__.py` | 新规则攻击策略及导出 |
| `envs/pursuit_env.py` | 任务配置、两类终止、捕获优先、失败奖励、mission info 和历史兼容 |
| `algorithms/mappo/normalization.py` | 新旧物理配置名称兼容，未改缩放公式 |
| `utils/evaluation.py` | 完整任务结局、三率、基地终距、成功时间与轨迹字段 |
| `scripts/train_mappo.py` | 正确区分两类 terminal，训练日志记录 mission metrics |
| `scripts/random_rollout.py`、`evaluate_mappo.py`、`visualize_mappo.py` | 随机运行、评价及基地/攻击球绘图 |
| `tests/test_attacker_policy.py`、`test_point_defense.py`、`test_mission_evaluation.py`（新增） | 策略、任务和 pipeline 回归 |
| `tests/conftest.py`、`test_pursuit_env.py`、`test_evader_policy.py`、`test_boundary_steering.py`、`test_mappo.py`、`test_mappo_pipeline.py` | 新默认任务与独立 legacy 覆盖 |
| `README.md`、`BASELINE_SPEC.md`、本报告 | 当前定义、运行方法、历史区别与实际限制 |

产物全部位于 `outputs/point_defense_validation/`：有效 YAML、原始 NPZ、逐回合 CSV/JSON、
3D 图与未平滑距离曲线、受控终局检查、manifest、`implementation_audit.json`、
smoke train/evaluation/visualization 和 `smoke_verification.json`。
`envs/dynamics.py`、旧 `envs/evader_policy.py`、网络/PPO/buffer、训练 YAML 和 dynamics tests 未因本轮修改。

当前仍需确认的是 attacker 到达控制的建模定义：给定方向加速度公式与保留物理映射能产生进攻意图，
但在完整无威胁回合中不能保证进入攻击球。是否允许加入速度/制动控制，应先作为明确任务决策讨论，
本轮未实施。其他限制是无地面/连续接触、没有真实探测过程、actor 缺乏基地关系信息、
jerk 隐藏历史及固定初始化无法证明泛化。上述问题没有通过改 reward、扩维或降低物理限制来规避。
