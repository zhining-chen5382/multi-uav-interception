# Direct Attacker Baseline 制导修改与实际验证

日期：2026-10-24。当前默认任务保持 3A1C 点防御，仅修改 attacker guidance。

**结论：Direct Attacker 从原 500 m 固定初态出发，在 28.2 s / 282 步进入严格的 10 m 攻击球，入球速度约 20 m/s。**
单机全程基地距离下降，无绕圈、长时间振荡、到达前过冲、timeout、hard clipping 或 NaN/Inf。
完整 pytest、普通随机运行、512 步 MAPPO smoke、checkpoint 加载、独立评估及绘图均实际通过。
短训练不代表我方已经学会拦截；本轮没有运行 100k。

## 1. 任务与修改范围

三架我方拦截机保护原点基地，一架 Direct Attacker 持续朝基地飞行，不主动规避 pursuers。
直到我方捕获、敌机进入攻击球或达到原 episode 上限。基地 `[0,0,0]`、detection radius 500 m、
attack radius 10 m、capture radius 5 m、双方初态与物理限制、dt=0.1 s、500 步上限均不变。

没有修改 reward、termination、21D observation、24D global state、pursuer 动作处理、
MAPPO 网络、初始化或训练超参数，没有增加 arrival braking、悬停、预测、随机初始化或机动策略。

Git HEAD 只读记录为 `23bf55576cca27944b4ee2ae5c53a70295c7dac3`，实际运行包含未提交工作区。
代码/config SHA256 见输出 manifest；不能仅凭 HEAD 还原此次未提交改动。
未执行 `git add`、`commit`、`merge`、`push`。

## 2. 旧策略问题与新制导公式

旧 `GoalDirectedAttackerPolicy` 将目标方向与 pursuer repulsion 合成后直接生成加速度。
目标项不根据当前速度修正侧向运动，在已有惯性、jerk 及水平/垂直加速度映射下会过冲绕行。
上一轮无近威胁对照最近距基地约 19.28 m，随后绕行并 timeout，详见历史
[POINT_DEFENSE_TASK_REPORT.md](POINT_DEFENSE_TASK_REPORT.md)。本轮不将该结果混为新策略结果。

新增 `DirectAttackerPolicy`，使用：

```text
goal_vector = base_position - attacker_position
goal_direction = goal_vector / norm(goal_vector)
desired_velocity = attack_speed * goal_direction
acceleration_command = velocity_gain * (desired_velocity - current_velocity)
```

Policy 返回原始物理加速度，不再对其归一化，不调用 `scale_action`，不重复限幅。
原 `integrate()` 统一施加水平/垂直加速度限制、jerk、水平/爬升/下降速度限制和位置更新。
因此横向速度不一致时会产生速度修正，而非持续积累横向运动。

默认 YAML 唯一变化是 `attacker.policy`：

```yaml
mode: direct
attack_speed: 20.0
velocity_gain: 1.0
distance_epsilon: 0.000001
```

`attack_speed` 是期望三维速度模长 20 m/s，与现最大水平速度数值相同。
`velocity_gain=1 s^-1` 对应未限幅连续模型的一阶速度误差时间常数 1 s，简单、可解释；
实际加速仍受原物理限制。本轮只使用这一组暂定值，没有搜索或择优重跑。

Direct Attacker 完全不读取 pursuer 状态，不使用 avoidance 或 soft boundary steering。
Direct 模式拒绝原 goal 权重、非零 avoidance 权重与已有 boundary steering 参数，避免意外混入旧制导项。
不按距离降低 `attack_speed`，不要求停在原点；入球即结束。
距离不超过 epsilon 时的零目标方向只用于防止除零，正常任务早已在 `<10 m` 时终止。

旧 GoalDirected、evader repulsion 和边界模块均保留，用于原无 `mode` 配置和 checkpoint 快照复现，
本轮没有新增 `evasive` mode，也没有将其用作默认 baseline。

## 3. 全量测试与保护项审计

首先实际执行：

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
$env:MPLBACKEND = "Agg"
.\.venv\Scripts\python.exe -m pytest -q
```

**203 passed in 12.92s**。保留此前 156 项覆盖，新增 Direct Attacker 的 47 项参数化测试。

新增覆盖期望速度指向、侧向速度修正、pursuer 不影响指令、原始加速度可超限且由 dynamics 接管、
近攻击球仍保持期望速度、有限值/非法输入、完整 500 m 到达而非只检查前几秒、全程距离下降、
零裁剪、真实攻击终止、21D/24D、YAML 参数以及旧配置兼容。原 dynamics 与 MAPPO pipeline 测试继续通过。

修改前快照位于 `outputs/direct_attacker_validation/before_guidance_change/`。
`implementation_audit.json` 独立确认：

- 环境 YAML 除 `attacker.policy` 外完全相同。
- `envs/dynamics.py`、旧 evader、全部 MAPPO 实现/训练 YAML、训练/评估/原绘图脚本逐字节不变。
- `PursuitEnv` 除 `_configure()` 的策略选择外，所有方法 AST 相同，包含 reward、observation、state、
  termination、reset、边界处理及 pursuer 控制流程。

## 4. 单架攻击机 sanity check

全量 pytest 通过后，实际运行：

```powershell
.\.venv\Scripts\python.exe -m scripts.sanity_direct_attacker --config configs/baseline_3a1c.yaml --output-dir outputs/direct_attacker_validation/sanity
```

此入口只模拟一架攻击机，直接使用生产策略与生产 `integrate()`，没有 pursuers、没有拦截干扰，
不是移动我方初态或修改任务半径来制造成功。初态仍为 `[497.49371855331,0,50]` m、零速度/加速度历史。
保留原 bounds hard clip 和严格更新后 `<10 m` 判定，入球后不再积分。

| 实际指标 | 结果 |
| --- | ---: |
| 初始基地距离 | 500.000000 m |
| 进入前一步距离（第 281 步） | 10.261147 m |
| 最小 / 最终基地距离（第 282 步） | 8.261147 m |
| 入球时间 | **28.2 s** |
| 入球三维速度 | **20.000000 m/s** |
| 入球水平速度 | 19.918701 m/s |
| 入球垂直速度 | -1.801484 m/s |
| 基地距离增加步数 | **0** |
| Hard clipping | **0** |
| Timeout | False |
| 数值有限性与物理限制审核 | 全部通过 |

全程距离严格下降，速度升至约 20 m/s 后保持；没有近基地绕圈或振荡，没有主动减速停车。
轨迹相对初始径向直线最大偏离约 4.234 m，来自起步速度跟踪及不同方向约束的修正，
不是回绕；原始飞行路径长约 492.127 m。
这里的“无过冲”指本次入球前已记录轨迹，未模拟终止后穿过原点的运动。

已保存并实际打开检查完整轨迹与攻击球局部放大图，以及未平滑的距离/速度曲线：

![Direct Attacker 单机完整轨迹与入球局部图](../outputs/direct_attacker_validation/sanity/trajectory_3d.png)

![距离和速度原始曲线](../outputs/direct_attacker_validation/sanity/distance_speed_raw.png)

`trajectory.npz` 保留初始及全部更新后状态、applied acceleration、desired velocity、raw acceleration command、
时间、距离和裁剪布尔值；manifest 明确 `N+1` 状态与 `N` 指令的对齐关系。

## 5. 普通 point-defense random rollout

单机入球及图像确认后执行当前默认配置，三个固定 seed：

```powershell
.\.venv\Scripts\python.exe -m scripts.random_rollout --config configs/baseline_3a1c.yaml --seed 0 --episodes 3
```

| Seed | 步数 / 结局 | Reward | 最小敌我距离 m | 碰撞步数 | Hard clips |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | 282 / Enemy Attack Success | 370.9360 | 24.6522 | 0 | 0 |
| 1 | 282 / Enemy Attack Success | 323.1922 | 60.2238 | 0 | 0 |
| 2 | 282 / Enemy Attack Success | 274.2582 | 30.5946 | 12 | 0 |

三个回合均在 28.2 s、距基地 8.261147 m 结束。拦截率 0%，敌方攻击成功率 100%，timeout rate 0%。
这表示随机我方机未能保护基地，不是攻击机制导失败。平均 reward 约 322.7955，
保留的 progress 项可因敌机主动接近而给正回报，因此正总回报不代表防御成功。

另外使用等价随机动作保存原始轨迹，累计 846 步独立验证数值、21D/24D、速度/加速度/jerk、积分和裁剪计数，全部通过。
三个随机回合中，attacker 的全部 283 个状态和 applied acceleration 与单机 sanity **逐元素完全相同**，
直接验证了 pursuer 不影响当前制导。保存脚本还与实际 CLI 输出逐字段核对，结果一致。

复现保存过程：

```powershell
.\.venv\Scripts\python.exe outputs/direct_attacker_validation/validate_random_rollout.py
```

![当前随机点防御完整回合](../outputs/direct_attacker_validation/random/seed0_trajectory_3d.png)

现有四机绘图函数已具备基地、10 m 攻击球、四机轨迹、起止点和任务结局信息；本轮复用它生成新制导的实际结果，
未为绘图修改环境。标题显示 Enemy Attack Success、282 steps、最小敌我距离和最终基地距离。

## 6. MAPPO smoke、checkpoint 与评价

完成上述验证后仅运行一次 **512 步**手动 smoke，使用默认网络、PPO、seed 0 和 CPU：

```powershell
.\.venv\Scripts\python.exe -m scripts.train_mappo --config configs/mappo_baseline.yaml --total-steps 512 --output-dir outputs/direct_attacker_validation/smoke_train
.\.venv\Scripts\python.exe -m scripts.evaluate_mappo --checkpoint outputs/direct_attacker_validation/smoke_train/checkpoints/final.pt --episodes 3 --output-dir outputs/direct_attacker_validation/evaluation_final
.\.venv\Scripts\python.exe -m scripts.evaluate_mappo --checkpoint outputs/direct_attacker_validation/smoke_train/checkpoints/best.pt --episodes 3 --output-dir outputs/direct_attacker_validation/evaluation_best
.\.venv\Scripts\python.exe -m scripts.visualize_mappo --checkpoint outputs/direct_attacker_validation/smoke_train/checkpoints/final.pt --output-dir outputs/direct_attacker_validation/visualization
```

512/512 步正常完成、1 次 PPO update、1 个完整 282 步敌方攻击成功回合，下一回合 230 步未完成。
后者仅存为 partial episode，没有误记为完整回合。完整回合 reward=282.6965，碰撞 16 步。

Actor loss=-0.0086782、value loss=326.9050021、entropy=1.7899636。
损失、梯度指标与网络参数均有限；`final.pt`、`best.pt` 均已生成并独立重新加载。
本次只有一次 update，final/best actor 权重相同。原周期 checkpoint 间隔为 10 updates，
不为 smoke 临时改间隔，因此本次没有周期 update 文件，final/best 均正常。
实际有效配置与来源记录在 `smoke_train/`，除 CLI 总步数和输出目录外，默认训练超参数均未改。

Final/best 各实际做 3 个 deterministic evaluation 回合，结果相同：

| 指标 | Final / Best |
| --- | ---: |
| Interception success rate | 0% |
| Enemy attack success rate | 100% |
| Timeout rate | 0% |
| 平均 episode length | 282 steps |
| 平均 attacker-base terminal distance | 8.261147 m |
| 平均 reward | 337.8725 |
| 平均最小敌我距离 | 52.9803 m |
| 碰撞步数 | 0 |
| 平均成功拦截时间 | null，无成功 |

每个模型的三个回合因固定初始化而重复相同轨迹，不是泛化评估。
攻击机轨迹也与独立单机 sanity 逐元素相同。评价、CSV/JSON/NPZ、环境快照与独立 3D 图均正常生成，
验证详情见 `smoke_verification.json`。仅一次 PPO 更新不能证明学会拦截或协同。

## 7. 修改文件、产物与限制

本轮源文件/文档清单：

| 文件 | 原因 |
| --- | --- |
| `configs/baseline_3a1c.yaml` | 仅切换 direct mode，配置 attack speed/gain，移除默认 goal/avoidance 权重 |
| `envs/attacker_policy.py` | 新增 DirectAttackerPolicy；保留旧类供快照复现 |
| `envs/pursuit_env.py` | 仅新增策略模式选择和互斥参数校验 |
| `envs/__init__.py` | 导出新策略 |
| `scripts/sanity_direct_attacker.py`（新增） | 无 pursuer 的生产动力学验证与独立轨迹图 |
| `tests/test_direct_attacker.py`（新增） | 新制导、完整到达、物理约束及兼容测试 |
| `README.md`、`BASELINE_SPEC.md` | 当前默认制导、命令、历史区别和局限 |
| 本报告（新增） | 实际结果与可复现产物索引 |

输出目录：`outputs/direct_attacker_validation/`。
包含 `sanity/`、`random/`、`smoke_train/`、`evaluation_final/`、`evaluation_best/`、`visualization/`、
修改前快照、`implementation_audit.json`、`smoke_verification.json` 和随机验证脚本。
旧点防御与 100k 实验产物未覆盖。

仍有的限制：只有固定方向/固定初态的到达验证；不同侧向初速度或陡直几何会受既有物理限制，
本次没有宣称全状态到达保证。Direct Attacker 不覆盖敌方规避能力；gain=1 仍是暂定值。
Actor 缺少明确基地关系、jerk 历史未入既定输入、离散步接触检测、无地面模型、
progress reward 与任务成败可能不同向等原有限制仍在。当前检查未发现需要再改参数才能跑通的工程问题。
