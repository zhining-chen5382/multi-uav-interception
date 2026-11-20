# v09-r03：固定 0° 场景的 v08 更新节奏复现验证

本轮验证同一环境下，v09 固定 0° 环境在对齐完整 rollout 更新后，能否从零复现 v08 的成功模型。它是独立复现实验，不是完整课程训练，也没有修改原课程入口。

## 实验与配置

- Run：`v09-r03`；输出 `outputs/v09_来袭方向课程/v09-r03_20261109_seed000`。参考 `outputs/v08_捕获奖励500/v08-r01_20261108_seed000`；短程检查 `outputs/v09_来袭方向课程/_zero_reproduction_checks_20261109/short_10240`。
- 实际起始 UTC：`2026-11-09T13:19:15.235292+00:00`；Git `c6fe7e7c64e62425c079960b3895615b6f274113`，dirty=True；seed=0，device=cpu，线程数=1。
- 原始参考配置、网络、PPO、归一化、物理环境均核验一致。独立配置为 `configs/mappo_zero_direction_reproduction.yaml`；仅预算和输出/复现记录不同，未加载旧模型初始化。
- 使用 `FixedZeroDirectionEnv`（原 v09 `DirectionCurriculumEnv` 的 0° 子类）临时提供给原 `scripts.train_mappo.train`；`finally` 恢复绑定，不修改任一 trainer、环境、奖励或算法。
- 目标固定为原始 0°，敌机位置 `[497.49371855331, 0.0, 50.0]`；我方初始位置 `[[1.5, 0.0, 1.0], [-0.75, 1.25, 1.0], [-0.75, -1.25, 1.0]]`，双方零初速。
- 基地 `[0.0, 0.0, 0.0]`；workspace [-750.0, -750.0, -750.0] 至 [750.0, 750.0, 750.0]；dt=0.1s、最大回合 500 步，capture<5.0m、attack<10.0m。
- 环境使用未归一化 21D Actor 观测与 24D Critic 状态，动作是三维加速度；Direct Attacker、动力学和原奖励完整保留。实际 YAML 快照见输出目录。

```yaml
training:
  total_env_steps: 100000
  rollout_steps: 512
  log_interval_updates: 1
  checkpoint_interval_updates: 10
  capture_rate_window: 50
network:
  actor_hidden_sizes:
  - 128
  - 128
  critic_hidden_sizes:
  - 256
  - 256
  initial_log_std: -0.5
  min_log_std: -5.0
  max_log_std: 2.0
  hidden_gain: 1.4142135623730951
  actor_output_gain: 0.01
  critic_output_gain: 1.0
ppo:
  actor_learning_rate: 0.0003
  critic_learning_rate: 0.001
  adam_epsilon: 1.0e-05
  gamma: 0.99
  gae_lambda: 0.95
  clip_ratio: 0.2
  update_epochs: 4
  minibatch_size: 64
  entropy_coefficient: 0.01
  value_loss_coefficient: 0.5
  max_grad_norm: 0.5
  advantage_normalization: true
  normalization_epsilon: 1.0e-08
normalization:
  observation: false
  global_state: false
  reward_scale: 1.0
```

## 更新节奏差异与短程验证

原 v09-r02 与 v08 前 19 次更新的共同日志逐字一致；第 20 次开始分叉：v08 在 10240 步用 512 条 transition 更新，v09 为对齐 10k 评估，在 10000 步仅用 272 条更新。
本次先单独运行 10240 步（20×512），短程所有共同日志、checkpoint 状态逐项精确相等=True；随后重新从相同 seed 从零运行单次 100k 验证，没有修改配置反复挑选结果。

## 完整训练复现结果

实际完成 **100,000 steps、196 updates、356 完整训练回合、5 次随机训练捕获**。短程检查额外消耗 10240 步，单独保留，不计入这轮 100k 训练。
完整 rollout 的共同前缀为 **99,840 步**；19 个共同周期 checkpoint 的 Actor、Critic、双 Adam、全局 RNG、环境、partial episode 等状态精确相等。metadata、预算/路径及 PT 序列化字节不作为权重相等判据。

| 日志 | 精确比较行数 | 相等 |
| --- | ---: | --- |
| updates.csv | 195 | True |
| training_metrics.csv | 355 | True |

| 关键 checkpoint | 步数 | 与原 v08 精确相等 |
| --- | ---: | --- |
| update_000180.pt | 92160 | equal |

100k 最后一批只有 **160 步**，与原 v08 300k 在下一批继续收满 512 的更新时域不同。因此该末次更新和 100k final 不纳入逐位复现前缀，不能据此误判前缀复现失败。

## 筛查、确定性评估与视频

完整筛查本轮 19 个周期 checkpoint，每个使用同一 0°、seed1000 完整确定性回合；成功的实际步数为 `[92160, 97280]`。按捕获率、低失守率、短成功时间、低最小距离、低碰撞率排序，选中 92160 步。
这里的 **best 是筛查 best-eval**，引用真实周期文件；原生 `checkpoints/best.pt` 仍按训练窗口捕获率与 reward 保存，未被覆盖。final 使用本轮真实 `checkpoints/final.pt`。
两模型各评估 50 回合，seed1000..1049，采用相同固定初态。它们是相同确定性轨迹的重复核对，不能解释为 50 个独立场景或泛化成绩。

| 模型 | 实际步数 | 捕获 | 失守 | reward | 回合步数 | 成功时间(s) | min距离(m) | 碰撞回合率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best | 92160 | 100% | 0% | 444.565869 | 173 | 17.3 | 3.525490 | 0% |
| final | 100000 | 0% | 100% | -1062.882639 | 282 | 无成功 | 43.362146 | 100% |

Best 与原 v08 的 92,160 步成功回合共有 12 个轨迹数组逐位比较，全部相等=True；逐数组结果保存在 `evaluation_and_video_summary.json`，不比较其中时间/路径 metadata。

- best：[完整视频](../outputs/v09_来袭方向课程/v09-r03_20261109_seed000/videos/best/best_full_episode.mp4)；完整解码 174 帧，包含初始帧和终止帧；[解码验证](../outputs/v09_来袭方向课程/v09-r03_20261109_seed000/videos/best/video_verification.json)。
- final：[完整视频](../outputs/v09_来袭方向课程/v09-r03_20261109_seed000/videos/final/final_full_episode.mp4)；完整解码 283 帧，包含初始帧和终止帧；[解码验证](../outputs/v09_来袭方向课程/v09-r03_20261109_seed000/videos/final/video_verification.json)。

## 轨迹演化与证据

- [XY 俯视轨迹演化主图](../outputs/v09_来袭方向课程/v09-r03_20261109_seed000/trajectory_evolution/trajectory_evolution_xy.png)；[逐 checkpoint 指标 CSV](../outputs/v09_来袭方向课程/v09-r03_20261109_seed000/trajectory_evolution/trajectory_evolution_metrics.csv)。
- 独立 `configs/trajectory_evolution_reproduction.yaml` 显式加入真实 92160/97280 步，保留原始 0°、seed1000；浅到深表示实际训练步数。新 run 保存真实初始模型，旧 v08 缺失的初始模型没有重造。
- [完整状态比较](../outputs/v09_来袭方向课程/v09-r03_20261109_seed000/reproduction_comparison.json)；[10240 步短程比较](../outputs/v09_来袭方向课程/_zero_reproduction_checks_20261109/short_10240/reproduction_comparison.json)；[全部周期模型筛查](../outputs/v09_来袭方向课程/v09-r03_20261109_seed000/checkpoint_screening/metrics.csv)。

## 测试与保护

- 本轮实际完整 pytest：**402 passed in 63.89s**，[原始日志](../outputs/v09_来袭方向课程/v09-r03_20261109_seed000/preflight/pytest.log)。
- 新增测试覆盖固定 0° 环境、临时绑定异常恢复、未截短训练状态比较、配置不匹配拒绝与比较器能检出差异。训练两份主要 CSV 的全部数值列均有限。
- 报告生成前实际核对 5698 个历史输出文件、77 个已有源码/配置文件，SHA256 全部保持不变。新增输出、验证入口/测试/独立配置及文档单独保存。
- 现有 v08、v09-r01/r02、原课程默认 YAML 和训练核心均未改动；registry 只增加本轮 run、模型、视频/图和测试记录，不改旧行。没有执行 Git add/commit/merge/push。

## 结论与限制

**本次在相同 seed/runtime 下成功复现 v08 的共同训练前缀及成功轨迹。** 证据支持先前失败并非初始几何或奖励配置不一致；对齐完整 rollout 更新后，v09 的 0° 环境能得到同一个成功策略。
这是单一 seed、固定初态的工程复现验证，不能证明策略已泛化或课程有效。现有课程入口为恰好 10k 评估而截短 rollout 的逻辑仍保留，本轮没有声称已经修正正式课程调度。
下一步应单独设计课程评估与完整 PPO rollout 的兼容方式，明确评估是在完整更新后的最近步数进行，还是在 10k 精确快照推理且保持训练 buffer 连续；再测试晋级与恢复，不直接修改研究设定或继续堆训练步数。

## 运行命令

```powershell
.\.venv\Scripts\python.exe -B -u -m scripts.verify_zero_direction_reproduction --config outputs/v09_来袭方向课程/_zero_reproduction_checks_20261109/short_training_config.yaml
.\.venv\Scripts\python.exe -B -u -m scripts.verify_zero_direction_reproduction --config configs/mappo_zero_direction_reproduction.yaml
```
