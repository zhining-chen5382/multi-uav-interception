# v09-r04：92,160 步成功模型直接进入 ±2° 的课程训练

本轮按用户最新要求跳过 0° 续训，从已验证成功的模型直接进入 ±2°，计划路线为 ±2° → ±5° → ±10° → ±30° → ±60° → ±90°。本报告只记录实际执行的阶段，不将计划视为完成。

## 来源、配置与步数含义

- Run：`v09-r04`；输出 `outputs/v09_来袭方向课程/v09-r04_20261109_seed000`；起始 UTC `2026-11-09T13:55:56.182351+00:00`。
- Git `c6fe7e7c64e62425c079960b3895615b6f274113`，dirty=True，seed=0，device=cpu，线程=1。
- 源模型 `outputs/v09_来袭方向课程/v09-r03_20261109_seed000/checkpoints/update_000180.pt`；SHA256 `7e665e052f102ae017e6b3243bc79600ca324d7931fffe93358ed4c6643c672e`；它已训练 92160 步，不是从零初始化。
- Actor、Critic、双 Adam 和完整全局 RNG 与源文件逐项精确一致；源未完成回合的 237 步不续接，环境重置为当前方位分布的新回合。
- 所有本轮 `env_steps`、图表和 registry 模型步数表示**新增训练工作量**。新增 0 步的 best 表示导入的已训练 92160 步策略。`lineage_env_steps = 92160 + 新增工作量` 仅表示总投入；选历史 best 回滚后不能把它解释为当前策略连续优化路径长度。
- 独立配置 `configs/mappo_direction_curriculum_warm_start.yaml`；每阶段新增上限 100k，六阶段合计最多新增 600k，未达标即停止。原 v08、v09-r01/r02/r03 结果与源权重均保留。
- 环境、Direct Attacker、奖励、原始21D/24D输入、动作、动力学、网络和 PPO 配置与源模型逐项核验一致；未修改任务难度。

```yaml
training:
  total_env_steps: 600000
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
curriculum:
  stage_half_angles_degrees:
  - 2.0
  - 5.0
  - 10.0
  - 30.0
  - 60.0
  - 90.0
  max_stage_steps: 100000
  evaluation_interval_steps: 10000
  evaluation_episodes: 100
  promotion_capture_rate: 0.8
  required_consecutive_evaluations: 2
  evaluation_seed: 1000
  stage_sampling_seed_stride: 10000
  monitor_previous_stage: true
  collision_selection_metric: collision_episode_rate
  zero_angle_evaluation_episodes: 1
  zero_angle_promotion_capture_rate: 1.0
  evaluation_timing: after_full_rollout
warm_start:
  checkpoint: outputs/v09_来袭方向课程/v09-r03_20261109_seed000/checkpoints/update_000180.pt
  expected_sha256: 7e665e052f102ae017e6b3243bc79600ca324d7931fffe93358ed4c6643c672e
  previous_stage_half_angle_degrees: 0.0
```

## 评估与晋级规则

- 本轮启用 after_full_rollout：跨过每 10k 阈值后，在第一次完整 512 步 PPO 更新完成时评估，例如 10240、20480、30208；不会为对齐评估截成272步。阶段/总硬预算尾部仍可不足512步，实际末次权重会评估。旧 exact_steps 配置行为保留。
- 当前阶段固定100个均匀角度确定性评估，连续两次捕获率≥80%才晋级；按捕获率、低失守率、短成功时间、低最小距离、低碰撞率选历史 best-eval，不按 reward 选。
- 初始导入策略在 ±2° 的捕获率为 41%，平均最小距离 7.367m；原 0° 监控捕获率 100%。初始评估只作为 best 候选，不计连续晋级。
- 首阶段持续监控原0°固定场景；后续阶段监控前一阶段固定集。监控不影响晋级或当前阶段 best 排序。升级继承历史 best 的双网络、双 Adam 和全局 RNG。

## 实际训练结果

状态 `stopped_stage_budget`；实际新增 **100,000 步、196 updates、355 个完整回合、4 次训练采样捕获**；实际进入 1 个阶段。尚未训练的角度：[5.0, 10.0, 30.0, 60.0, 90.0]。

| 阶段 | 新增阶段步数 | 状态 | best新增总步数 | best捕获率 | final捕获率 | 连续达标 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| ±2° | 100000 | stopped_stage_budget | 0 | 41% | 0% | 0 |

## 阶段 1：±2°

| 新增总步数 | 评估角色/角度 | 捕获率 | 失守率 | 成功平均时间(s) | 平均最小距离(m) | 碰撞回合率 | 连续达标 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | initial / ±2.0° | 41% | 59% | 17.307317073170733 | 7.367 | 36% | 0 |
| 0 | initial_previous / ±0.0° | 100% | 0% | 17.3 | 3.525 | 0% | 不参与 |
| 10240 | current / ±2.0° | 0% | 100% | 无成功 | 102.780 | 11% | 0 |
| 10240 | previous / ±0.0° | 0% | 100% | 无成功 | 105.582 | 0% | 不参与 |
| 20480 | current / ±2.0° | 0% | 100% | 无成功 | 36.764 | 2% | 0 |
| 20480 | previous / ±0.0° | 0% | 100% | 无成功 | 35.395 | 0% | 不参与 |
| 30208 | current / ±2.0° | 0% | 100% | 无成功 | 80.402 | 12% | 0 |
| 30208 | previous / ±0.0° | 0% | 100% | 无成功 | 81.032 | 0% | 不参与 |
| 40448 | current / ±2.0° | 0% | 100% | 无成功 | 130.359 | 0% | 0 |
| 40448 | previous / ±0.0° | 0% | 100% | 无成功 | 110.360 | 0% | 不参与 |
| 50176 | current / ±2.0° | 0% | 100% | 无成功 | 189.564 | 100% | 0 |
| 50176 | previous / ±0.0° | 0% | 100% | 无成功 | 197.008 | 100% | 不参与 |
| 60416 | current / ±2.0° | 0% | 100% | 无成功 | 62.224 | 4% | 0 |
| 60416 | previous / ±0.0° | 0% | 100% | 无成功 | 39.323 | 0% | 不参与 |
| 70144 | current / ±2.0° | 0% | 100% | 无成功 | 38.924 | 0% | 0 |
| 70144 | previous / ±0.0° | 0% | 100% | 无成功 | 39.203 | 0% | 不参与 |
| 80384 | current / ±2.0° | 0% | 100% | 无成功 | 129.889 | 0% | 0 |
| 80384 | previous / ±0.0° | 0% | 100% | 无成功 | 152.676 | 0% | 不参与 |
| 90112 | current / ±2.0° | 0% | 100% | 无成功 | 98.673 | 0% | 0 |
| 90112 | previous / ±0.0° | 0% | 100% | 无成功 | 106.114 | 0% | 不参与 |
| 100000 | current / ±2.0° | 0% | 100% | 无成功 | 139.874 | 37% | 0 |
| 100000 | previous / ±0.0° | 0% | 100% | 无成功 | 93.950 | 0% | 不参与 |

训练原始样本首/末各20%均值（精确窗口大小在 analysis_summary.json，不是平滑曲线）：

- episode_reward：-226.221 → -220.883；范围 [-568.992, 444.406]，每端 71 个样本。
- min_distance：43.4525 → 104.997；范围 [3.93323, 214.136]，每端 71 个样本。
- episode_length：279.535 → 282；范围 [174, 282]，每端 71 个样本。
- actor_loss：-0.00576322 → -0.00805398；范围 [-0.0186839, -0.000106244]，每端 40 个样本。
- value_loss：155.607 → 24.6677；范围 [0.711967, 2336.15]，每端 40 个样本。
- entropy：1.62875 → 1.67298；范围 [1.45744, 1.77356]，每端 40 个样本。
- approx_kl：0.00510713 → 0.00683061；范围 [0.00282259, 0.0134639]，每端 40 个样本。
- actor_grad_norm：3.58023 → 5.15732；范围 [1.47205, 10.8557]，每端 40 个样本。
- critic_grad_norm：477.111 → 208.829；范围 [28.6743, 1319.02]，每端 40 个样本。

## 末阶段 best / final

| 模型 | 新增总步数 | 预训练来源步数 | 捕获率 | 失守率 | 平均reward | 平均回合步数 | 成功时间(s) | 平均最小距离(m) | 碰撞回合率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best | 0 | 92160 | 41% | 59% | 4.609067 | 237.3 | 17.307 | 7.367446 | 36% |
| final | 100000 | 92160 | 0% | 100% | -277.521336 | 282.0 | 无成功 | 139.874481 | 37% |

best 是末阶段历史 best-eval，可能仍是新增0步的导入模型；final 使用实际阶段末 `last.pt`，未改名或覆盖权重。固定100角度网格属于选模/晋级集，不是独立留出测试；原0°单例只表示固定轨迹是否保留。

## 视频与轨迹演化

两段完整视频使用相同角度 -0.020202°、seed 1049；选择方法：Middle of best-eval's 41 successful fixed-grid angles; both policies use this identical angle and seed. This illustrative success is not a representative capture-rate estimate.
- best：[完整仿真视频](../outputs/v09_来袭方向课程/v09-r04_20261109_seed000/videos/best/best_full_episode.mp4)；[来源manifest](../outputs/v09_来袭方向课程/v09-r04_20261109_seed000/videos/best/best_video_manifest.json)。本视频单例 Interception Success，173步、最近距离3.470360m、reward444.542535；不与上表100角度均值混用。
- final：[完整仿真视频](../outputs/v09_来袭方向课程/v09-r04_20261109_seed000/videos/final/final_full_episode.mp4)；[来源manifest](../outputs/v09_来袭方向课程/v09-r04_20261109_seed000/videos/final/final_video_manifest.json)。本视频单例 Enemy Attack Success，282步、最近距离93.086092m、reward-182.393925；不与上表100角度均值混用。

两段视频均保留初始至终止全轨迹并通过整段解码检查。成功角度示例不能代替上表整体固定网格捕获率。
- 优先查看 [含初始点分布的XY轨迹演化图](../outputs/v09_来袭方向课程/v09-r04_20261109_seed000/trajectory_evolution/with_initial_distribution/trajectory_evolution_xy.png)；[实际初始点CSV](../outputs/v09_来袭方向课程/v09-r04_20261109_seed000/trajectory_evolution/with_initial_distribution/initial_distribution.csv)。增强图保留共同固定0°测试轨迹，并附实际已完成训练回合的敌机初始点分布；未训练的±5°至±90°不能当作实际采样结果。
- [XY轨迹演化主图](../outputs/v09_来袭方向课程/v09-r04_20261109_seed000/trajectory_evolution/trajectory_evolution_xy.png)；[逐checkpoint指标CSV](../outputs/v09_来袭方向课程/v09-r04_20261109_seed000/trajectory_evolution/trajectory_evolution_metrics.csv)。
- 图表从浅到深表示新增实际训练步数，源92160步单独注明，新增0不是未训练网络。所有checkpoint共用原始0°、seed1000完整确定性回合；这是一种共同场景行为对照，不是当前角度区间的成功率估计。

## 曲线

全部来自实际 CSV，保留原始曲线、不做滑动平均或平滑；初始评估独立标星，不算训练后晋级点。
- [evaluation_capture_rate](../outputs/v09_来袭方向课程/v09-r04_20261109_seed000/plots/evaluation_capture_rate.png)
- [evaluation_min_distance](../outputs/v09_来袭方向课程/v09-r04_20261109_seed000/plots/evaluation_min_distance.png)
- [training_episodes](../outputs/v09_来袭方向课程/v09-r04_20261109_seed000/plots/training_episodes.png)
- [actor_value_loss](../outputs/v09_来袭方向课程/v09-r04_20261109_seed000/plots/actor_value_loss.png)
- [entropy](../outputs/v09_来袭方向课程/v09-r04_20261109_seed000/plots/entropy.png)

## 测试、保护与限制

- 本轮实际完整 pytest：432 passed in 57.96s，[原始日志](../outputs/v09_来袭方向课程/v09-r04_20261109_seed000/preflight/pytest.log)。
- 33 个checkpoint（含模型、Adam、RNG与环境状态）及读取的数值CSV无 NaN/Inf；数值有限不等于策略性能稳定。
- 复查 7567 个历史输出文件保持原SHA；已有源码/配置中只允许本轮声明的调度、媒体、索引和测试变更。环境、动力学、奖励、观测、网络和原权重保持不变。
- `warm_start` 与现有 `continuation` 不能同时设置；本轮可阶段间完整继承，但尚不能直接沿用旧同run续训入口继续这个warm run。原无warm的续训功能保持兼容。
- 本轮只使用一个训练seed。初态/角度分布、离散捕获检测、原始观测和完整物理限制均保留；失败结果不通过改奖励、网络、课程门槛或任务难度掩盖。

新增纯制图模块另行专项 **3 passed in 0.82s**，[制图测试日志](../outputs/v09_来袭方向课程/_warm_start_checks_20261109/initial_distribution_pytest_final.log)；这不是重新执行全量435项测试。

## 结论

本轮在 ±2° 阶段用尽100k新增预算，未达到连续两次80%，因此按规则停止，未进入 ±5°。导入策略在±2°初始捕获率41%，训练末为0%；这表明原固定场景成功模型并不自动保证角度课程能稳定改善。
±2° 的初始41%在首个新增10240步评估已降至0%，本轮全部10次训练后评估均为0；平均最小距离由初始7.367m变为首评102.780m、末评139.874m。它体现了成功行为丢失，不能解释为课程训练改善。
原0°监控从导入时100%下降到所有10次训练后检查均失败；在这个原始固定场景上存在明确遗忘。此结论不等于对所有旧场景的统计性证明。
若0°监控从初始成功变为后续失败，只能据同一固定场景确认遗忘；若current捕获长期为零，则不能把value loss下降或reward上升描述成学会拦截。可能原因包括策略更新丢失稀疏成功行为、固定初态策略对角度变化敏感，以及现有局部观测/共享团队反馈限制；这些是分析方向，本轮未据此改算法。

## 运行

```powershell
.\.venv\Scripts\python.exe -B -u -m scripts.train_direction_curriculum --config configs/mappo_direction_curriculum_warm_start.yaml
```

主要改动文件：`scripts/train_direction_curriculum.py`、`utils/curriculum_warm_start.py`、独立warm YAML、warm流程/媒体/索引测试、只读媒体与轨迹图标注、README/BASELINE_SPEC和本报告。没有执行Git add、commit、merge或push。
