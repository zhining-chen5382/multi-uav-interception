# v10：固定 0° 的 Adam 状态与策略保持对照

本轮只做短期策略保持诊断：同一原始 v08 成功策略、同一固定场景与随机数起点，A 保留原 Adam，B 仅清空 Adam 状态。未继续 curriculum，未改学习率、奖励、网络、观测或环境。

## 实验来源与公平性

- 原始源：`outputs/v08_捕获奖励500/v08-r01_20261108_seed000/checkpoints/update_000180.pt`；SHA256 `d5d78f3045a2bb9d865028182db21a7128603e02d890cef5dfcabd0ed97609cd`，真实 92160 steps / update 180。不是 r03 复现副本。
- Git `c6fe7e7c64e62425c079960b3895615b6f274113`，dirty=True；seed=0，device=cpu，线程数=1；本次配置 [独立 YAML](../configs/mappo_policy_retention.yaml)。
- 正式诊断实际开始：北京时间 `2026-11-10 14:51:19`（UTC+8，取自metadata，不按文件修改时间推断）。
- 两组各新增 20000 步；共新增 40000 步，不能把它们相加解释为同一策略的连续训练。源92160步单列。A=`v10-r01`，B=`v10-r02`。
- Actor/Critic 权重、原全局 RNG、全部 Adam param-group 设置相同；13 组首批512步训练数组逐位相同，包含 GAE/returns、raw actions、状态和奖励。初始确定性轨迹 12 组数组亦逐位相同。
- 源文件有 237 步未结束回合，两组均丢弃该 partial，重置同一固定0°初态；这是公平的新回合初始化，不是恢复原环境的同回合续训。

| 组 | Actor/Critic | Adam 初始化 | 新增步数 | updates | 完整训练回合 | 随机训练捕获 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| A | 同一源权重 | 继承全部原状态 | 20000 | 40 | 71 | 1 |
| B | 同一源权重 | 清空state，保留参数组 | 20000 | 40 | 71 | 2 |

### Adam 与 PPO 的实际设置

- 源 actor_optimizer：7 组参数state，step=[5760.0]，lr=0.0003，betas=[0.9, 0.999]，eps=1e-05；exp_avg总L2=0.11159949，exp_avg_sq总L2=0.05778281。
- 源 critic_optimizer：6 组参数state，step=[5760.0]，lr=0.001，betas=[0.9, 0.999]，eps=1e-05；exp_avg总L2=0.08841091，exp_avg_sq总L2=0.06870704。
B 同时重置一阶矩、二阶矩及 bias-correction 计数，不只是移除“旧动量”；两组同时涉及 Actor 与 Critic 优化器，不能据此分离两者贡献。

```yaml
training:
  total_env_steps: 20000
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
retention:
  checkpoint: outputs/v08_捕获奖励500/v08-r01_20261108_seed000/checkpoints/update_000180.pt
  expected_sha256: d5d78f3045a2bb9d865028182db21a7128603e02d890cef5dfcabd0ed97609cd
  groups:
  - name: A
    reset_adam: false
  - name: B
    reset_adam: true
  evaluation_interval_steps: 2000
  evaluation_seed: 1000
```

## 评估时点语义

每新增2000步准确暂停采样做一次完整 deterministic episode，不清空或截断512步 PPO batch。中间评估使用最后一次完整更新后的权重：例如2000点对应last_update=1536、pending=464；4000点对应3584、pending=416。评估恢复训练 RNG 与 train/eval 模式，训练环境与 buffer 不动。
硬预算20000末尾仅剩32步，先完成这次末尾更新再评估，故最后点last_update=20000。中途checkpoint未序列化尚未完成的on-policy buffer，仅作推理/分析快照，不承诺直接断点恢复。
每个点只有一个原始固定0°确定性场景，结果是捕获/失守，不是泛化捕获率；所有完整轨迹均保存。相邻检查点之间没有逐更新评估，不能保证中间每次更新都成功，也不能排除未观测的失效后恢复。下文的保持前缀仅指连续被观测成功的检查点。

## 全过程确定性比较

| 新增采样步数 | 最近完整更新步数 | A捕获 | A最近距离(m) | A捕获时间(s) | B捕获 | B最近距离(m) | B捕获时间(s) |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 0 | 0 | 成功 | 3.525 | 17.3 | 成功 | 3.525 | 17.3 |
| 2000 | 1536 | 成功 | 4.043 | 17.2 | 失败 | 7.767 | — |
| 4000 | 3584 | 失败 | 40.858 | — | 失败 | 7.252 | — |
| 6000 | 5632 | 失败 | 39.636 | — | 失败 | 6.699 | — |
| 8000 | 7680 | 失败 | 33.402 | — | 失败 | 35.532 | — |
| 10000 | 9728 | 失败 | 58.478 | — | 失败 | 27.608 | — |
| 12000 | 11776 | 失败 | 62.790 | — | 失败 | 10.110 | — |
| 14000 | 13824 | 失败 | 46.924 | — | 失败 | 27.555 | — |
| 16000 | 15872 | 失败 | 10.989 | — | 失败 | 47.010 | — |
| 18000 | 17920 | 失败 | 63.759 | — | 失败 | 83.410 | — |
| 20000 | 20000 | 失败 | 68.396 | — | 失败 | 64.150 | — |

### 保持时长与恢复

- A：初始后10次检查中成功1次；初始检查前缀最后确认成功的新增工作点=2000。首个观测失败检查点=4000，其相邻观测区间=[2000, 4000]，对应完整更新权重区间=[1536, 3584]；后来观测到的恢复点=[]。
- B：初始后10次检查中成功0次；初始检查前缀最后确认成功的新增工作点=0。首个观测失败检查点=2000，其相邻观测区间=[0, 2000]，对应完整更新权重区间=[0, 1536]；后来观测到的恢复点=[]。
**本次单seed、2k检查分辨率下，A 保留原 Adam 的观测成功前缀长于 B；重置没有改善该指标。两组最终都失败，A 也没有稳定保持。** 不能说退化恰在首个失败检查的采样步数发生，或断言未观测的中间更新都保持成功。

## 训练指标趋势

下列首/末各20%更新样本均值只用于趋势归纳，窗口样本数在JSON注明；图保留原始数据，没有平滑。

| 指标 | A首20% → 末20% | B首20% → 末20% | A范围 | B范围 |
| --- | ---: | ---: | ---: | ---: |
| actor_loss | -0.00840381 → -0.00879689 | -0.009187 → -0.0046814 | [-0.0165935, -0.000100226] | [-0.0176565, 5.73302e-05] |
| value_loss | 280.89 → 13.6283 | 537.141 → 42.3187 | [0.566915, 2104.89] | [0.461892, 2051.23] |
| entropy | 1.60152 → 1.60843 | 1.59807 → 1.51298 | [1.48816, 1.68196] | [1.38754, 1.7545] |
| approx_kl | 0.00665428 → 0.00560635 | 0.00598849 → 0.00552295 | [0.00231021, 0.00866193] | [0.00352027, 0.00805209] |
| clip_fraction | 0.0855103 → 0.0685832 | 0.0725708 → 0.0693563 | [0.0286458, 0.111165] | [0.0354818, 0.106608] |
| actor_std_mean | 0.624625 → 0.635209 | 0.62246 → 0.62817 | [0.62387, 0.636928] | [0.620738, 0.629562] |
| advantage_std | 15.0608 → 4.78494 | 25.2122 → 9.46772 | [0.370452, 67.8995] | [0.831967, 67.8995] |
| return_std | 30.3234 → 16.5763 | 39.4922 → 22.0267 | [1.17033, 72.6937] | [3.68754, 76.9544] |
| actor_parameter_delta_l2 | 0.426766 → 1.03964 | 0.455714 → 1.03192 | [0.225264, 1.12787] | [0.223985, 1.06424] |
| critic_parameter_delta_l2 | 3.05966 → 6.15298 | 2.78298 → 6.19102 | [1.89799, 6.28781] | [1.73376, 6.39584] |

- Actor学习率全程0.0003、Critic全程0.001，两组均未变化。
- 两组共同首512步批有1次捕获和1次失守；reward范围[-100.380615,499.976410]，value均值-31.589081，return最大499.976410，raw advantage最大535.443909、标准差67.899462。
- 首次value loss：A=2104.893752、B=2051.231845。后续value loss下降不等于成功策略被保持，可能只是更贴合后来以失败为主的on-policy数据。
- 梯度范数是实际裁剪前的minibatch加权平均；原max_grad_norm=0.5仍在每次优化前生效，不能把较大记录值解释为未裁剪梯度爆炸。
- `approx_kl` 和 `clip_fraction` 是本批旧策略相对更新策略的minibatch/epoch平均，不能当作累计相对源策略的漂移。Entropy为变换Gaussian的Monte Carlo估计，微小变化不能单独证明策略坍塌。

## 同状态的源策略漂移

独立probe只读取原成功轨迹的同一组173步观测与已保存权重，不运行新环境或新episode。原源动作被逐位重建；比较对象是相同观测上的动作，不把不同轨迹上不同观测的输出差当作策略漂移。

| 组 | 新增步数 | 同状态动作ΔL2均值 | Source→Current latent KL均值 | 源状态Critic预测ΔRMS |
| --- | ---: | ---: | ---: | ---: |
| A | 0 | 0.000000 | 0.000000 | 0.000000 |
| A | 2000 | 0.067852 | 0.008982 | 15.791180 |
| A | 4000 | 0.145555 | 0.037574 | 13.125683 |
| A | 20000 | 0.297604 | 0.139725 | 12.974905 |
| B | 0 | 0.000000 | 0.000000 | 0.000000 |
| B | 2000 | 0.082729 | 0.012124 | 26.748310 |
| B | 4000 | 0.124762 | 0.031101 | 11.877080 |
| B | 20000 | 0.235557 | 0.100473 | 19.848512 |
该analytic latent KL与训练日志的approxKL含义不同。Critic预测漂移并不等于value error，因为没有把该固定deterministic轨迹回报冒充当前随机策略的真实价值标签。

## 证据支持与可能原因

已证实：在固定0°、没有课程变化、同起始网络和第一批相同数据的情况下，两组继续PPO更新后都丢失成功；本次清空双Adam状态没有保留得更久。因此，旧Adam状态并不是退化所必需的唯一条件。
已证实：模型权重/同状态控制会漂移，且训练成功样本非常少；source deterministic成功并不保证随机训练策略每回合成功。源latent标准差为[0.632981, 0.61103, 0.628379]，训练时仍有明显随机扰动。
可能原因：稀疏终局奖励与高方差advantage/value target使有限样本的更新方向难以稳定保持精细拦截几何；PPO clip约束本批概率比，而没有保证原固定场景的捕获结果。上面的首批统计和动作漂移与此相容，但本次单一ablation不足以证明唯一机制。
不能直接下结论：学习率一定过大、旧momentum一定有害、Critic一定是唯一根因，或reward一定必须重写。A/B同时改变两套Adam的一阶矩、二阶矩和计数，未分离Actor/Critic各自作用。
下一步可讨论只降低学习率的短期受控验证，例如保持较优的原Adam状态，仅降低Actor学习率、Critic学习率保持不变；或预先定义两者同比例缩小的另一独立对照。两种设计回答不同问题，应先选定单一对照。本轮没有执行任何降学习率实验。

## 曲线、轨迹和完整视频

- [retention_evaluation_comparison](../outputs/v10_策略保持诊断/retention_20261110_seed000/plots/retention_evaluation_comparison.png)
- [ppo_losses_entropy](../outputs/v10_策略保持诊断/retention_20261110_seed000/plots/ppo_losses_entropy.png)
- [ppo_kl_clip_learning_rate](../outputs/v10_策略保持诊断/retention_20261110_seed000/plots/ppo_kl_clip_learning_rate.png)
- [raw_advantage_return_diagnostics](../outputs/v10_策略保持诊断/retention_20261110_seed000/plots/raw_advantage_return_diagnostics.png)
- [source_parameter_drift](../outputs/v10_策略保持诊断/retention_20261110_seed000/plots/source_parameter_drift.png)
- [training_episode_metrics](../outputs/v10_策略保持诊断/retention_20261110_seed000/plots/training_episode_metrics.png)
- [same_state_source_policy_drift](../outputs/v10_策略保持诊断/retention_20261110_seed000/plots/same_state_source_policy_drift.png)
- A：[每2k完整评估表](../outputs/v10_策略保持诊断/retention_20261110_seed000/A/evaluations.csv)；[XY轨迹演化](../outputs/v10_策略保持诊断/retention_20261110_seed000/A/trajectory_evolution/trajectory_evolution_xy.png)；[演化指标CSV](../outputs/v10_策略保持诊断/retention_20261110_seed000/A/trajectory_evolution/trajectory_evolution_metrics.csv)；[PPO指标CSV](../outputs/v10_策略保持诊断/retention_20261110_seed000/A/updates.csv)。
  - A best：新增2000步，实际最后更新1536步，pending=464；[完整视频](../outputs/v10_策略保持诊断/retention_20261110_seed000/A/videos/best/best_full_episode.mp4)。
  - A final：新增20000步，实际最后更新20000步，pending=0；[完整视频](../outputs/v10_策略保持诊断/retention_20261110_seed000/A/videos/final/final_full_episode.mp4)。
- B：[每2k完整评估表](../outputs/v10_策略保持诊断/retention_20261110_seed000/B/evaluations.csv)；[XY轨迹演化](../outputs/v10_策略保持诊断/retention_20261110_seed000/B/trajectory_evolution/trajectory_evolution_xy.png)；[演化指标CSV](../outputs/v10_策略保持诊断/retention_20261110_seed000/B/trajectory_evolution/trajectory_evolution_metrics.csv)；[PPO指标CSV](../outputs/v10_策略保持诊断/retention_20261110_seed000/B/updates.csv)。
  - B best：新增0步，实际最后更新0步，pending=0；[完整视频](../outputs/v10_策略保持诊断/retention_20261110_seed000/B/videos/best/best_full_episode.mp4)。
  - B final：新增20000步，实际最后更新20000步，pending=0；[完整视频](../outputs/v10_策略保持诊断/retention_20261110_seed000/B/videos/final/final_full_episode.mp4)。

Best按固定场景捕获、短成功时间、低最近距离和低碰撞选择，包含导入的0新增步模型；没有覆盖原v08权重。四个视频仅重绘既有NPZ，没有重新选择新初态或追加模型评估。

## 测试、可复现性与限制

- 本次完整pytest：448 passed in 105.95s，[测试日志](../outputs/v10_策略保持诊断/_retention_run_checks_20261110/pytest.log)；训练前已通过。
- 36个checkpoint与读取CSV数值有限，无NaN/Inf；source权重SHA和10872个历史输出文件均保持不变。
- 单训练seed、固定单场景，2k检查仅定位区间，不能给出精确失效更新或泛化统计。尚未序列化中途pending rollout，分析checkpoint不等于完整恢复点。
- 仅新增独立retention入口、测试和YAML；既有索引只扩展诊断模型0新增步的来源校验。原环境、PPO算法、奖励、网络、curriculum实现与历史结果未改。

## 运行

```powershell
.\.venv\Scripts\python.exe -B -u -m scripts.train_policy_retention --config configs/mappo_policy_retention.yaml
```
