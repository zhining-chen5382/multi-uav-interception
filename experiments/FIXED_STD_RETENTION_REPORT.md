# v13 固定 Action Std 的策略保持对照

原始v08 92,160 steps成功模型，固定0°，Actor LR=3e-4、Critic LR=1e-3。A=1.0×、B=0.15×、C=0.10×，三组各新增20k，log_std设置后固定，不允许PPO更新。

## 执行记录

首次执行的进程在A完成20k、B日志记录到2560步后消失，现存日志没有Python异常，具体原因未确定。原目录完整保留。随后从同一源文件、配置和RNG在独立recovery01目录重新运行三组，各完成20k。
完整重跑退出码为0，运行耗时224.09秒，包含各组训练后的轨迹演化分析。
独立核验确认重跑A的40次更新、保存模型和双Adam/RNG，以及B的前5次更新逐位复现。报告采用这次完整执行的三组数据。
[原中断日志](../outputs/v13_低Std策略保持/_checks_20261116/training_console.log) · [完整执行状态](../outputs/v13_低Std策略保持/_checks_20261116/recovery_training_status.json) · [独立复现与冻结核验](../outputs/v13_低Std策略保持/fixed_std_20261116_seed000_recovery01/verification.json)

## 来源、干预和公平性

- 源`outputs/v08_捕获奖励500/v08-r01_20261108_seed000/checkpoints/update_000180.pt`；SHA256 `d5d78f3045a2bb9d865028182db21a7128603e02d890cef5dfcabd0ed97609cd`。
- Git `c6fe7e7c64e62425c079960b3895615b6f274113`，dirty=True；seed=0，device=cpu，torch线程=1。开始时间UTC `2026-11-16T03:56:50.851507+00:00`。
- 三组均重新读取同一文件，继承均值网络、Critic、双Adam全部参数组/矩/step及同一源RNG；只预设log_std并冻结它。原237步未结束回合共同重置，均从同一固定初态新回合开始。
- 原始log_std为`[-0.4573156237602234, -0.49260884523391724, -0.4646120071411133]`。处理为`log_std_group = log_std_source + ln(multiplier)`，全部落在原clamp范围内。tanh前Gaussian的采样、旧/新log probability与entropy共用这一冻结参数，未只在环境动作上额外乘倍率。
- 保留log_std在原Adam参数列表和其全部原状态；因requires_grad=False且grad=None，Adam跳过该参数，矩和step不变化。均值网络及Critic继续正常更新。
- 本轮新增固定1×的A。历史v11 A采用可学习log_std，保存在历史参照表中。
- 三组初始deterministic轨迹逐位一致。std倍率改变随机动作、访问状态、回报及GAE目标。

| 组 | 倍率 | 初始并固定的log_std | 实际latent std | 更新数 | 完整训练回合 | 随机训练捕获 |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| A | 1× | [-0.4573156, -0.4926088, -0.464612] | [0.6329805, 0.6110302, 0.6283789] | 40 | 71 | 1 |
| B | 0.15× | [-2.3544357, -2.3897288, -2.361732] | [0.0949471, 0.0916545, 0.0942568] | 40 | 72 | 5 |
| C | 0.1× | [-2.7599008, -2.7951939, -2.7671971] | [0.063298, 0.061103, 0.0628379] | 40 | 74 | 8 |

## 实际共同配置

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
  historical_trainable_std_reference: outputs/v11_ActorLR策略保持/actor_lr_20261110_seed000/A
  groups:
  - name: A
    reset_adam: false
    frozen_std_multiplier: 1.0
  - name: B
    reset_adam: false
    frozen_std_multiplier: 0.15
  - name: C
    reset_adam: false
    frozen_std_multiplier: 0.1
  evaluation_interval_steps: 2000
  evaluation_seed: 1000
```

共同reward、环境、观测、动力学、网络和PPO配置均与源快照核对。每组新增20k环境步，另记录源预训练92160步。三组累计新增60k环境步。

## 每2k的全部确定性结果

| 新增步数 | 最近PPO更新步 | A捕获 | A最近m | A捕获秒 | B捕获 | B最近m | B捕获秒 | C捕获 | C最近m | C捕获秒 |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 0 | 0 | 成功 | 3.525 | 17.3 | 成功 | 3.525 | 17.3 | 成功 | 3.525 | 17.3 |
| 2000 | 1536 | 成功 | 3.476 | 17.2 | 失败 | 8.320 | — | 失败 | 11.927 | — |
| 4000 | 3584 | 失败 | 41.198 | — | 失败 | 13.885 | — | 失败 | 13.925 | — |
| 6000 | 5632 | 失败 | 33.995 | — | 失败 | 19.023 | — | 成功 | 4.329 | 17.2 |
| 8000 | 7680 | 失败 | 20.855 | — | 失败 | 25.054 | — | 失败 | 13.968 | — |
| 10000 | 9728 | 失败 | 18.884 | — | 失败 | 32.085 | — | 失败 | 17.636 | — |
| 12000 | 11776 | 失败 | 24.215 | — | 失败 | 47.155 | — | 失败 | 15.775 | — |
| 14000 | 13824 | 失败 | 6.012 | — | 失败 | 45.290 | — | 失败 | 20.589 | — |
| 16000 | 15872 | 失败 | 47.846 | — | 失败 | 54.848 | — | 失败 | 22.487 | — |
| 18000 | 17920 | 失败 | 27.515 | — | 失败 | 63.996 | — | 失败 | 41.543 | — |
| 20000 | 20000 | 失败 | 62.932 | — | 失败 | 77.036 | — | 失败 | 36.196 | — |

0/2k/4k/.../20k各运行一个固定0°完整deterministic回合。精确评估暂停采样并保留512步rollout。2k点的最近更新在1536，尚有464步pending。4k点在3584，pending416。20k末尾32步先更新再评估。
首次观察失败落在相邻检查点的区间内。实验以2k步为观测分辨率，中间策略可能短暂失败或恢复。中途分析快照记录pending数量，完整续训仍需要保存相应on-policy buffer。

## Retention与恢复

| 组 | 最后确认的初始成功前缀 | 首个失败检查 | 对应检查区间 | 后续成功恢复检查 | 训练后成功检查数 | Final |
| --- | ---: | --- | --- | --- | ---: | --- |
| A | 2000 | 4000 | [2000, 4000] | [] | 1/10 | 失败 |
| B | 0 | 2000 | [0, 2000] | [] | 0/10 | 失败 |
| C | 0 | 2000 | [0, 2000] | [6000] | 1/10 | 失败 |

低噪声两组与本轮固定1× A比较。
- B（0.15×）：初始成功前缀更短；仍须一起看后续恢复和final。
- C（0.1×）：初始成功前缀更短；仍须一起看后续恢复和final。

历史可学习std A最后确认初始成功前缀2000，首个失败检查4000，后续恢复[]，final最近68.396m。该组与本轮固定1× A之间还包含冻结std的处理差异。

## PPO与数值趋势

下表用40次更新的前8次均值→后8次均值概括。图和CSV保留全部原始更新。

| 指标 | A 1× | B 0.15× | C 0.10× |
| --- | --- | --- | --- |
| actor_loss | -0.0082281 → -0.00996555 | 0.00558745 → -0.000969519 | 0.0122183 → 0.012494 |
| value_loss | 280.328 → 9.58791 | 1353.71 → 0.568588 | 1663.23 → 91.0292 |
| entropy | 1.60152 → 1.60741 | -3.17809 → -3.2162 | -4.38134 → -4.32791 |
| approx_kl | 0.00656937 → 0.00571151 | 0.0370943 → 0.0308071 | 0.0431976 → 0.036829 |
| clip_fraction | 0.0845744 → 0.070048 | 0.340881 → 0.299255 | 0.373515 → 0.342082 |
| actor_grad_norm | 2.59493 → 3.54697 | 15.8248 → 14.2415 | 25.9012 → 27.3877 |
| critic_grad_norm | 396.096 → 124.565 | 686.504 → 35.6918 | 737.484 → 171.955 |
| advantage_mean | 0.44135 → -0.402295 | 0.0881505 → 0.427979 | 0.469078 → -1.63983 |
| advantage_std | 14.9708 → 3.90828 | 41.2033 → 1.23174 | 45.8082 → 11.6546 |
| return_mean | -42.7527 → -45.7647 | -21.4851 → -50.3236 | -11.354 → -67.2756 |
| return_std | 30.896 → 16.8867 | 54.7713 → 16.7805 | 61.3387 → 19.7527 |
| value_mean | -43.194 → -45.3624 | -21.5732 → -50.7515 | -11.8231 → -65.6358 |
| actor_std_mean | 0.62413 → 0.62413 | 0.0936195 → 0.0936195 | 0.062413 → 0.062413 |

Entropy采用tanh变换后连续动作分布的Monte Carlo微分熵。连续微分熵允许为负，训练数值状态由有限值检查验证。approxKL和clip fraction按每批更新相对其旧采样策略计算。
在固定协方差下，同一均值位移对应的Gaussian KL随1/std²缩放。减小std可能同时改变梯度范数和PPO裁剪比例，需结合本轮原始曲线判断更新稳定性。

## 均值策略与轨迹漂移

| 组 | 20k均值网络参数ΔL2（排除log_std） |
| --- | ---: |
| A | 1.093559 |
| B | 0.688543 |
| C | 0.445425 |

updates.csv的actor_parameter_delta_l2同时包含预设log_std位移。表中均值网络漂移采用fixed_std_updates.csv的actor_mean_parameter_delta_l2，它仅统计mean网络参数。各组XY演化图展示完整闭环轨迹。

另在同一源成功轨迹的固定观测上只读取已保存网络，不新增环境回合；以下均值动作漂移不包含采样噪声。

| 组 | 新增步数 | 同状态均值动作ΔL2 | 源状态Critic预测ΔRMS |
| --- | ---: | ---: | ---: |
| A | 0 | 0.000000 | 0.000000 |
| A | 2000 | 0.066496 | 15.199846 |
| A | 4000 | 0.159092 | 13.623523 |
| A | 10000 | 0.204532 | 14.292420 |
| A | 20000 | 0.209248 | 12.210665 |
| B | 0 | 0.000000 | 0.000000 |
| B | 2000 | 0.070221 | 24.453174 |
| B | 4000 | 0.085095 | 16.886083 |
| B | 10000 | 0.118467 | 15.000595 |
| B | 20000 | 0.218399 | 13.692554 |
| C | 0 | 0.000000 | 0.000000 |
| C | 2000 | 0.045384 | 56.964977 |
| C | 4000 | 0.055541 | 20.540409 |
| C | 10000 | 0.089312 | 42.496661 |
| C | 20000 | 0.130883 | 44.746841 |
同状态KL采用各组预设std之后的initial分布为基准，三组在新增0步时均为0。采样std越小，相同mean位移对应的KL越大。同状态mean动作ΔL2单独度量控制输出变化。

## 结论与限制

固定1×的A在新增2k仍捕获、4k首次观察失败。B和C在2k首次观察失败。C在6k检查短暂恢复捕获，8k再次失败。20k最终三组均失守。以本次seed和2k观测分辨率衡量，固定1×的A保持初始成功前缀最久，降低std两组的初始成功前缀更短。
C在20k的最近距离36.196m，低于A的62.932m和B的77.036m。同状态确定性动作平均L2漂移为A 0.209、B 0.218、C 0.131，C的控制漂移较小，但20k时仍超过捕获距离要求。
首个PPO更新的approx KL分别为A 0.00820、B 0.03700、C 0.06175，clip fraction分别约11.2%、36.4%、45.0%。本轮小std组在相同学习率下出现更大的分布变化和裁剪比例。固定协方差的Gaussian KL包含均值位移平方除以std平方，小std会放大同一均值位移对应的KL。
首个更新的value loss约为A 2104.9、B 4612.5、C 4603.2。改变采样std会改变回报、访问状态和GAE目标，继承的Critic和Adam需要在这一新分布中继续适应。现有数据支持这些机制作为可能解释，尚不能把退化单独归因于Critic、Adam或某一项损失。
各组log_std和该参数的Adam状态均逐位固定，均值网络和Critic持续更新。所有保存权重和更新指标通过有限数检查，低std组的负微分熵符合连续分布定义。
C的20k确定性回合有72个仿真步触发原有我方安全距离惩罚，累计扣360分，最终回报约-542.70。该回合三机轨迹接近，正面错过敌机后继续远离基地。距离和回报应结合终止结果、碰撞项及完整轨迹评估。
实验覆盖一个source、一个训练seed、固定0°单场景、20k新增预算和2k评估分辨率。结论限定在这一受控设置内。

## 图表、原始数据和完整视频

- [fixed_std_retention_comparison](../outputs/v13_低Std策略保持/fixed_std_20261116_seed000_recovery01/plots/fixed_std_retention_comparison.png)
- [ppo_losses_entropy](../outputs/v13_低Std策略保持/fixed_std_20261116_seed000_recovery01/plots/ppo_losses_entropy.png)
- [ppo_kl_clip_learning_rate](../outputs/v13_低Std策略保持/fixed_std_20261116_seed000_recovery01/plots/ppo_kl_clip_learning_rate.png)
- [raw_advantage_return_diagnostics](../outputs/v13_低Std策略保持/fixed_std_20261116_seed000_recovery01/plots/raw_advantage_return_diagnostics.png)
- [mean_parameter_drift_and_fixed_std](../outputs/v13_低Std策略保持/fixed_std_20261116_seed000_recovery01/plots/mean_parameter_drift_and_fixed_std.png)
- [training_episode_metrics](../outputs/v13_低Std策略保持/fixed_std_20261116_seed000_recovery01/plots/training_episode_metrics.png)
- [same_state_mean_action_drift](../outputs/v13_低Std策略保持/fixed_std_20261116_seed000_recovery01/plots/same_state_mean_action_drift.png)
- A：[逐2k评估](../outputs/v13_低Std策略保持/fixed_std_20261116_seed000_recovery01/A/evaluations.csv)；[完整PPO指标](../outputs/v13_低Std策略保持/fixed_std_20261116_seed000_recovery01/A/updates.csv)；[std冻结证据](../outputs/v13_低Std策略保持/fixed_std_20261116_seed000_recovery01/A/fixed_std_updates.csv)；[XY演化图](../outputs/v13_低Std策略保持/fixed_std_20261116_seed000_recovery01/A/trajectory_evolution/trajectory_evolution_xy.png)；[演化CSV](../outputs/v13_低Std策略保持/fixed_std_20261116_seed000_recovery01/A/trajectory_evolution/trajectory_evolution_metrics.csv)。
  - A best：新增2000步，last_update=1536，pending=464；[完整回合视频](../outputs/v13_低Std策略保持/fixed_std_20261116_seed000_recovery01/A/videos/best/best_full_episode.mp4)。
  - A final：新增20000步，last_update=20000，pending=0；[完整回合视频](../outputs/v13_低Std策略保持/fixed_std_20261116_seed000_recovery01/A/videos/final/final_full_episode.mp4)。
- B：[逐2k评估](../outputs/v13_低Std策略保持/fixed_std_20261116_seed000_recovery01/B/evaluations.csv)；[完整PPO指标](../outputs/v13_低Std策略保持/fixed_std_20261116_seed000_recovery01/B/updates.csv)；[std冻结证据](../outputs/v13_低Std策略保持/fixed_std_20261116_seed000_recovery01/B/fixed_std_updates.csv)；[XY演化图](../outputs/v13_低Std策略保持/fixed_std_20261116_seed000_recovery01/B/trajectory_evolution/trajectory_evolution_xy.png)；[演化CSV](../outputs/v13_低Std策略保持/fixed_std_20261116_seed000_recovery01/B/trajectory_evolution/trajectory_evolution_metrics.csv)。
  - B best：新增0步，last_update=0，pending=0；[完整回合视频](../outputs/v13_低Std策略保持/fixed_std_20261116_seed000_recovery01/B/videos/best/best_full_episode.mp4)。
  - B final：新增20000步，last_update=20000，pending=0；[完整回合视频](../outputs/v13_低Std策略保持/fixed_std_20261116_seed000_recovery01/B/videos/final/final_full_episode.mp4)。
- C：[逐2k评估](../outputs/v13_低Std策略保持/fixed_std_20261116_seed000_recovery01/C/evaluations.csv)；[完整PPO指标](../outputs/v13_低Std策略保持/fixed_std_20261116_seed000_recovery01/C/updates.csv)；[std冻结证据](../outputs/v13_低Std策略保持/fixed_std_20261116_seed000_recovery01/C/fixed_std_updates.csv)；[XY演化图](../outputs/v13_低Std策略保持/fixed_std_20261116_seed000_recovery01/C/trajectory_evolution/trajectory_evolution_xy.png)；[演化CSV](../outputs/v13_低Std策略保持/fixed_std_20261116_seed000_recovery01/C/trajectory_evolution/trajectory_evolution_metrics.csv)。
  - C best：新增6000步，last_update=5632，pending=368；[完整回合视频](../outputs/v13_低Std策略保持/fixed_std_20261116_seed000_recovery01/C/videos/best/best_full_episode.mp4)。
  - C final：新增20000步，last_update=20000，pending=0；[完整回合视频](../outputs/v13_低Std策略保持/fixed_std_20261116_seed000_recovery01/C/videos/final/final_full_episode.mp4)。

Best按真实deterministic检查的捕获/低攻击成功/短成功捕获时间/低最近距离/低碰撞依次选取，包含导入时0新增步；可能仍是源均值策略。视频重绘已存初始至终止NPZ，不额外运行环境；不能用best单例替代全部评估轨迹。

## 工程核验与运行

本轮实现涉及以下文件。

- `scripts/train_fixed_std_retention.py` 提供三组固定std的独立训练入口。
- `configs/mappo_fixed_std_retention.yaml` 记录倍率、预算、评估间隔及共同超参数。
- `scripts/train_policy_retention.py` 增加可选的log_std冻结与逐次核验，复用原采样、PPO和评估循环。
- `tests/test_fixed_std_retention.py` 覆盖真实PPO更新、原Adam继承、冻结参数及保存与重载。
- README、BASELINE_SPEC和实验索引同步记录v13协议与实际结果。

- 开训前实际完整pytest：535 passed in 209.92s，[日志](../outputs/v13_低Std策略保持/_checks_20261116/full_pytest.log)。
- 遍历54个保存checkpoint：全部有限，log_std及其Adam矩/step逐位不变；初始其余网络、完整双Adam和RNG逐位继承。均值网络及Critic实际发生更新。
- 三组40个附加update日志均证明requires_grad=False/grad=None/std与Adam不变。检查点保存参数数值和训练元数据；普通from_checkpoint只保证推理数值重建，冻结requires_grad状态须由本轮训练入口明确设置，不能把分析快照宣称为完整续训快照。
- 改动限定为独立固定std入口/配置/测试及共享retention runner的可选处理；原环境、网络架构、PPO核心、奖励和旧结果保持。

```powershell
.\.venv\Scripts\python.exe -B -u -m scripts.train_fixed_std_retention --config configs/mappo_fixed_std_retention.yaml --output-dir outputs/v13_低Std策略保持/fixed_std_20261116_seed000_recovery01
```
