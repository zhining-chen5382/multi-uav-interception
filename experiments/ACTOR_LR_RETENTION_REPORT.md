# v11：固定 0° 的 Actor learning rate 策略保持对照

同一原始 v08 成功 checkpoint，保留 Actor、Critic、双 Adam 全部矩/计数及同一全局 RNG。只比较 Actor LR A=3e-4、B=1e-4、C=3e-5；Critic LR 固定1e-3。没有 scheduler、curriculum、奖励/观测/环境/动力学/网络改动。

## 来源与公平性

- 源 `outputs/v08_捕获奖励500/v08-r01_20261108_seed000/checkpoints/update_000180.pt`；SHA256 `d5d78f3045a2bb9d865028182db21a7128603e02d890cef5dfcabd0ed97609cd`；源预训练92160步单列。
- seed=0、device=cpu、线程=1；Git `c6fe7e7c64e62425c079960b3895615b6f274113`，dirty=True；实际开始北京时间2026-11-10 15:46:41（UTC+8，metadata）。
- 每组新增20k、40次PPO更新；共60k环境工作不代表某一模型连续训练60k。源237步未结束回合三组共同重置，开始同一固定初态的新回合。
- Actor/Critic 初始权重、原双 Adam.state 和原全局 RNG 逐位一致；Actor 参数组仅 lr 改为该组值，Critic 参数组及 lr=1e-3 完全保留。每个原Adam参数的step仍为5760，未重置优化器。
- B/C相对A的首批512步各13组训练数组逐位相同（含GAE/returns）；初始确定性轨迹也完全一致，并复现原v08成功NPZ。A的共同日志和11个评估checkpoint的网络/双Adam/RNG逐位复现上轮v10保留Adam对照。

| 组 / Run | Actor LR | Critic LR | 新增步数 | Updates | 完整训练回合 | 随机训练捕获 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A / v11-r01 | 0.0003 | 0.001 | 20000 | 40 | 71 | 1 |
| B / v11-r02 | 0.0001 | 0.001 | 20000 | 40 | 72 | 5 |
| C / v11-r03 | 3e-05 | 0.001 | 20000 | 40 | 72 | 3 |

## 实际配置

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
    actor_learning_rate: 0.0003
  - name: B
    reset_adam: false
    actor_learning_rate: 0.0001
  - name: C
    reset_adam: false
    actor_learning_rate: 3.0e-05
  evaluation_interval_steps: 2000
  evaluation_seed: 1000
```

父配置ppo.actor_learning_rate保留源基值用于共同设置校验，实际每臂在加载源Adam后覆盖Actor参数组lr；各臂training_config.yaml和checkpoint.config.ppo记录实际值，updates.csv逐次核验有效学习率不变。没有引入scheduler。

## 精确评估时点

0、2000、4000、…、20000每点执行一个完整固定0° deterministic回合，保存NPZ。2k点对应last_update=1536/pending464，4k点对应3584/pending416；暂停评估不切断或清空512步采样批。硬预算20k末32步先做更新再评估。
评估恢复RNG/train-eval模式，不动训练环境或buffer。未序列化pending buffer，途中checkpoint仅用于推理分析，不能直接视为完整续训点。首次失败只是相邻已检查策略之间的观察，不是精确失效更新；未观测更新中可能短暂失败或恢复。单场景布尔捕获也不是泛化率。

## 全部确定性结果

| 新增步数 | 最近更新步数 | A捕获 | A最近m | A捕获秒 | B捕获 | B最近m | B捕获秒 | C捕获 | C最近m | C捕获秒 |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 0 | 0 | 成功 | 3.525 | 17.3 | 成功 | 3.525 | 17.3 | 成功 | 3.525 | 17.3 |
| 2000 | 1536 | 成功 | 4.043 | 17.2 | 成功 | 4.932 | 17.4 | 失败 | 11.485 | — |
| 4000 | 3584 | 失败 | 40.858 | — | 失败 | 16.041 | — | 失败 | 35.132 | — |
| 6000 | 5632 | 失败 | 39.636 | — | 失败 | 9.692 | — | 失败 | 36.554 | — |
| 8000 | 7680 | 失败 | 33.402 | — | 失败 | 10.876 | — | 失败 | 26.327 | — |
| 10000 | 9728 | 失败 | 58.478 | — | 失败 | 49.609 | — | 失败 | 19.515 | — |
| 12000 | 11776 | 失败 | 62.790 | — | 失败 | 72.899 | — | 失败 | 18.832 | — |
| 14000 | 13824 | 失败 | 46.924 | — | 失败 | 48.978 | — | 失败 | 26.469 | — |
| 16000 | 15872 | 失败 | 10.989 | — | 失败 | 31.558 | — | 失败 | 45.049 | — |
| 18000 | 17920 | 失败 | 63.759 | — | 失败 | 10.765 | — | 失败 | 69.713 | — |
| 20000 | 20000 | 失败 | 68.396 | — | 失败 | 25.843 | — | 失败 | 93.963 | — |

## Retention 与恢复

| 组 | 最后确认的初始成功前缀新增步 | 首个失败检查 | 相邻检查区间 | 对应更新权重区间 | 后来恢复的检查点 | 初始后成功检查数 | Final |
| --- | ---: | --- | --- | --- | --- | ---: | --- |
| A | 2000 | 4000 | [2000, 4000] | [1536, 3584] | [] | 1/10 | 失败 |
| B | 2000 | 4000 | [2000, 4000] | [1536, 3584] | [] | 1/10 | 失败 |
| C | 0 | 2000 | [0, 2000] | [0, 1536] | [] | 0/10 | 失败 |
三组初始成功前缀没有随Actor LR降低形成严格单调改善；应同时看首次失败、恢复和全部中间点，不能只报告最好一帧或final。
本次初始成功前缀最长的是A/B，最后确认于新增2000步；三组20k final的最近距离分别为A 68.396m、B 25.843m、C 93.963m。

## 轨迹/策略漂移

只在原成功轨迹的同一173步观测上读取已存权重的动作，未新增环境回合。原源动作逐位重建，避免用不同访问状态上的输出差冒充策略漂移。

动作漂移是归一化动作tanh(mean)的每机3维L2差，再对173个观测时刻和三机取均值，无物理加速度单位。它反映相同输入下Actor的改变；真实闭环轨迹另见各组XY演化图。

| 组 | 新增步 | 同状态动作ΔL2均值 | Source→Current latent KL | Critic源状态预测ΔRMS |
| --- | ---: | ---: | ---: | ---: |
| A | 0 | 0.000000 | 0.000000 | 0.000000 |
| A | 2000 | 0.067852 | 0.008982 | 15.791180 |
| A | 4000 | 0.145555 | 0.037574 | 13.125683 |
| A | 10000 | 0.244835 | 0.091675 | 15.751625 |
| A | 20000 | 0.297604 | 0.139725 | 12.974905 |
| B | 0 | 0.000000 | 0.000000 | 0.000000 |
| B | 2000 | 0.046474 | 0.004367 | 15.733368 |
| B | 4000 | 0.091400 | 0.015510 | 15.650129 |
| B | 10000 | 0.214341 | 0.070434 | 21.314444 |
| B | 20000 | 0.102117 | 0.019395 | 18.498459 |
| C | 0 | 0.000000 | 0.000000 | 0.000000 |
| C | 2000 | 0.055383 | 0.004929 | 16.842991 |
| C | 4000 | 0.110436 | 0.017354 | 13.474246 |
| C | 10000 | 0.132408 | 0.030780 | 23.435111 |
| C | 20000 | 0.275515 | 0.107387 | 28.231927 |

| 组 | 全部采样点动作漂移均值 | 20k动作漂移 | 全部采样点latent KL均值 | 20k latent KL |
| --- | ---: | ---: | ---: | ---: |
| A | 0.201117 | 0.297604 | 0.081523 | 0.139725 |
| B | 0.129614 | 0.102117 | 0.034541 | 0.019395 |
| C | 0.142219 | 0.275515 | 0.039332 | 0.107387 |
20k动作漂移没有呈现随Actor LR降低而单调减少的端点排序；本轮不能笼统宣称低LR总是漂移更慢。
该解析latent KL是固定同观测上的source||current分布距离，与训练批approxKL不同；Critic预测变化不是value target error。
全部11个观测点（含0步）的同状态动作漂移均值最小为B组。这个时间采样均值及20k端点不能证明每个未检查更新都漂移更慢，更不能替代捕获结果。

## PPO 与训练趋势

首/末各20%更新样本均值用于趋势归纳，窗口样本数在JSON中；全部曲线使用原始点，不做平滑。

| 指标 | A首→末 | B首→末 | C首→末 |
| --- | ---: | ---: | ---: |
| actor_loss | -0.00840381 → -0.00879689 | -0.00597369 → -0.00732993 | -0.00273256 → -0.00217242 |
| value_loss | 280.89 → 13.6283 | 793.006 → 299.513 | 285.85 → 35.4587 |
| entropy | 1.60152 → 1.60843 | 1.6244 → 1.59419 | 1.61937 → 1.56022 |
| approx_kl | 0.00665428 → 0.00560635 | 0.0028607 → 0.0027145 | 0.000764359 → 0.000459939 |
| clip_fraction | 0.0855103 → 0.0685832 | 0.0282389 → 0.0300293 | 0.00360107 → 0.00113932 |
| actor_grad_norm | 2.584 → 3.79299 | 1.9769 → 2.60945 | 2.47072 → 3.07502 |
| critic_grad_norm | 384.922 → 138.989 | 667.203 → 288.147 | 385.522 → 294.195 |
| actor_std_mean | 0.624625 → 0.635209 | 0.623181 → 0.621424 | 0.624084 → 0.624947 |
| advantage_std | 15.0608 → 4.78494 | 31.4898 → 13.9621 | 15.5888 → 9.12096 |
| return_std | 30.3234 → 16.5763 | 46.2222 → 25.5203 | 31.6672 → 19.5493 |
| actor_parameter_delta_l2 | 0.426766 → 1.03964 | 0.212731 → 0.463563 | 0.0877112 → 0.21855 |
| critic_parameter_delta_l2 | 3.05966 → 6.15298 | 3.73501 → 7.55604 | 2.98106 → 6.40847 |

有效Actor LR三组分别恒定为3e-4、1e-4、3e-5；Critic均恒定1e-3。approxKL/clip fraction是本批旧策略相对当前更新的minibatch/epoch平均，不能当相对源策略的累计漂移。Entropy为变换Gaussian的Monte Carlo估计；梯度范数为裁剪前记录，max_grad_norm=0.5仍实际生效。

首个512步批相同，其第一次PPO更新后的Actor参数L2变化为A 0.225264、B 0.108973、C 0.049875；Critic loss三组相同为2104.893752。这证明Actor LR处理已实际生效，且初始Critic/训练批差异不能解释组间首批差异。随后on-policy轨迹和GAE目标会分叉，不能把后期差异只归因于优化器的瞬时步长。

## 结论与边界

本实验仅改变Actor优化器lr，三组原Adam矩和step都保留，因此比同时重置双优化器更直接地检验Actor步长对retention的影响。后续on-policy数据会因策略不同而自然分叉，这属于该处理的结果，并非初始条件不公平。
较小的Actor LR可能让策略在有限20k内更靠近源策略，但更小漂移并不等于已学到更优拦截，也可能只是更新更慢。成功保持、最终捕获、恢复和轨迹偏移应分别报告。
单seed、固定单场景、2k观测分辨率和20k短预算限制了结论；没有进行curriculum或新的学习率配置筛选，也没有依据结果修改奖励、网络、动力学或scheduler。

## 图表和完整视频

- [actor_lr_retention_comparison](../outputs/v11_ActorLR策略保持/actor_lr_20261110_seed000/plots/actor_lr_retention_comparison.png)
- [ppo_losses_entropy](../outputs/v11_ActorLR策略保持/actor_lr_20261110_seed000/plots/ppo_losses_entropy.png)
- [ppo_kl_clip_learning_rate](../outputs/v11_ActorLR策略保持/actor_lr_20261110_seed000/plots/ppo_kl_clip_learning_rate.png)
- [raw_advantage_return_diagnostics](../outputs/v11_ActorLR策略保持/actor_lr_20261110_seed000/plots/raw_advantage_return_diagnostics.png)
- [source_parameter_drift](../outputs/v11_ActorLR策略保持/actor_lr_20261110_seed000/plots/source_parameter_drift.png)
- [training_episode_metrics](../outputs/v11_ActorLR策略保持/actor_lr_20261110_seed000/plots/training_episode_metrics.png)
- [same_state_source_policy_drift](../outputs/v11_ActorLR策略保持/actor_lr_20261110_seed000/plots/same_state_source_policy_drift.png)
- A：[每2k评估CSV](../outputs/v11_ActorLR策略保持/actor_lr_20261110_seed000/A/evaluations.csv)；[PPO指标CSV](../outputs/v11_ActorLR策略保持/actor_lr_20261110_seed000/A/updates.csv)；[XY轨迹演化](../outputs/v11_ActorLR策略保持/actor_lr_20261110_seed000/A/trajectory_evolution/trajectory_evolution_xy.png)；[逐checkpoint演化CSV](../outputs/v11_ActorLR策略保持/actor_lr_20261110_seed000/A/trajectory_evolution/trajectory_evolution_metrics.csv)。
  - A best：新增2000步，last_update=1536，pending=464；[完整视频](../outputs/v11_ActorLR策略保持/actor_lr_20261110_seed000/A/videos/best/best_full_episode.mp4)。
  - A final：新增20000步，last_update=20000，pending=0；[完整视频](../outputs/v11_ActorLR策略保持/actor_lr_20261110_seed000/A/videos/final/final_full_episode.mp4)。
- B：[每2k评估CSV](../outputs/v11_ActorLR策略保持/actor_lr_20261110_seed000/B/evaluations.csv)；[PPO指标CSV](../outputs/v11_ActorLR策略保持/actor_lr_20261110_seed000/B/updates.csv)；[XY轨迹演化](../outputs/v11_ActorLR策略保持/actor_lr_20261110_seed000/B/trajectory_evolution/trajectory_evolution_xy.png)；[逐checkpoint演化CSV](../outputs/v11_ActorLR策略保持/actor_lr_20261110_seed000/B/trajectory_evolution/trajectory_evolution_metrics.csv)。
  - B best：新增0步，last_update=0，pending=0；[完整视频](../outputs/v11_ActorLR策略保持/actor_lr_20261110_seed000/B/videos/best/best_full_episode.mp4)。
  - B final：新增20000步，last_update=20000，pending=0；[完整视频](../outputs/v11_ActorLR策略保持/actor_lr_20261110_seed000/B/videos/final/final_full_episode.mp4)。
- C：[每2k评估CSV](../outputs/v11_ActorLR策略保持/actor_lr_20261110_seed000/C/evaluations.csv)；[PPO指标CSV](../outputs/v11_ActorLR策略保持/actor_lr_20261110_seed000/C/updates.csv)；[XY轨迹演化](../outputs/v11_ActorLR策略保持/actor_lr_20261110_seed000/C/trajectory_evolution/trajectory_evolution_xy.png)；[逐checkpoint演化CSV](../outputs/v11_ActorLR策略保持/actor_lr_20261110_seed000/C/trajectory_evolution/trajectory_evolution_metrics.csv)。
  - C best：新增0步，last_update=0，pending=0；[完整视频](../outputs/v11_ActorLR策略保持/actor_lr_20261110_seed000/C/videos/best/best_full_episode.mp4)。
  - C final：新增20000步，last_update=20000，pending=0；[完整视频](../outputs/v11_ActorLR策略保持/actor_lr_20261110_seed000/C/videos/final/final_full_episode.mp4)。

Best仅在真实固定场景评估的checkpoint中依次按捕获、较低敌方攻击成功、较短成功捕获时间、低最近距离和低碰撞选择，可能仍为导入时0新增步；六个视频均重绘已存NPZ，没有追加新推理。

## 测试与保护

- 开训前实际完整pytest：466 passed in 79.63s，[测试日志](../outputs/v11_ActorLR策略保持/_run_checks_20261110/pytest.log)。
- 54个checkpoint及读取CSV数值有限，无NaN/Inf；14394个历史输出SHA保持不变。
- 只新增Actor-LR入口/配置/测试并为共享retention runner添加可选Actor LR覆盖；v10无该字段的原Adam实验已回归。原环境、PPO算法、Critic LR、源权重及全部历史实验保持。

## 运行

```powershell
.\.venv\Scripts\python.exe -B -u -m scripts.train_actor_lr_retention --config configs/mappo_actor_lr_retention.yaml
```
