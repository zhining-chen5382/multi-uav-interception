# v12：冻结 v08 成功策略的 Action Std 敏感性测试

本轮只评估：冻结原始 v08 92,160 steps 成功模型，在固定 0° 场景比较采样噪声倍率。没有训练，没有生成新的模型，没有修改 Actor/Critic、双 Adam、学习率、entropy 系数、reward、观测、动作映射或动力学。

## 来源与实际协议

- 源 checkpoint：`outputs/v08_捕获奖励500/v08-r01_20261108_seed000/checkpoints/update_000180.pt`。
- SHA256：`d5d78f3045a2bb9d865028182db21a7128603e02d890cef5dfcabd0ed97609cd`。
- Git：`c6fe7e7c64e62425c079960b3895615b6f274113`；dirty=True，以本次源快照/哈希为补充；未执行 commit/merge/push。
- 实际 UTC 开始：`2026-11-10T14:23:57.633705+00:00`；结束：`2026-11-10T14:25:12.816679+00:00`。
- 正式500回合推理耗时：75.184秒，不包含测试、制图与视频编码。
- 输出：`outputs/v12_ActionStd敏感性/std_sensitivity_20261110_seed1000`。
- 5 档倍率 × 每档 100 个完整回合 = 500 回合；各档 seed 均为 1000–1099。只改变 pre-tanh Normal 的采样 std 倍率。
- 各倍率复用相同 seed；正倍率之间共用标准高斯噪声前缀（paired common random numbers），每个回合使用独立 CPU torch.Generator，0×不抽取噪声；不消耗策略全局 Torch RNG。
- 敌机/三架我方机位置及速度、基地等全部使用 checkpoint 环境快照，初始状态逐回合核对完全相同。敌机保持 Direct Attacker。
- 0× 是同一 deterministic 轨迹重复 100 次；这些重复不是 100 个独立泛化场景。正倍率的每回合噪声 seed 不同，衡量同场景下对动作噪声的稳健程度。

## Action Std 的准确含义

```text
sigma = exp(clamp(actor.log_std, actor.min_log_std, actor.max_log_std))
epsilon ~ Normal(0, I)
raw_action = mean(observation) + multiplier * sigma * epsilon
normalized_action = tanh(raw_action)
```

0× 直接使用 `tanh(mean)`，不构造零方差 Normal，也不修改保存的 log_std。std 是共享 Actor 的三轴、与状态无关的 latent Normal 标准差；不是最终归一化动作的标准差，更不是物理加速度标准差。tanh、加速度/jerk/速度限制会继续改变实际运动响应。

| 量 | X | Y | Z |
| --- | --- | --- | --- |
| 原始 log_std | -0.4573156 | -0.4926088 | -0.4646120 |
| clamp 后 log_std | -0.4573156 | -0.4926088 | -0.4646120 |
| exp 后原始 std | 0.6329805 | 0.6110302 | 0.6283789 |

| 倍率 | 实际 latent std X | Y | Z |
| --- | --- | --- | --- |
| 0× | 0.0000000 | 0.0000000 | 0.0000000 |
| 0.25× | 0.1582451 | 0.1527576 | 0.1570947 |
| 0.5× | 0.3164903 | 0.3055151 | 0.3141894 |
| 1× | 0.6329805 | 0.6110302 | 0.6283789 |
| 1.5× | 0.9494708 | 0.9165453 | 0.9425683 |

## 全部 500 回合的实际结果

| 倍率 | 捕获数/100 | 攻击成功率 | 超时率 | 平均最小距离m | 中位最小距离m | 平均回报 | 成功平均拦截时间s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0× | 100/100 | 0.0% | 0.0% | 3.525 | 3.525 | 444.566 | 17.300 |
| 0.25× | 55/100 | 45.0% | 0.0% | 5.331 | 4.862 | 66.165 | 17.295 |
| 0.5× | 31/100 | 69.0% | 0.0% | 8.062 | 7.011 | -67.012 | 17.397 |
| 1× | 2/100 | 98.0% | 0.0% | 14.631 | 13.617 | -226.496 | 18.750 |
| 1.5× | 1/100 | 99.0% | 0.0% | 19.800 | 18.300 | -220.003 | 18.200 |

平均回报包含成功与失败回合，团队奖励每步只累计一次；碰撞与时间/距离成本仍按原环境执行。

正倍率捕获率 Wilson 95% 区间分别为：0.25× 45.24%–64.39%，0.5× 22.78%–40.63%，1× 0.55%–7.00%，1.5× 0.18%–5.45%。0×是相同轨迹的重复，不将形式二项区间当作泛化证据，图中不画区间。

### 最小敌我距离分布

| 倍率 | Min | P10 | P25 | Median | P75 | P90 | Max | Std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0× | 3.525 | 3.525 | 3.525 | 3.525 | 3.525 | 3.525 | 3.525 | 0.000 |
| 0.25× | 1.996 | 3.127 | 3.884 | 4.862 | 6.563 | 7.785 | 11.728 | 2.035 |
| 0.5× | 2.147 | 3.272 | 4.633 | 7.011 | 10.433 | 13.868 | 20.978 | 4.391 |
| 1× | 4.442 | 6.774 | 9.154 | 13.617 | 18.685 | 24.035 | 37.777 | 6.931 |
| 1.5× | 4.957 | 8.619 | 13.418 | 18.300 | 24.105 | 32.123 | 47.998 | 9.452 |

### Episode Return 分布

| 倍率 | Min | P10 | P25 | Median | P75 | P90 | Max | Std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0× | 444.566 | 444.566 | 444.566 | 444.566 | 444.566 | 444.566 | 444.566 | 0.000 |
| 0.25× | -717.013 | -331.393 | -177.408 | 92.106 | 444.150 | 444.666 | 444.976 | 326.413 |
| 0.5× | -490.397 | -342.882 | -274.919 | -176.597 | 215.435 | 443.983 | 445.417 | 289.698 |
| 1× | -475.688 | -327.785 | -290.640 | -218.145 | -175.893 | -173.625 | 443.892 | 119.630 |
| 1.5× | -378.044 | -307.650 | -270.374 | -201.443 | -175.373 | -172.916 | 376.823 | 82.326 |

分位数按每档真实100个回合计算，无平滑。全部逐回合距离/回报、seed 和 NPZ 路径保留在 CSV；NPZ 包含完整初始至终止轨迹、实际动作及 latent mean/std/epsilon/raw_action。

### 相邻倍率的配对捕获变化

| 倍率变化 | 成功→失败 | 失败→成功 | 均成功 | 均失败 |
| --- | --- | --- | --- | --- |
| 0×→0.25× | 45 | 0 | 55 | 0 |
| 0.25×→0.5× | 28 | 4 | 27 | 41 |
| 0.5×→1× | 30 | 1 | 1 | 68 |
| 1×→1.5× | 2 | 1 | 0 | 97 |

倍率间共用随机数，属于配对对照；图中的 Wilson 95% 区间是正倍率组的边际二项比例区间，不能通过“区间是否重叠”直接判断配对差异。0×不画置信区间，避免把完全相同轨迹的重复当成独立证据。

## 代表性轨迹及选择方式

每档主代表按全部100回合的 min distance 排序，取下中位数（索引49）；另在存在的成功/失败子集各取下中位数。相同最小距离按 episode_index 打破排序并列。未按最好单例选择，也没有用代表性单例替代整体捕获率。

| 倍率 | 子集 | Episode index | Seed | 结果 | 最小距离m | 回合时间s | 轨迹 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0× | typical | 49 | 1049 | 捕获 | 3.525 | 17.3 | groups/std_0/episode_049.npz |
| 0× | success | 49 | 1049 | 捕获 | 3.525 | 17.3 | groups/std_0/episode_049.npz |
| 0.25× | typical | 64 | 1064 | 捕获 | 4.854 | 17.2 | groups/std_025/episode_064.npz |
| 0.25× | success | 2 | 1002 | 捕获 | 4.022 | 17.7 | groups/std_025/episode_002.npz |
| 0.25× | failure | 15 | 1015 | Enemy Attack Success | 6.585 | 28.2 | groups/std_025/episode_015.npz |
| 0.5× | typical | 78 | 1078 | Enemy Attack Success | 7.000 | 28.2 | groups/std_05/episode_078.npz |
| 0.5× | success | 69 | 1069 | 捕获 | 4.139 | 17.4 | groups/std_05/episode_069.npz |
| 0.5× | failure | 47 | 1047 | Enemy Attack Success | 8.905 | 28.2 | groups/std_05/episode_047.npz |
| 1× | typical | 33 | 1033 | Enemy Attack Success | 13.580 | 28.2 | groups/std_1/episode_033.npz |
| 1× | success | 64 | 1064 | 捕获 | 4.442 | 19.0 | groups/std_1/episode_064.npz |
| 1× | failure | 43 | 1043 | Enemy Attack Success | 13.655 | 28.2 | groups/std_1/episode_043.npz |
| 1.5× | typical | 16 | 1016 | Enemy Attack Success | 18.065 | 28.2 | groups/std_15/episode_016.npz |
| 1.5× | success | 6 | 1006 | 捕获 | 4.957 | 18.2 | groups/std_15/episode_006.npz |
| 1.5× | failure | 92 | 1092 | Enemy Attack Success | 18.536 | 28.2 | groups/std_15/episode_092.npz |

## 主要结论与限制

1. **当前冻结成功策略对动作噪声高度敏感。**捕获率从0×的100/100，下降到0.25×的55/100、0.5×的31/100、原始1×的2/100、1.5×的1/100。原始探索采样明显破坏了该固定场景下原本成功的闭环执行。
2. **降低噪声的效果在这次执行测试中很清楚，但不是训练结论。**0.25×比1×多53次捕获，0.5×比1×多29次；仍远未恢复deterministic策略的表现。本轮没有PPO更新，所以不能据此证明“之前继续PPO后确定性能力退化”完全由探索噪声导致，也不能直接认定某倍率训练最优。
3. 平均最小距离随倍率增大依次为3.525/5.331/8.062/14.631/19.800m，支持噪声增加后闭环拦截精度变差。0×最近3.525m距5m捕获边界只有约1.475m余量，较小横向或时序偏差就可能让原有效拦截错过目标。
4. 平均回报为+444.566/+66.165/−67.012/−226.496/−220.003，最后1.5×比1×略高，**不能把回报描绘为严格单调下降**。回报同时包含终局、距离、时间与碰撞项；捕获率和距离更直接回答本轮噪声敏感性问题。
5. 1×的2/100与1.5×的1/100差别很小，其边际区间大幅重叠；不能强调这两档之间的显著性。强证据是从确定性/低噪声到原始std的整体大幅退化。
6. 当前结果支持把“训练探索强度与成功策略的执行精度不匹配”列为优先验证假设。它既反映噪声影响，也反映该源策略成功轨迹的鲁棒性有限；只在固定0°下评估，不能推广为对新来袭方向的泛化结论。本轮未调整entropy、log_std、learning rate、optimizer或任何训练超参数。

### 碰撞与已有奖励分量

| 倍率 | 有碰撞回合率 | 平均累计碰撞惩罚 | 平均终局项 | 平均距离时间成本 | 平均存活惩罚 |
| --- | --- | --- | --- | --- | --- |
| 0× | 0.0% | 0.000 | 500.000 | -51.974 | -3.460 |
| 0.25× | 47.0% | -98.850 | 230.000 | -60.544 | -4.440 |
| 0.5× | 53.0% | -83.000 | 86.000 | -65.042 | -4.970 |
| 1× | 55.0% | -63.100 | -88.000 | -69.794 | -5.602 |
| 1.5× | 53.0% | -50.900 | -94.000 | -69.483 | -5.620 |

噪声组47%–55%的回合至少发生一次我方安全距离违规，0×为0%；由原环境已有碰撞惩罚计入回报。碰撞惩罚和任务结果的共同变化能解释部分回报下降，但未开展因果分解，不据此修改奖励。

## 视频与可点击结果入口

本轮只有一个冻结 checkpoint，不把它命名为两个不同的 best/final。两个视频来自既有500回合中的全体下中位数代表，未重新推理或挑选最佳单例：

| 推理方式 | Seed | 代表结果 | 原组捕获率 | 完整视频 |
| --- | --- | --- | --- | --- |
| 0× deterministic | 1049 | 17.3秒捕获，173步 | 100/100，同轨迹重复 | [0×视频](../outputs/v12_ActionStd敏感性/std_sensitivity_20261110_seed1000/videos/std_0/episode.mp4) |
| 1×原始std stochastic | 1033 | 28.2秒失守，282步 | 2/100 | [1×视频](../outputs/v12_ActionStd敏感性/std_sensitivity_20261110_seed1000/videos/std_1/episode.mp4) |

两段视频完整包含初始至终止状态，全量解码分别验证174帧与283帧；保留源trajectory/checkpoint SHA、采样倍率、seed、代表选择和整体组表现：[视频核对](../outputs/v12_ActionStd敏感性/std_sensitivity_20261110_seed1000/videos/video_delivery_verification.json)。

- [捕获率主图](../outputs/v12_ActionStd敏感性/std_sensitivity_20261110_seed1000/plots/std_multiplier_vs_capture_rate.png)
- [最小距离分布](../outputs/v12_ActionStd敏感性/std_sensitivity_20261110_seed1000/plots/minimum_distance_distribution.png) · [回报分布](../outputs/v12_ActionStd敏感性/std_sensitivity_20261110_seed1000/plots/episode_return_distribution.png)
- [每倍率典型XY轨迹](../outputs/v12_ActionStd敏感性/std_sensitivity_20261110_seed1000/plots/representative_trajectories_typical_xy.png) · [典型3D轨迹](../outputs/v12_ActionStd敏感性/std_sensitivity_20261110_seed1000/plots/representative_trajectories_typical_3d.png)
- [全部500回合CSV](../outputs/v12_ActionStd敏感性/std_sensitivity_20261110_seed1000/episode_metrics.csv) · [全部500个NPZ目录](../outputs/v12_ActionStd敏感性/std_sensitivity_20261110_seed1000/groups/) · [选中代表来源](../outputs/v12_ActionStd敏感性/std_sensitivity_20261110_seed1000/representatives.json)
- [汇总及冻结状态](../outputs/v12_ActionStd敏感性/std_sensitivity_20261110_seed1000/summary.json) · [独立逐回合重算核对](../outputs/v12_ActionStd敏感性/std_sensitivity_20261110_seed1000/verification.json)

## 工程验证与文件

- 全量 pytest：507 passed in 75.35s；日志：`outputs/v12_ActionStd敏感性/_checks_20261110/full_pytest.log`。
- 源 checkpoint、环境配置、原成功参考轨迹和本轮协议 SHA 均未改变；Actor/Critic state、双 Adam state、原 log_std 及全局 Torch RNG 前后哈希相同，参数 requires_grad=False，无梯度。
- 所有500回合有限数值；0×逐回合状态/动作/奖励完全相同，且第一回合与原成功参考NPZ的共有数组逐位一致。
- 独立从500个NPZ重算初态、std缩放公式、tanh动作、距离/回报/终止条件，并逐seed核对倍率间标准高斯噪声前缀；全部通过。19,121个历史输出及原源码核对保持原内容（允许刷新生成的outputs/README导航）。
- 独立评估入口：`scripts/evaluate_action_std_sensitivity.py`；协议：`configs/eval_action_std_sensitivity.yaml`；独立制图入口：`scripts/plot_action_std_sensitivity.py`。原训练、网络、环境和优化器实现未改。
- `episode_metrics.csv`：500行原始结果；`groups/*/episode_*.npz`：500个完整回合；`representatives.json`：固定选择规则和实际选中来源；`summary.json` / `metadata.json`：汇总及冻结/来源核对。
- `plots/std_multiplier_vs_capture_rate.png`：主捕获率曲线；`plots/minimum_distance_distribution.png` / `episode_return_distribution.png`：原始全点、箱线图和经验CDF；`plots/representative_trajectories_{typical,success,failure}_{xy,3d}.png`：轨迹；`plots/plot_manifest.json`：源哈希及制图说明。XY保持物理等比例；3D为清晰显示轨迹独立缩放轴框，已在标题标注。

```powershell
.\.venv\Scripts\python.exe -B -m scripts.evaluate_action_std_sensitivity --config configs/eval_action_std_sensitivity.yaml --output-dir outputs/v12_ActionStd敏感性/<新的未存在目录>
.\.venv\Scripts\python.exe -B -m scripts.plot_action_std_sensitivity --output-dir outputs/v12_ActionStd敏感性/<本次新目录>
```

评估/制图均拒绝覆盖既有结果。本轮没有新训练，因此不存在新的 best/final checkpoint，也不创建训练轨迹演化图；同一冻结模型的 std 对照轨迹与视频单独标识。
