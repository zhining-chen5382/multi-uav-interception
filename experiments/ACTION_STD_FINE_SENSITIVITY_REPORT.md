# v12：冻结成功策略的低 Action Std 细化测试

**本次直接命中80%–90%经验捕获率的已测倍率是0.15×：83/100。**0.10×为96/100，可作更保守的低噪声备选；0.20×已降为68/100。本轮没有训练，也没有修改学习率、entropy系数、optimizer、reward或环境。

## 实验来源与协议

- 源 checkpoint：`outputs/v08_捕获奖励500/v08-r01_20261108_seed000/checkpoints/update_000180.pt`；训练步数92,160。
- SHA256：`d5d78f3045a2bb9d865028182db21a7128603e02d890cef5dfcabd0ed97609cd`。
- Git：`c6fe7e7c64e62425c079960b3895615b6f274113`；dirty=True。没有commit、merge或push。
- 实际UTC开始：`2026-11-10T14:45:06.642942+00:00`；结束：`2026-11-10T14:46:32.582882+00:00`。
- 600回合推理耗时：85.940秒，不含测试、制图与视频编码。
- 新独立目录：`outputs/v12_ActionStd敏感性/std_sensitivity_fine_20261110_seed1000`；原粗扫目录和结果保留。
- 固定0°，六档0/0.05/0.10/0.15/0.20/0.25×，每档100回合，总600回合。
- 每档使用同一组seed1000–1099；正倍率共用每个seed对应的独立标准高斯噪声前缀。0×不抽取噪声。
- 我方三机和敌机初始位置/速度、基地、Direct Attacker、奖励/观测/动作映射/动力学均按源checkpoint快照执行。
- Actor、Critic和原双Adam状态完全冻结；只在采样时改变tanh前的Normal标准差倍率，未写回log_std。

```text
sigma = exp(clamp(actor.log_std, min_log_std, max_log_std))
raw_action = mean(observation) + multiplier * sigma * epsilon
epsilon ~ Normal(0, I)
normalized_action = tanh(raw_action)
```

std是latent空间尺度，不是tanh后的动作标准差或物理加速度标准差。三机共享三轴std参数，各机/各轴/各步噪声独立；物理加速度、jerk和速度约束按原环境继续生效。

| 量 | X | Y | Z |
| --- | --- | --- | --- |
| 原始log_std | -0.4573156 | -0.4926088 | -0.4646120 |
| clamp后log_std | -0.4573156 | -0.4926088 | -0.4646120 |
| 原始std | 0.6329805 | 0.6110302 | 0.6283789 |

| 倍率 | 有效latent std X | Y | Z |
| --- | --- | --- | --- |
| 0× | 0.0000000 | 0.0000000 | 0.0000000 |
| 0.05× | 0.0316490 | 0.0305515 | 0.0314189 |
| 0.1× | 0.0632981 | 0.0611030 | 0.0628379 |
| 0.15× | 0.0949471 | 0.0916545 | 0.0942568 |
| 0.2× | 0.1265961 | 0.1222060 | 0.1256758 |
| 0.25× | 0.1582451 | 0.1527576 | 0.1570947 |

## 全部600回合结果

| 倍率 | 捕获数/100 | 正倍率Wilson95%区间 | Min distance均值±Std m | 平均回报 | 回报Std | 攻击成功率 | 超时率 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0× | 100/100 | 同轨迹重复，不画区间 | 3.525 ± 0.000 | 444.566 | 0.000 | 0.0% | 0.0% |
| 0.05× | 100/100 | 96.30%–100.00% | 3.736 ± 0.720 | 397.769 | 68.824 | 0.0% | 0.0% |
| 0.1× | 96/100 | 90.16%–98.43% | 3.814 ± 0.962 | 337.717 | 157.056 | 4.0% | 0.0% |
| 0.15× | 83/100 | 74.45%–89.11% | 4.272 ± 1.351 | 228.355 | 264.673 | 17.0% | 0.0% |
| 0.2× | 68/100 | 58.34%–76.33% | 4.860 ± 1.692 | 137.025 | 309.621 | 32.0% | 0.0% |
| 0.25× | 55/100 | 45.24%–64.39% | 5.331 ± 2.035 | 66.165 | 326.413 | 45.0% | 0.0% |

团队reward每步只累计一次。0×100回合是同一确定性轨迹的重复，不能解释为100个独立泛化场景；正倍率衡量的是固定场景中对随机动作噪声的稳健程度。

### Min distance完整分布

| 倍率 | Min | P10 | P25 | Median | P75 | P90 | Max | Std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0× | 3.525 | 3.525 | 3.525 | 3.525 | 3.525 | 3.525 | 3.525 | 0.000 |
| 0.05× | 1.846 | 2.737 | 3.231 | 3.794 | 4.253 | 4.715 | 4.988 | 0.720 |
| 0.1× | 1.516 | 2.598 | 3.221 | 3.834 | 4.468 | 4.786 | 6.857 | 0.962 |
| 0.15× | 1.463 | 2.825 | 3.409 | 4.186 | 4.807 | 5.713 | 9.319 | 1.351 |
| 0.2× | 1.746 | 3.197 | 3.807 | 4.588 | 5.594 | 6.804 | 10.292 | 1.692 |
| 0.25× | 1.996 | 3.127 | 3.884 | 4.862 | 6.563 | 7.785 | 11.728 | 2.035 |

### Episode return完整分布

| 倍率 | Min | P10 | P25 | Median | P75 | P90 | Max | Std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0× | 444.566 | 444.566 | 444.566 | 444.566 | 444.566 | 444.566 | 444.566 | 0.000 |
| 0.05× | 139.538 | 299.560 | 348.271 | 444.535 | 444.575 | 444.629 | 444.695 | 68.824 |
| 0.1× | -241.865 | 169.540 | 258.360 | 444.431 | 444.595 | 444.672 | 444.761 | 157.056 |
| 0.15× | -661.665 | -176.920 | 143.247 | 307.127 | 444.578 | 444.703 | 444.894 | 264.673 |
| 0.2× | -831.788 | -238.398 | -176.203 | 222.262 | 444.531 | 444.707 | 444.945 | 309.621 |
| 0.25× | -717.013 | -331.393 | -177.408 | 92.106 | 444.150 | 444.666 | 444.976 | 326.413 |

均值、分位数和标准差均来自真实100回合；图中保留全点/原始经验CDF，没有平滑或趋势拟合。

### 相邻倍率的配对捕获变化

| 倍率变化 | 成功→失败 | 失败→成功 | 均成功 | 均失败 |
| --- | --- | --- | --- | --- |
| 0×→0.05× | 0 | 0 | 100 | 0 |
| 0.05×→0.1× | 4 | 0 | 96 | 0 |
| 0.1×→0.15× | 13 | 0 | 83 | 4 |
| 0.15×→0.2× | 17 | 2 | 66 | 15 |
| 0.2×→0.25× | 15 | 2 | 53 | 30 |

倍率之间共用噪声，是配对对照。Wilson区间是单组边际比例区间，不能用区间是否重叠直接检验配对差异。个别seed不要求捕获结果随倍率严格单调；本次单调下降的是整体经验捕获率。

## 候选倍率与限制

1. **建议把0.15×列为下一轮低探索retention的首个候选。**这是本轮唯一直接观测到80%–90%经验捕获率的档位，83/100；对应三轴latent std约[0.0949471,0.0916545,0.0942568]。本轮仅提出候选，没有开始训练或写回源模型。
2. **0.10×是更保守的对照候选。**其96/100更利于起始成功策略保持，但超过本次目标的80%–90%范围；0.05×为100/100，探索更小。不能仅凭冻结执行表现判断哪一档后续学习效果更好。
3. **不能宣称整个0.10–0.20×区间都满足80%–90%。**已测两端分别是96%和68%；0.15×只是直接命中的测试点。未测倍率不插值成已验证结论；目前尚未定位一个被独立采样验证的连续合格区间。
4. **83%是100回合的经验结果，不是精确真值或可靠下限。**0.15×的Wilson95%区间为74.45%–89.11%，还不能统计保证真实成功率≥80%。这些seed同时用于筛选候选，因此还存在选择偏差；不能把它当未见seed、变化场景或训练后策略的保证。
5. **冻结执行和继续PPO更新是两个问题。**当前结果支持降低原始探索幅度，不能保证PPO继续更新后仍保留成功能力，也不能证明先前退化只有探索噪声这一个原因。
6. 0×和0.25×的整组summary与上一轮粗扫逐字段完全一致；两轮共有场景/seed/模型/噪声设置复现。新细扫没有替换粗扫的500回合结果。

### 碰撞与回报的解释

| 倍率 | 安全距离违规回合率 | 平均累计碰撞惩罚 | 平均终局项 | 平均距离成本 | 平均存活惩罚 |
| --- | --- | --- | --- | --- | --- |
| 0× | 0.0% | 0.000 | 500.000 | -51.974 | -3.460 |
| 0.05× | 39.0% | -46.800 | 500.000 | -51.974 | -3.457 |
| 0.1× | 45.0% | -82.000 | 476.000 | -52.736 | -3.546 |
| 0.15× | 47.0% | -110.600 | 398.000 | -55.214 | -3.832 |
| 0.2× | 46.0% | -108.750 | 308.000 | -58.069 | -4.156 |
| 0.25× | 47.0% | -98.850 | 230.000 | -60.544 | -4.440 |

即使0.05×全部捕获，平均回报也低于0×，主要已有差异来自我方安全距离违规惩罚：平均−46.8分。0.15×平均碰撞惩罚−110.6分，不能只用捕获率推断整体轨迹安全或团队协同已解决。违规是原环境的两两距离阈值统计，未增加新的碰撞模型或惩罚。

## 代表性轨迹

每档主代表按全部100回合min distance排序，取下中位数（索引49）；相同距离按episode_index排序。另在存在的成功/失败子集分别取下中位数。没有挑选最好单例，也不将代表轨迹的结果代替整体捕获率。

| 倍率 | 子集 | Episode | Seed | 结果 | 最小距离m | 时间s | 来源NPZ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0× | typical | 49 | 1049 | 捕获 | 3.525 | 17.3 | groups/std_0/episode_049.npz |
| 0× | success | 49 | 1049 | 捕获 | 3.525 | 17.3 | groups/std_0/episode_049.npz |
| 0.05× | typical | 7 | 1007 | 捕获 | 3.786 | 17.3 | groups/std_005/episode_007.npz |
| 0.05× | success | 7 | 1007 | 捕获 | 3.786 | 17.3 | groups/std_005/episode_007.npz |
| 0.1× | typical | 60 | 1060 | 捕获 | 3.833 | 17.3 | groups/std_010/episode_060.npz |
| 0.1× | success | 67 | 1067 | 捕获 | 3.803 | 17.3 | groups/std_010/episode_067.npz |
| 0.1× | failure | 84 | 1084 | Enemy Attack Success | 5.998 | 28.2 | groups/std_010/episode_084.npz |
| 0.15× | typical | 29 | 1029 | 捕获 | 4.156 | 17.3 | groups/std_015/episode_029.npz |
| 0.15× | success | 33 | 1033 | 捕获 | 3.954 | 17.3 | groups/std_015/episode_033.npz |
| 0.15× | failure | 47 | 1047 | Enemy Attack Success | 6.004 | 28.2 | groups/std_015/episode_047.npz |
| 0.2× | typical | 21 | 1021 | 捕获 | 4.586 | 17.3 | groups/std_020/episode_021.npz |
| 0.2× | success | 2 | 1002 | 捕获 | 4.217 | 17.6 | groups/std_020/episode_002.npz |
| 0.2× | failure | 19 | 1019 | Enemy Attack Success | 6.221 | 28.2 | groups/std_020/episode_019.npz |
| 0.25× | typical | 64 | 1064 | 捕获 | 4.854 | 17.2 | groups/std_025/episode_064.npz |
| 0.25× | success | 2 | 1002 | 捕获 | 4.022 | 17.7 | groups/std_025/episode_002.npz |
| 0.25× | failure | 15 | 1015 | Enemy Attack Success | 6.585 | 28.2 | groups/std_025/episode_015.npz |

XY图保持物理等比例；3D图为清晰显示独立缩放轴框，标题已注明，不用视觉角度/长度替代数值分析。

## 完整代表回合视频

两段均来自本轮600回合的实际NPZ，按各组全部100回合最近距离的下中位数选取，没有按捕获成功筛选。只有同一个92,160步冻结模型，不虚构新的best/final。

| 倍率 | Seed | 本回合结果 | 整组捕获率 | 完整解码帧数 | 视频 |
| --- | --- | --- | --- | --- | --- |
| 0× | 1049 | 17.3秒捕获，最近3.525m | 100/100，同轨迹重复 | 174/174 | [确定性参考](../outputs/v12_ActionStd敏感性/std_sensitivity_fine_20261110_seed1000/videos/std_0/episode.mp4) |
| 0.15× | 1029 | 17.3秒捕获，最近4.156m | 83/100 | 174/174 | [0.15×典型回合](../outputs/v12_ActionStd敏感性/std_sensitivity_fine_20261110_seed1000/videos/std_015/episode.mp4) |

每状态一帧，含初始与终止，保留来源manifest和完整解码验证：[视频交付核对](../outputs/v12_ActionStd敏感性/std_sensitivity_fine_20261110_seed1000/videos/video_delivery_verification.json)。0.15×单例成功不能替代整组83%的表现。

## 文件、复现与验证

- 本轮实际专项pytest：41 passed in 4.26s，见[专项日志](../outputs/v12_ActionStd敏感性/_fine_checks_20261110/pytest.log)。原评估代码未改，未把上一轮507全量测试冒称为本轮重跑。
- Actor/Critic、双Adam及log_std前后哈希相同；源checkpoint、协议、参考环境和原成功轨迹SHA保持不变；参数不求梯度、无训练或optimizer更新。
- 600个完整回合均保存实际初态、状态/动作/奖励及latent mean/base_std/epsilon/raw_action；所有回合初始状态相同且无NaN/Inf。0×全部逐位复现，且与历史成功参考共有数组逐位一致。
- [独立逐回合重算核对](../outputs/v12_ActionStd敏感性/std_sensitivity_fine_20261110_seed1000/verification.json)全部通过：0×/0.25×共200回合的3,200个数组、CSV和NPZ metadata与粗扫逐位一致，762个保护文件及102个原source文件保持原内容。倍率间标准高斯噪声前缀、sampling公式、终止/距离/回报和初始状态均逐项核对。
- 复用既有`scripts/evaluate_action_std_sensitivity.py`和`scripts/plot_action_std_sensitivity.py`，仅新增细扫YAML与本报告/导航记录；没有改训练代码或算法。

- [细扫协议](../configs/eval_action_std_sensitivity_fine.yaml)
- [捕获率曲线](../outputs/v12_ActionStd敏感性/std_sensitivity_fine_20261110_seed1000/plots/std_multiplier_vs_capture_rate.png)
- [Min distance分布](../outputs/v12_ActionStd敏感性/std_sensitivity_fine_20261110_seed1000/plots/minimum_distance_distribution.png) · [Return分布](../outputs/v12_ActionStd敏感性/std_sensitivity_fine_20261110_seed1000/plots/episode_return_distribution.png)
- [每倍率典型XY轨迹](../outputs/v12_ActionStd敏感性/std_sensitivity_fine_20261110_seed1000/plots/representative_trajectories_typical_xy.png) · [典型3D轨迹](../outputs/v12_ActionStd敏感性/std_sensitivity_fine_20261110_seed1000/plots/representative_trajectories_typical_3d.png)
- [全部600回合CSV](../outputs/v12_ActionStd敏感性/std_sensitivity_fine_20261110_seed1000/episode_metrics.csv) · [600个NPZ目录](../outputs/v12_ActionStd敏感性/std_sensitivity_fine_20261110_seed1000/groups/) · [代表选择来源](../outputs/v12_ActionStd敏感性/std_sensitivity_fine_20261110_seed1000/representatives.json)
- [本轮汇总与冻结核验](../outputs/v12_ActionStd敏感性/std_sensitivity_fine_20261110_seed1000/summary.json) · [配对分析](../outputs/v12_ActionStd敏感性/std_sensitivity_fine_20261110_seed1000/analysis_summary.json)
- [原粗扫报告](ACTION_STD_SENSITIVITY_REPORT.md)，与本轮分开保存。

```powershell
.\.venv\Scripts\python.exe -B -m scripts.evaluate_action_std_sensitivity --config configs/eval_action_std_sensitivity_fine.yaml --output-dir outputs/v12_ActionStd敏感性/<新的未存在目录>
.\.venv\Scripts\python.exe -B -m scripts.plot_action_std_sensitivity --output-dir outputs/v12_ActionStd敏感性/<本次新目录>
```

评估/制图均拒绝覆盖既有结果。本轮同一冻结checkpoint没有新增best/final，也没有训练轨迹演化；不同std的轨迹均按实际采样方式标识。
