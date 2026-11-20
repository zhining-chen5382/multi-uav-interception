# v09-r02：从 0° 起步的来袭方向课程实验

本报告基于本轮实际训练、逐回合评估 CSV 和 checkpoint 核对生成。原 v08 与 v09-r01 结果保持保留。

## 来源与规则

- 独立 run：`v09-r02`；输出：`outputs/v09_来袭方向课程/v09-r02_20261109_seed000`。
- 路线：0° → ±2° → ±5° → ±10° → ±30° → ±60° → ±90°。0° 从零，不使用任何旧成功模型初始化。
- 每阶段最多 100,000 步；每 10,000 步更新后确定性评估。0° 每次 1 回合，连续两个 checkpoint 捕获才升级；正角度固定 100 个均匀角度，连续两次捕获率≥80%才升级。
- 当前阶段历史 best-eval 按捕获率、较低失守率、较短成功拦截时间、较小平均最小距离、较低碰撞回合率依次选择；奖励不参与排名。
- 升级继承 Actor、Critic、双 Adam 及全局 RNG；前阶段固定评估集仅监控遗忘，不参与晋级。0° 前阶段监控同样只用 1 回合。
- reward、observation、动作、Direct Attacker、动力学、网络和 PPO 沿用 v08，已对源快照逐项核验。
- Git `c6fe7e7c64e62425c079960b3895615b6f274113`，dirty=True，seed=0；实际新配置 `configs/mappo_direction_curriculum_zero_start.yaml`。

## 实际结果

实际状态 `stopped_stage_budget`；训练 100,000 步、200 updates、354 完整训练回合，其中随机探索捕获 1 次；确定性评估实际运行 10 回合。
尚未开始阶段：[2.0, 5.0, 10.0, 30.0, 60.0, 90.0]。

| 阶段 | 实际阶段步数 | 状态 | best 自身累计步数 | best 捕获 | last 捕获 | 评估回合数/次 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 0.0° | 100000 | stopped_stage_budget | 70000 | 0% | 0% | 1 |

## 阶段 1：0.0°

| 实际累计步数 | 角色 / 角度 | 捕获 | 失守 | 成功平均时间(s) | min距离(m) | 碰撞回合率 | 连续达标 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10000 | current 0.0° | 0% | 100% | 无成功 | 169.273 | 0% | 0 |
| 20000 | current 0.0° | 0% | 100% | 无成功 | 87.192 | 100% | 0 |
| 30000 | current 0.0° | 0% | 100% | 无成功 | 91.466 | 100% | 0 |
| 40000 | current 0.0° | 0% | 100% | 无成功 | 77.126 | 100% | 0 |
| 50000 | current 0.0° | 0% | 100% | 无成功 | 112.263 | 100% | 0 |
| 60000 | current 0.0° | 0% | 100% | 无成功 | 81.081 | 100% | 0 |
| 70000 | current 0.0° | 0% | 100% | 无成功 | 73.405 | 100% | 0 |
| 80000 | current 0.0° | 0% | 100% | 无成功 | 150.156 | 0% | 0 |
| 90000 | current 0.0° | 0% | 100% | 无成功 | 222.347 | 0% | 0 |
| 100000 | current 0.0° | 0% | 100% | 无成功 | 140.936 | 100% | 0 |

best-eval：`outputs/v09_来袭方向课程/v09-r02_20261109_seed000/stage_01_pm0/checkpoints/best_eval.pt`，实际累计 70000 步；last：`outputs/v09_来袭方向课程/v09-r02_20261109_seed000/stage_01_pm0/checkpoints/last.pt`。
阶段间继承核对：`{"fresh_start": true, "parent_checkpoint": null}`。

首/末各 20% 样本均值（精确样本窗口见 JSON，不是曲线平滑）：
- episode_reward：-234.438 → -227.791；范围 [-517.437, 431.674]。
- min_distance：65.4128 → 144.578；范围 [4.62703, 236.711]。
- episode_length：281.443 → 282；范围 [243, 282]。
- actor_loss：-0.00680013 → -0.00643299；范围 [-0.0133724, 0.000450439]。
- value_loss：108.915 → 26.0543；范围 [0.827531, 1999.51]。
- entropy：1.78043 → 1.74471；范围 [1.58431, 1.8067]。
- approx_kl：0.00521497 → 0.0055513；范围 [0.00245878, 0.00837257]。

## 验证与分析边界

- 完整 pytest：388 passed in 49.31s，出处 `outputs/v09_来袭方向课程/v09-r02_20261109_seed000/preflight/pytest.log`。
- 实际 14 个 CSV、33 个 PT 全部数值有限，包含模型和 Adam 状态，无 NaN/Inf；这不代表策略性能稳定或已经协同。
- 0° 的 0%/100% 只表示单个固定初态是否捕获，不能解释为泛化成功率；正角度固定网格用于晋级和选模，也不是独立留出测试。
- 随机训练回合捕获与确定性策略表现是不同指标；不能仅凭 reward 或 loss 改善判断任务学会。
- 梯度范数按现 MAPPO 实现为裁剪前总 L2 范数的 minibatch 样本加权均值；实际更新前仍按 max_grad_norm=0.5 裁剪。
- 本轮因预算停止，未把已执行前缀误报为全课程成功。

## 曲线与可视化

曲线全部使用原始实际数据；当前阶段与前阶段监控分线，0° gate 为 100%、正角度 gate 为 80%。
- [evaluation_capture_rate.png](../outputs/v09_来袭方向课程/v09-r02_20261109_seed000/plots/evaluation_capture_rate.png)
- [evaluation_min_distance.png](../outputs/v09_来袭方向课程/v09-r02_20261109_seed000/plots/evaluation_min_distance.png)
- [training_episodes.png](../outputs/v09_来袭方向课程/v09-r02_20261109_seed000/plots/training_episodes.png)
- [actor_value_loss.png](../outputs/v09_来袭方向课程/v09-r02_20261109_seed000/plots/actor_value_loss.png)
- [entropy.png](../outputs/v09_来袭方向课程/v09-r02_20261109_seed000/plots/entropy.png)
- [训练后 XY 轨迹演化图](../outputs/v09_来袭方向课程/v09-r02_20261109_seed000/trajectory_evolution/trajectory_evolution_xy.png)

## 配套完整视频

两模型采用同一角度 0.0°、seed 1000；选择方式：Best-eval has no successful fixed-grid angles; use the middle of its 1 evaluation angles for both policies, with an identical seed. This illustration is not a capture-rate estimate.。
- [best 完整视频](../outputs/v09_来袭方向课程/v09-r02_20261109_seed000/videos/best/best_full_episode.mp4)：checkpoint 70000 步，Enemy Attack Success，最近 73.405m。
- [final 完整视频](../outputs/v09_来袭方向课程/v09-r02_20261109_seed000/videos/final/final_full_episode.mp4)：checkpoint 100000 步，Enemy Attack Success，最近 140.936m。

本轮只改变课程路线与对应评估调度，没有改奖励、观测、动力学或 MAPPO 更新。旧 v09-r01 和历史报告未覆盖。

## 实现变更与复现

| 文件 | 变化 |
| --- | --- |
| `configs/mappo_direction_curriculum_zero_start.yaml` | 独立七阶段配置；0° 单场景/阈值1.0，其余100场景/阈值0.8，每阶段100k |
| `utils/direction_curriculum.py` | 0° 固定 reset 不消耗角度 RNG；单点评估角；按阶段读取评估参数 |
| `scripts/train_direction_curriculum.py` | 默认新路线；当前及前阶段评估采用正确场景数；旧正角路线显式配置保持兼容 |
| `tests/test_zero_start_curriculum.py` | 新增16项专项；完整388项通过 |
| README、BASELINE_SPEC、实验registry及索引 | 新独立r02入口、规则与真实结果；保留r01 |

正式执行命令（已执行；同目录不可重复覆盖）：

```powershell
.\.venv\Scripts\python.exe -B -u -m scripts.train_direction_curriculum --config configs/mappo_direction_curriculum_zero_start.yaml
```

本轮没有实际进入 ±2°，因此后续阶段的学习效果和泛化收益尚未验证；跨阶段继承已在短集成测试中核对。
0° 固定任务本身仍未在100k内满足规定的连续成功要求，不能通过改用旧成功模型、额外加预算或跳阶段来掩盖这一结果。
