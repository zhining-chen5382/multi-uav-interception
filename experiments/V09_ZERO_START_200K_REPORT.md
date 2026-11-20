# v09-r02：100k → 200k 续训报告

本报告核对同一 run 的实际续训，不把原 100k 与累计 200k 重复计为两次独立训练。原 100k、v09-r01 和 v08 保留。

## 来源、恢复与规则

- Run：`v09-r02`；累计结果目录：`outputs/v09_来袭方向课程/v09-r02_20261109_seed000/continuation_200k`；原始 100k：`outputs/v09_来袭方向课程/v09-r02_20261109_seed000`。
- 原 run 起始 UTC：`2026-11-09T09:14:02.479794+00:00`；本次续训起始 UTC：`2026-11-09T11:15:17.479151+00:00`。
- 恢复原 100k 的末次 `last.pt`，不从 70k 历史 best 重新起步；源文件 `outputs/v09_来袭方向课程/v09-r02_20261109_seed000/stage_01_pm0/checkpoints/last.pt`，SHA256 `7e5078f3209b1e53a51c291a39fef0ab91bdd4ff90885f42892344549618b962`。
- `resume_initial.pt` 与源 last 的 Actor、Critic、双 Adam、全局 RNG、角度采样 RNG、环境 RNG/内部状态、动力学状态及未结束回合逐项精确一致。恢复时该回合已执行 211 步，继续该回合而非 reset。
- 路线保持 0° → ±2° → ±5° → ±10° → ±30° → ±60° → ±90°；原始 0° 从零初始化，本次是完整状态续训。
- 用户授权将累计总预算延至 200k，0° 阶段上限延至 200k，后续阶段各 100k；实际配置 stage_step_budgets=`[200000, 100000, 100000, 100000, 100000, 100000, 100000]`，全局上限=200000。
- 每 10k 评估。0° 固定 1 回合，连续两个评估 checkpoint 都成功才升级；正角度固定 100 个均匀角度，连续两次捕获率≥80%才升级。前阶段监控不参与升级。
- 历史 best-eval 按捕获率、低失守率、短成功时间、低最小距离、低碰撞回合率排序，reward 不参与；阶段晋级继承 best 的双网络、双 Adam 和全局 RNG。
- reward、observation、动作、Direct Attacker、动力学、网络、PPO、seed、rollout 等均核验与原 100k/v08 一致；只扩大获授权的训练预算。
- Git `c6fe7e7c64e62425c079960b3895615b6f274113`，dirty=True，seed=0；续训配置 `configs/mappo_direction_curriculum_zero_start_200k.yaml`。下一次从零训练配置仍为 `configs/mappo_direction_curriculum_zero_start.yaml`。

## 实际结果

状态 `stopped_stage_budget`；累计 200,000 步、400 updates、709 完整训练回合、1 次训练采样捕获。本次新增 100,000 步、355 完整回合、0 次训练采样捕获。
累计实际确定性评估 20 回合，含原 100k 复制保留的历史。尚未开始阶段：[2.0, 5.0, 10.0, 30.0, 60.0, 90.0]。

| 阶段 | 实际阶段步数 | 状态 | best 自身累计步数 | best 捕获 | last 捕获 | 评估回合数/次 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 0.0° | 200000 | stopped_stage_budget | 70000 | 0% | 0% | 1 |

### 原 100k 与当前模型比较

| 模型 | 自身累计训练步数 | 捕获 | 失守 | reward | min距离(m) | 碰撞步数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原 100k 历史 best | 70000 | 0% | 100% | -258.439 | 73.405 | 15 |
| 原 100k last | 100000 | 0% | 100% | -389.010 | 140.936 | 40 |
| 累计当前历史 best | 70000 | 0% | 100% | -258.439 | 73.405 | 15 |
| 累计当前 last | 200000 | 0% | 100% | -194.614 | 204.414 | 0 |

0° 的每次评估只对应一个固定确定性初态，表中 0%/100% 不能解释为泛化性能。历史 best 若仍是 70k，只说明新增评估未超过该选择标准，不表示发生新的成功。

### Reward 改善的实际来源

同一 0° 固定场景：100k last reward -389.010 → 200k last -194.614，改善 194.396；碰撞惩罚 -200.000 → 0.000，该项改善 200.000；距离时间成本 -83.370 → -88.974，该项变化 -5.604。
最近敌我距离 140.936m → 204.414m；本次新增评估捕获 0/10，训练新增采样捕获 0。Reward 改善主要来自碰撞减少，距离指标未改善，不能认为已获得更好的拦截策略；不据此自动改动 reward。

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
| 110000 | current 0.0° | 0% | 100% | 无成功 | 171.782 | 100% | 0 |
| 120000 | current 0.0° | 0% | 100% | 无成功 | 164.314 | 0% | 0 |
| 130000 | current 0.0° | 0% | 100% | 无成功 | 158.437 | 100% | 0 |
| 140000 | current 0.0° | 0% | 100% | 无成功 | 152.014 | 100% | 0 |
| 150000 | current 0.0° | 0% | 100% | 无成功 | 132.473 | 100% | 0 |
| 160000 | current 0.0° | 0% | 100% | 无成功 | 196.488 | 0% | 0 |
| 170000 | current 0.0° | 0% | 100% | 无成功 | 175.343 | 0% | 0 |
| 180000 | current 0.0° | 0% | 100% | 无成功 | 248.836 | 100% | 0 |
| 190000 | current 0.0° | 0% | 100% | 无成功 | 147.302 | 0% | 0 |
| 200000 | current 0.0° | 0% | 100% | 无成功 | 204.414 | 0% | 0 |

best-eval：`outputs/v09_来袭方向课程/v09-r02_20261109_seed000/continuation_200k/stage_01_pm0/checkpoints/best_eval.pt`，实际累计 70000 步；last：`outputs/v09_来袭方向课程/v09-r02_20261109_seed000/continuation_200k/stage_01_pm0/checkpoints/last.pt`。
阶段间继承核对：`{"original_fresh_start": true, "parent_checkpoint": null, "continuation_from_steps": 100000, "continuation_resume_exact": true}`。

首/末各 20% 样本均值（精确样本窗口见 JSON，不是曲线平滑）：

- episode_reward：-236.74 → -245.422；范围 [-609.9, 431.674]。
- min_distance：80.9808 → 194.754；范围 [4.62703, 271.857]。
- episode_length：281.723 → 282；范围 [243, 282]。
- actor_loss：-0.00581667 → -0.00500898；范围 [-0.0141052, 0.00080726]。
- value_loss：69.5809 → 28.5233；范围 [0.776984, 1999.51]。
- entropy：1.75832 → 1.6965；范围 [1.58431, 1.8067]。
- approx_kl：0.00512942 → 0.00638772；范围 [0.00245878, 0.0130365]。

仅本次新增完整回合的首/末 20% 样本均值：

- episode_reward：-264.904 → -250.185；范围 [-609.9, -180.863]。
- min_distance：158.853 → 191.72；范围 [66.6965, 271.857]。
- episode_length：282 → 282；范围 [282, 282]。

## 验证与分析边界

- 原 100k 开训前完整 pytest：388 passed in 49.31s，出处 `outputs/v09_来袭方向课程/v09-r02_20261109_seed000/preflight/pytest.log`。本次沿用该全量证据，没有重新执行全套；核心训练源码与原测试时一致。
- 本次实际续训专项 pytest：30 passed in 22.49s，出处 `outputs/v09_来袭方向课程/v09-r02_20261109_seed000/continuation_200k/preflight/pytest.log`；不把本次专项数量计为全套测试。
- 实际 24 个 CSV、64 个 PT 全部数值有限，包含模型和 Adam 状态，无 NaN/Inf；这不代表策略性能稳定或已经协同。
- 0° 的 0%/100% 只表示单个固定初态是否捕获，不能解释为泛化成功率；正角度固定网格用于晋级和选模，也不是独立留出测试。
- 随机训练回合捕获与确定性策略表现是不同指标；不能仅凭 reward 或 loss 改善判断任务学会。
- 梯度范数按现 MAPPO 实现为裁剪前总 L2 范数的 minibatch 样本加权均值；实际更新前仍按 max_grad_norm=0.5 裁剪。
- 本轮因预算停止，未把已执行前缀误报为全课程成功。

## 曲线与可视化

曲线全部使用原始实际数据，不平滑；黑色虚线标记 100k 续训分界（loss/entropy 对应 update 200）。当前阶段与前阶段监控分线，0° gate 为 100%、正角度 gate 为 80%。

- [evaluation_capture_rate.png](../outputs/v09_来袭方向课程/v09-r02_20261109_seed000/continuation_200k/plots/evaluation_capture_rate.png)
- [evaluation_min_distance.png](../outputs/v09_来袭方向课程/v09-r02_20261109_seed000/continuation_200k/plots/evaluation_min_distance.png)
- [training_episodes.png](../outputs/v09_来袭方向课程/v09-r02_20261109_seed000/continuation_200k/plots/training_episodes.png)
- [actor_value_loss.png](../outputs/v09_来袭方向课程/v09-r02_20261109_seed000/continuation_200k/plots/actor_value_loss.png)
- [entropy.png](../outputs/v09_来袭方向课程/v09-r02_20261109_seed000/continuation_200k/plots/entropy.png)
- [训练后 XY 轨迹演化图](../outputs/v09_来袭方向课程/v09-r02_20261109_seed000/continuation_200k/trajectory_evolution/trajectory_evolution_xy.png)
- [演化图逐 checkpoint 指标 CSV](../outputs/v09_来袭方向课程/v09-r02_20261109_seed000/continuation_200k/trajectory_evolution/trajectory_evolution_metrics.csv)

## 配套完整视频

两模型采用同一角度 0.0°、seed 1000；选择方式：Best-eval has no successful fixed-grid angles; use the middle of its 1 evaluation angles for both policies, with an identical seed. This illustration is not a capture-rate estimate.。final 视频使用阶段 `last.pt`，未覆盖或改名权重。

- [best 完整视频](../outputs/v09_来袭方向课程/v09-r02_20261109_seed000/continuation_200k/videos/best/best_full_episode.mp4)：checkpoint 70000 步，Enemy Attack Success，最近 73.405m。
- [final 完整视频](../outputs/v09_来袭方向课程/v09-r02_20261109_seed000/continuation_200k/videos/final/final_full_episode.mp4)：checkpoint 200000 步，Enemy Attack Success，最近 204.414m。

## 结论

新增训练是否带来有效确定性拦截，应由评估捕获和最小距离共同判断。训练采样成功、reward 改善或 loss 下降均不等于确定性任务成功。
本次严格在预算停止后保存；未启动未达升级条件的后续阶段，也未另行调参或重训挑选结果。原始 100k 报告、权重和评估保留；registry 同一 v09-r02 中保存 100k_stage 入口，新增模型使用 continuation-200k 后缀。
