# V09 来袭方向课程训练结果

本报告由已结束正式运行的真实 CSV、summary 和 checkpoint 自动核对生成。

- 输出目录：`E:\projects\research\multi-uav-interception\outputs\v09_来袭方向课程\v09-r01_20261109_seed000`
- 终止状态：`stopped_stage_budget`
- 实际训练步数：100000；更新：200；完整训练回合：354；训练捕获成功回合：3。
- 所有实际 CSV 数值及全部 PT 检查通过，无 NaN/Inf；这不单独证明策略稳定或存在协同。
- 未开始阶段：[30.0, 60.0, 90.0]。

## 可复现性与设定核对

- code_version：`c6fe7e7c64e62425c079960b3895615b6f274113`；训练开始时 git_dirty=`True`（因此 commit hash 单独不足以复现，需结合 source_snapshot 和 preflight/source_manifest.json）。
- seed=0；fresh_stage1=True；initial_checkpoint=None。已核对 initial Adam 为空。
- 训练配置源：`E:\projects\research\multi-uav-interception\configs\mappo_direction_curriculum.yaml`。
- 环境配置源：`E:\projects\research\multi-uav-interception\configs\baseline_capture500.yaml`。
- 对照本次 source_snapshot 中原 v08 mappo_capture500.yaml：network、PPO、normalization、rollout_steps 和完整环境 YAML 均一致。课程仅另加方位角采样、固定网格评估与晋级调度；未变更原 v08 网络和 PPO 超参数。
- ±10° 从零训练，之后计划 ±30°/±60°/±90°；每阶段预算上限 100k、每 10k 评估固定 100 角度、连续两次 ≥80% 晋级；本次实际执行以结果表为准。

## 各阶段实际结果

| 阶段 | 实际步数 | 更新 | 训练成功/回合 | best 所在阶段步数 | best 捕获率 | 阶段状态 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ±10° | 100000 | 200 | 3/354 | 30000 | 9.0% | stopped_stage_budget |

### 阶段 1：±10°

best：`E:\projects\research\multi-uav-interception\outputs\v09_来袭方向课程\v09-r01_20261109_seed000\stage_01_pm10\checkpoints\best_eval.pt`，来自阶段 30000 步／实际累计 30000 步。

| 评估阶段步数 | 集合 | 捕获率 | 敌方成功率 | 成功平均拦截时间(s) | 平均最小距离(m) | 碰撞回合率 | 连续达标 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10000 | current ±10° | 0.0% | 100.0% | 无成功回合 | 168.556 | 60.0% | 0 |
| 20000 | current ±10° | 0.0% | 100.0% | 无成功回合 | 88.860 | 68.0% | 0 |
| 30000 | current ±10° | 9.0% | 91.0% | 22.900 | 26.662 | 36.0% | 0 |
| 40000 | current ±10° | 0.0% | 100.0% | 无成功回合 | 76.913 | 36.0% | 0 |
| 50000 | current ±10° | 0.0% | 100.0% | 无成功回合 | 144.395 | 39.0% | 0 |
| 60000 | current ±10° | 0.0% | 100.0% | 无成功回合 | 196.921 | 72.0% | 0 |
| 70000 | current ±10° | 0.0% | 100.0% | 无成功回合 | 217.698 | 17.0% | 0 |
| 80000 | current ±10° | 0.0% | 100.0% | 无成功回合 | 158.637 | 7.0% | 0 |
| 90000 | current ±10° | 0.0% | 100.0% | 无成功回合 | 184.980 | 8.0% | 0 |
| 100000 | current ±10° | 0.0% | 100.0% | 无成功回合 | 229.919 | 26.0% | 0 |

best 与预算结束 last 对比（分别使用对应 current 固定网格评估）：

| 模型 | 阶段步数 | 捕获率 | 敌方成功率 | 超时率 | 平均 reward | 平均最小距离(m) | 碰撞回合率 | 碰撞 step 比例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| best_eval.pt | 30000 | 9.0% | 91.0% | 0.0% | -246.266 | 26.662 | 36.0% | 8.94% |
| last.pt | 100000 | 0.0% | 100.0% | 0.0% | -331.421 | 229.919 | 26.0% | 9.32% |

best 成功角度：2.323232, 2.525253, 2.727273, 2.929293, 3.131313, 3.333333, 3.535354, 3.737374, 3.939394°；范围 [2.323232°, 3.939394°]，跨度 1.616162°。
这些成功角度属于同一段连续采样点，说明成功集中于狭窄方向范围；网格之间未单独评估，不能断言整个连续区间都成功。

阶段 1 从零开始；initial checkpoint 的两组 Adam state 均为空，未使用历史 v08 权重。

本阶段用尽预算后未满足连续达标条件，流程停止，未自动晋级。

训练过程（按样本数取首/末 20%，精确窗口见 JSON；不是曲线平滑）：
- episode_reward：首段平均 -245.7753 → 末段平均 -262.2086；范围 [-634.1286, 426.9838]。
- min_distance：首段平均 76.9380 → 末段平均 171.7415；范围 [4.2714, 304.9188]。
- episode_length：首段平均 281.6571 → 末段平均 282.0000；范围 [233.0000, 282.0000]。
- actor_loss：首段平均 -0.00658308 → 末段平均 -0.0070273；范围 [-0.0134162, -0.000614197]。
- value_loss：首段平均 121.436 → 末段平均 33.3681；范围 [1.48173, 2180.9]。
- entropy：首段平均 1.76985 → 末段平均 1.72352；范围 [1.57097, 1.80885]。
- approx_kl：首段平均 0.00463819 → 末段平均 0.00585578；范围 [0.00195711, 0.0103516]。
- actor_grad_norm：首段平均 1.68905 → 末段平均 3.19004；范围 [1.33983, 5.23434]。
- critic_grad_norm：首段平均 468.611 → 末段平均 721.92；范围 [42.4461, 2757.6]。

梯度范数口径：algorithms/mappo/mappo.py 的 MAPPOAgent.update 记录 clip_grad_norm_ 返回的裁剪前总 L2 范数，按 minibatch 样本量对所有 PPO epochs 加权平均；实际 Adam 更新前的梯度已按 max_grad_norm=0.5 裁剪。较大的记录范数不能直接解释为数值发散。

观察到阶段 30000 步短暂改善，随后最终捕获率退化为 0.0%，而非持续学习改善；全部数值有限不能消除这种行为退化。

本次未达到初阶段训练目标，后续阶段从未启动，无法据此证明扩大来袭方向的课程训练带来收益，也没有在这次正式运行中实际发生跨阶段 optimizer 继承。

## 图表与原始数据

图像使用实际原始点，无滑动平均、插值或补造趋势。current 与 previous 分线，阶段分色；previous 仅用于遗忘监测。
- [evaluation_capture_rate.png](../outputs/v09_来袭方向课程/v09-r01_20261109_seed000/plots/evaluation_capture_rate.png)
- [evaluation_min_distance.png](../outputs/v09_来袭方向课程/v09-r01_20261109_seed000/plots/evaluation_min_distance.png)
- [reward.png](../outputs/v09_来袭方向课程/v09-r01_20261109_seed000/plots/reward.png)
- [actor_value_loss.png](../outputs/v09_来袭方向课程/v09-r01_20261109_seed000/plots/actor_value_loss.png)
- [entropy.png](../outputs/v09_来袭方向课程/v09-r01_20261109_seed000/plots/entropy.png)

各阶段 training_metrics.csv、updates.csv 和逐回合 evaluation_metrics.csv 均保留原样；数值范围、首末趋势、文件哈希及 checkpoint 清单见运行目录 analysis_summary.json。

## 结论边界

best 按捕获率优先，其次防御失败率、成功回合平均捕获时间、平均最小距离、碰撞回合率选择。
训练中随机探索出现捕获，不等于确定性策略已在整个方向区间学会任务。固定 100 角度评估用于晋级和选模，不能替代独立未见方向的泛化测试。
有限数值检查不能证明控制动作没有饱和，也不能证明角色分工；本脚本未执行新的推理、训练或轨迹评估。

## 训练后补充：完整仿真视频

本节在上述分析完成后新增，仅重新执行冻结模型的确定性推理并录制视频，没有重新训练或修改原评估数据。
两个模型统一使用来袭角度 +3.131313°、seed 1065，对应原固定评估集第 66 回合。
该点是 best 的 9 个成功网格点的中间一点；这是选择过的成功示例，整体捕获率仍为 best 9%、final 0%。

| 模型 | Checkpoint 步数 | 该回合结果 | 最小距离 | 回报 |
| --- | ---: | --- | ---: | ---: |
| [best 视频](../outputs/v09_来袭方向课程/v09-r01_20261109_seed000/videos/best/best_full_episode.mp4) | 30000 | 22.6s 捕获，226 步 | 4.0549m | +429.1548 |
| [final 视频](../outputs/v09_来袭方向课程/v09-r01_20261109_seed000/videos/final/final_full_episode.mp4) | 100000 | 28.2s 敌方攻击成功，282 步 | 218.1110m | −200.0668 |

final 视频实际读取 `stage_01_pm10/checkpoints/last.pt`，未改名覆盖模型。
本次重新记录的全部回合指标均与对应原评估 CSV 行一致。
完整轨迹、原始来源、manifest 与视频解码检查见 [视频说明](../outputs/v09_来袭方向课程/v09-r01_20261109_seed000/videos/README.md)。
