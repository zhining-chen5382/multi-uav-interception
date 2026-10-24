# Baseline v1：单次 100000 步 MAPPO 正式实验

日期：2026-10-24。输出目录：`outputs/formal_run_100k/`。

## 1. 实验目的与结论

保持当前 3A1C、3D 环境、21D observation、24D global state、奖励、规则逃逸机、
动力学、网络结构及固定初始化，检验当前 MAPPO 从零训练 100000 步后能否可靠拦截。

**完整训练已完成，但尚不能认为学会了可靠拦截或有效三机协同。**
训练随机采样过程中出现 26/206 次成功，最近 50 回合捕获率最高 32%、末尾 18%；
final 和 best 的确定性评估各实际运行 50 回合，均为 0/50 次捕获。
相对随机策略，模型的平均回报和部分距离指标改善，但捕获率与超时率没有改善。
这些结论来自本次唯一一次正式训练，未筛选多个训练配置或重新训练以挑选结果。

## 2. 训练前检查、代码版本与复现记录

- 训练前完整执行 pytest：**52 passed in 12.22s**，无失败。
- 命令：`.venv/Scripts/python.exe -m pytest -q`，设置
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`、`MPLBACKEND=Agg`。
- 当前 Git HEAD：`23bf55576cca27944b4ee2ae5c53a70295c7dac3`。
- `git_dirty=true`：本轮开始时工作区已有尚未提交的 MAPPO 实现，因此不能仅凭 HEAD
  还原本次代码。`source_snapshot/` 保存实际运行的 31 个源代码、测试、配置和文档文件；
  `preflight.json` 保存逐文件 SHA-256。运行后核验这 31 个文件均未变化。
- 未发现需要修复的明显程序错误，本轮没有改动任何 baseline 源码或配置。
- 唯一正式训练从新建网络开始，无 checkpoint 加载或恢复训练。
  pytest 内部既有 16 步 smoke test 仅用于测试，不作为正式实验或模型选择依据。
- Python `3.8.8`、PyTorch `2.4.1+cpu`、NumPy `1.24.4`、matplotlib `3.7.5`、
  PyYAML `6.0.3`、pytest `8.3.5`；CPU，完全隔离的 `.venv`。
- 本轮未执行 `git add`、`git commit`、`git merge` 或 `git push`。

主要复现文件：`preflight.json`、`training_config.yaml`、`environment_config.yaml`、
`metadata.json`、`training_console.log`、`postflight_audit.json`。
保存的 `training_config.yaml` 是有效参数记录；其中原环境引用仍相对原始配置目录，
若以后复制它重跑，需要显式正确引用同目录保存的 `environment_config.yaml`。

## 3. 使用的环境配置

来源：`configs/baseline_3a1c.yaml`，完整快照已保存。

| 项目 | 实际值 |
| --- | --- |
| 场景 | 三维、3 pursuers + 1 rule-based evader，无障碍 |
| 空间下界 / 上界 | `[-100,-100,0]` / `[100,100,100]` m |
| dt / 回合上限 | 0.1 s / 500 步，即 50 s |
| 初始 pursuer 位置 | `[-30,-10,50]`、`[-30,0,50]`、`[-30,10,50]` m |
| 初始 evader 位置 | `[0,0,50]` m |
| 初始速度 / 加速度历史 | 四机均为零，固定初始化 |
| 双方最大水平速度 / 爬升 / 下降 | 20 / 8 / 6 m/s |
| 双方最大水平 / 垂直加速度 | 3 / 2 m/s² |
| 双方最大 jerk | 5 m/s³，三维指令变化模长限制 |
| 逃逸机策略 | 原有逆距离平方排斥合力，`distance_epsilon=1e-6` |
| 捕获 / pursuer 安全距离 | 严格距离 `<5` m 捕获；安全距离 2 m |
| 团队 reward | 捕获 +100；最近距离差系数 1；每步 -0.01；任意不安全 pursuer 对每步共扣 -5 |
| 边界处理 | 原有位置裁剪，保留速度，无额外边界奖励或终止 |

Actor 使用原始 21D 局部观测，Critic 使用原始 24D 全局状态。
本轮没有改变上述定义，也没有新增研究功能。

## 4. 实际 MAPPO 参数

原始来源：`configs/mappo_baseline.yaml`。全部有效配置如下。
训练步数 CLI 值与 YAML 的 100000 相同；仅按用户要求覆盖输出目录。
独立评估回合数另外显式从默认 10 覆盖为每模型 50，不影响训练参数。

```yaml
env_config: baseline_3a1c.yaml
seed: 0
device: cpu
torch_num_threads: 1
output_dir: outputs/formal_run_100k
training:
  total_env_steps: 100000
  rollout_steps: 512
  log_interval_updates: 1
  checkpoint_interval_updates: 10
  capture_rate_window: 50
network:
  actor_hidden_sizes: [128, 128]
  critic_hidden_sizes: [256, 256]
  initial_log_std: -0.5
  min_log_std: -5.0
  max_log_std: 2.0
  hidden_gain: 1.4142135623730951
  actor_output_gain: 0.01
  critic_output_gain: 1.0
ppo:
  actor_learning_rate: 0.0003
  critic_learning_rate: 0.001
  adam_epsilon: 0.00001
  gamma: 0.99
  gae_lambda: 0.95
  clip_ratio: 0.2
  update_epochs: 4
  minibatch_size: 64
  entropy_coefficient: 0.01
  value_loss_coefficient: 0.5
  max_grad_norm: 0.5
  advantage_normalization: true
  normalization_epsilon: 0.00000001
normalization:
  observation: false
  global_state: false
  reward_scale: 1.0
evaluation:
  episodes: 10
  seed: 1000
visualization:
  seed: 1000
  dpi: 160
  figure_size: [10, 8]
```

网络沿用隐藏层 Tanh：共享 Actor `21→128→128→3D mean`，共享可学习 log std，
动作 `tanh(Gaussian sample)`；Critic `24→256→256→1`。
每批 64 个团队时间步对应 192 个 Actor 样本及 64 个 Critic 样本。
没有开启输入缩放、reward normalization、随机初始化或课程学习。

## 5. 训练执行、计数和 checkpoint

实际执行且进程正常退出：

```powershell
.\.venv\Scripts\python.exe -u -m scripts.train_mappo --config configs/mappo_baseline.yaml --total-steps 100000 --output-dir outputs/formal_run_100k
```

| 项目 | 实测 |
| --- | --- |
| 总环境步数 | 100000 |
| PPO 更新数 | 196，最后一个 rollout 为 160 步 |
| 完整回合 | 206，覆盖 99932 步 |
| 最后未完成回合 | 68 步，保存在 checkpoint，不计入完整回合统计 |
| 训练捕获 | 26 / 206 = 12.62% |
| 全部成功回合平均捕获时间 | 38.2 s，范围 26.4–47.1 s |
| 定期 checkpoint | 19 个，update 10、20、…、190 |
| final.pt | step 100000 / update 196 / 206 完整回合 |
| best.pt | step 77312 / update 151 / 159 完整回合 |

best 按原实现的最近训练窗口捕获率、再平均回报选择，选择分数为
`(0.32, -101.927743)`。它不是独立确定性评估最优模型。
两份模型参数均通过有限值检查，两个 optimizer 状态均已保存。
全部训练 CSV 的实填数值均有限；失败回合捕获时间留空是预期行为。
配置与环境快照逐项匹配 preflight，未发生静默参数变化。

## 6. 全过程训练趋势

以下按**回合结束时的累计环境步数**分桶，均为完整回合统计。
`±` 是总体标准差，仅描述本次运行，不能视为独立多 seed 的不确定性区间。

| 回合结束步数区间 | 回合数 | 捕获率 | reward 均值 ± std | 平均回合步数 | 平均回合最小距离 m | 平均终态最近距离 m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| (0,25000] | 50 | 0% | -319.900 ± 333.427 | 500.000 | 30.000 | 176.500 |
| (25000,50000] | 50 | 4% | -636.656 ± 368.050 | 496.400 | 28.968 | 109.692 |
| (50000,75000] | 53 | 22.64% | -124.500 ± 267.336 | 469.906 | 20.519 | 83.197 |
| (75000,100000] | 53 | 22.64% | -139.896 ± 303.047 | 475.604 | 18.733 | 67.499 |

分桶按结束点归属，因此桶内完整回合的长度总和不一定恰为 25000，不能将该分桶的
碰撞统计误解为严格对应那段物理时间的逐步统计。

- **Reward 非单调。** 25–50k 阶段明显退化，50k 后回升，最后一段略有回落。
  前 20 / 后 20 个完整回合均值为 -405.503 / -180.194，后期改善伴随很大波动。
  不能只拿最差阶段或最好的窗口声称稳定提升；首回合 -73.541、末回合 -87.068
  也说明单回合首末比较不足以代表整个趋势。
- **首次捕获较晚。** 第 98 回合、48849 步首次成功。最近 50 回合捕获率在约
  77.3k–78.8k 的回合结束点达到 32%，最后降至 18%（9/50）。
- **回合长度只有有限缩短。** 后半程出现较早捕获，但大多数回合仍超时；
  不能把平均长度小幅下降视为任务完成。
- **距离有改善。** 回合最小距离的阶段均值从 30 降到 18.733 m，终态最近距离
  从 176.500 降到 67.499 m。最小距离包含初始状态，早期恒为 30 m 意味着未比
  初始最近距离更近，而不是在 30 m 处稳定追踪。

Reward 可以由未改动的公式精确分解：

`R_episode = 100*captured + (30-final_nearest_distance) - 0.01*length - 5*collision_steps`。

四阶段平均碰撞奖励依次为 **-168.4、-556.0、-89.245、-120.283**。
因此中期低谷主要伴随碰撞惩罚增加，后期回报改善既包含更近终态和捕获，也包含
碰撞减少。按 CSV 重构总回报的最大误差约 `3.8e-12`，未修改或拟合 reward。

全部完整回合总计 9465/99932 个碰撞步，92/206 回合发生过碰撞。
详细数据在 `training_analysis.json` 与 `training_phase_summary.csv`。

## 7. Actor loss、value loss、entropy 与稳定性

| 指标 | 0–25k 均值 | 25–50k 均值 | 50–75k 均值 | 75–100k 均值 |
| --- | ---: | ---: | ---: | ---: |
| Actor policy loss | -0.007564 | -0.009540 | -0.010821 | -0.011167 |
| Value loss | 76.084 | 129.299 | 48.124 | 47.659 |
| Entropy，nats | 1.774889 | 1.703827 | 1.748250 | 1.689507 |

更新按 update 结束的环境步数分桶。Value loss 是乘配置系数前的
`0.5*mean((V-return)^2)`；Actor loss 是不含 entropy bonus 的 PPO surrogate。

- 没有出现 NaN/Inf、loss 数值爆炸或训练异常退出，但 **value loss 持续有尖峰**，
  全程范围约 1.092–582.982，不能称值函数稳定收敛。
- Entropy 前 20 / 后 20 次更新均值为 1.785590 / 1.662020，最终为 1.623737。
  它中途回升后再下降，没有骤然趋零的证据。
- 近似 KL 全程 0.002228–0.009714；clip fraction 0.018392–0.132975，
  没有明显单次策略更新失控的迹象。
- CSV 的梯度范数是**裁剪前**范数。Critic 最大 3483.962（update 193），
  最终 2379.552；实际裁剪阈值仍为配置的 0.5。
  大梯度反映学习压力和回报/目标波动，不能将这些数直接当作裁剪后梯度。

应区分“运行数值稳定”和“策略性能稳定”：前者通过，后者未达到。
中途有明显行为退化和回报大幅波动；尚无动作方差归零式坍塌证据，也不能仅凭
entropy 与 KL 排除行为层面的失败模式。

## 8. Final 模型正式评估

实际执行：

```powershell
.\.venv\Scripts\python.exe -u -m scripts.evaluate_mappo --checkpoint outputs/formal_run_100k/checkpoints/final.pt --episodes 50 --output-dir outputs/formal_run_100k/evaluation/final
```

50 回合、seeds 1000–1049、确定性动作，0 次捕获，全部在 500 步超时。
平均回报 -81.026314，平均回合最小距离 21.419558 m，平均终态最近距离 106.026314 m。
无成功捕获时间（JSON `null`），无 pursuer 碰撞。

结果：`evaluation/final/evaluation_metrics.csv`、`evaluation_summary.json`，
补充终态距离、超时率及一致性检查的 `evaluation_summary_extended.json`。

## 9. Best 模型正式评估

实际执行：

```powershell
.\.venv\Scripts\python.exe -u -m scripts.evaluate_mappo --checkpoint outputs/formal_run_100k/checkpoints/best.pt --episodes 50 --output-dir outputs/formal_run_100k/evaluation/best
```

50 回合、seeds 1000–1049、确定性动作，0 次捕获，全部在 500 步超时。
平均回报 -84.362816，平均回合最小距离 30 m，平均终态最近距离 74.362816 m。
无成功捕获时间；每回合有 7 个碰撞步，涉及 P0/P2。

两个确定性模型在各自 50 次运行中均得到完全相同的逐回合指标；各自独立评估和
绘图保存的所有数值轨迹数组也逐元素一致。固定初始化、无噪声、确定性策略下，
改变 seed 不产生新场景。**这 50 次是重复运行验证，不是 50 个独立泛化测试场景。**

## 10. Random baseline 对照

实际运行 50 回合、25000 步，seeds 同为 1000–1049。使用已有
`env.sample_actions()`：每架、每轴独立均匀采样 `[-1,1]`，环境与初始化均相同。
种子只改变随机动作。完整结果保存于 `random/`，没有筛选种子。

| 指标 | final deterministic | best deterministic | random |
| --- | ---: | ---: | ---: |
| 实际评估回合数 | 50 | 50 | 50 |
| 捕获率 | 0% | 0% | 0% |
| 平均团队回报 | -81.026314 | -84.362816 | -137.076373 |
| 回报总体 std | 0，重复轨迹 | 0，重复轨迹 | 54.360367 |
| 平均回合步数 | 500 | 500 | 500 |
| 超时率 | 100% | 100% | 100% |
| 平均成功捕获时间 | null | null | null |
| 平均回合最小敌我距离 m | 21.419558 | 30.000000 | 30.000000 |
| 平均终态最近敌我距离 m | 106.026314 | 74.362816 | 159.176373 |
| 发生碰撞的回合比例 | 0% | 100% | 6% |
| 碰撞步数 / 总步数 | 0 / 25000 | 350 / 25000 | 29 / 25000 |
| 碰撞步比例 | 0% | 1.4% | 0.116% |
| 平均 pursuer 两两距离 m | 72.592631 | 119.538467 | 81.591222 |

表中的回报 std=0 表示逐回合指标相同；计算均值再计算 std 可能留下约 `1e-14`
的浮点残差，不是场景变化。所有回报均只累计一份共享团队 reward。

**比较结论：回报与终态距离指标优于本次 random 对照；final 还有最小距离改善，
但任务捕获率与超时率没有优势，因此不能认定已明显优于 random 地完成了任务。**
best 的终态距离比 final 小，但有碰撞惩罚，所以回报反而略差，不能仅按一个距离
指标判断模型整体好坏。对照没有多训练 seed、独立场景或置信区间，不能扩展为
论文级性能或泛化结论。

汇总已保存至 `comparison.csv` 与 `comparison.json`。

## 11. 训练曲线与 3D 轨迹文件

`plots/` 下共 8 张 PNG：

- `reward_vs_training_steps.png`
- `capture_rate_vs_training_steps.png`
- `episode_length_vs_training_steps.png`
- `min_distance_vs_training_steps.png`
- `actor_value_loss_vs_update.png`
- `entropy_vs_update.png`
- `ppo_diagnostics_vs_update.png`
- `training_overview.png`

图保留全部原始数据。滑动均值明确标注为完整 **10 回合**或 **10 更新**窗口；
capture rate 使用 CSV 原有最多 **50 个完整回合**窗口，不再次平滑。
min distance 图同时显示终态最近距离，以免初始距离形成的平线被误读。

![真实 100k 训练总览](../outputs/formal_run_100k/plots/training_overview.png)

分别实际调用 `scripts.visualize_mappo` 加载 final/best，输出目录与相应评估相同。
两张图均已视觉检查，标明 P0/P1/P2/evader、start/end、米制坐标轴、图例、
captured=False 与 500 steps。`trajectory.npz` 保存初始状态及全部 500 步轨迹。

![Final 3D trajectory](../outputs/formal_run_100k/evaluation/final/trajectory_3d.png)

![Best 3D trajectory](../outputs/formal_run_100k/evaluation/best/trajectory_3d.png)

补充的完整回合视频（直接从上述已保存轨迹渲染，没有重跑或修改仿真）：

- [Final model 完整视频](../outputs/formal_run_100k/videos/final_full_episode.mp4)
- [Best model 完整视频](../outputs/formal_run_100k/videos/best_full_episode.mp4)

两份均为 1280×720 H.264 MP4、10 fps、1 倍速，保留原始全部 501 个状态帧，
覆盖 0–50 s 的 500 步仿真；因包含初始帧，文件播放时长为 50.1 s。
画面同步显示三维轨迹、仿真时间、步数、三机敌我距离、捕获阈值与累计团队回报。
视频生成脚本和来源 SHA-256 记录分别位于 `analysis_scripts/render_episode_video.py`
及 `videos/*_video_manifest.json`。

## 12. 轨迹主要行为与单机差异

两模型前段都向逃逸机初始方向运动，随后在边界附近分散、升降或偏离。
逃逸机两条确定性轨迹均于第 86 步首次触边，83% 的 step 后状态在边界，
最终位置均为 `[100,-100,0]` m。追击机未持续跟上其三维位置。

| 模型 / agent | 全回合个人最小敌我距离 m | 终态距离 m | 成为最近追击机的时间份额 | 贴边状态占比 |
| --- | ---: | ---: | ---: | ---: |
| final P0 | 31.623 | 209.392 | 0% | 62.0% |
| final P1 | 27.181 | 106.026 | 17.8% | 71.8% |
| final P2 | 21.420 | 114.790 | 82.2% | 52.4% |
| best P0 | 31.623 | 90.400 | 65.2% | 65.4% |
| best P1 | 30.000 | 282.843 | 2.8% | 60.4% |
| best P2 | 31.622 | 74.363 | 32.0% | 72.0% |

最近追击机份额统计 step 后状态，距离相同则平分份额；它是几何描述，
不是通过移除某机或反事实实验得到的因果贡献。
final P2 在 12.4 s 时达到全队最小 21.42 m，P1 在 13.4 s 达到自身最小距离；
P0 没有比自身初始距离更近。best P1 明显脱离目标，平均速度朝向目标的 cosine
约 -0.516，但边界保留速度使该速度指标需要结合实际位移解读。

## 13. 是否存在协同与“一机主追”

final 存在 P2 主追倾向，P0/P1 的持续有效接近有限；best P0/P2 更常成为最近机，
P1 明显脱离。但 final 的终态最近机反而是 P1，说明整个回合并非固定单一角色。
这不足以证明其他追击机没有通过势场影响逃逸机，也不能从 nearest 份额直接
推断谁对成功贡献最大。

两条确定性轨迹没有显示有效夹击、包围或稳定互补拦截；没有成功捕获来支撑
“已形成有效协同”的结论。训练阶段未保存每个成功回合的完整轨迹，因此不能
将最终失败轨迹直接推广到那 26 个训练成功回合的具体行为。

## 14. 贴边、动作饱和与策略坍塌诊断

- **明显贴边。** pursuer 贴边状态比例约 52%–72%，evader 为 83%。既有边界
  只裁剪位置、保留向外速度，所以较大的模拟速度不一定对应实际位移。
  不能把贴边时的非零速度当作仍在有效追击。
- **未观察到确定性动作端点饱和。** 以 `abs(action_component)>=0.95` 为诊断
  阈值，两模型每架机均为 0%。该阈值只用于分析，没有改变训练或环境。
- **没有探索方差趋零证据。** final 的 Gaussian std 约 `[0.659,0.629,0.620]`，
  best 约 `[0.658,0.637,0.615]`。这些是压缩前标准差，不能当作最终动作 std。
- **隐藏层饱和是另一种现象。** 在已保存确定性轨迹上，用 `abs(Tanh activation)>=0.95`
  诊断首层隐藏激活饱和，final 约 90.30%、best 约 92.97%；第二层约 18.11% / 16.82%。
  这与动作端点饱和不同，提示原始输入尺度可能影响学习，但尚无消融或因果证明。
- saved state 重建的确定性动作与轨迹最大误差低于 `8e-7`，支持诊断所用模型
  与保存轨迹对应。所有轨迹数组及模型张量均有限。

本次证据更符合“性能不稳定、边界失败与角色失衡”，不支持“由于 entropy
归零而完全坍塌”的具体判断。诊断 JSON、每机 CSV、距离与动作图位于
`behavior_analysis/{final,best}/`。

## 15. 是否学会任务，以及是否应立即改 baseline

**尚未学会可靠的当前任务。** 后半程训练成功和距离改善表明并非全程毫无学习迹象，
但两份模型的确定性执行均无法捕获，且末尾训练窗口从峰值回落。
不能将 12.62% 的跨时间随机训练成功率、18% 的末尾窗口，或 50 次重复的
确定性评估混为同一种模型能力指标。

当前不建议直接重设 reward、修改网络或提高/降低环境难度。
应先解释训练采样成功与确定性执行失败的差异，以及边界和输入尺度相关诊断。
本次结果保留为后续对照，没有因结果不理想调整参数或追加其他配置重训。

## 16. 未学会的可能原因与证据边界

以下是待验证假设，不是已确认的根因：

1. **随机采样与均值策略轨迹不同。** 非零 std 下偶发动作序列可以到达均值策略
   未到达的状态；且训练成功率汇集了不同时间的策略，不能当作 final 的
   stochastic 成功率。需要对冻结 checkpoint 另做匹配协议验证。
2. **原始数值尺度及值函数学习压力。** 首层 Tanh 大量饱和、value loss 尖峰与很大的
   裁剪前梯度值得检查，但本轮没有打开归一化或改变任何系数。
3. **边界裁剪影响控制含义。** 贴边比例高、模拟速度与几何位移不一致，可能形成
   与真实追击不同的策略激励或状态混淆；没有擅自修改边界规则。
4. **既定状态并非完整 Markov 信息。** jerk 历史未包含在 21D/24D 输入内，
   边界位置也未直接进入局部 observation；这些属于保持不变的研究设定。
5. **单 seed 和有限训练预算。** 100k 步不能证明方法在所有种子下失败，也不能
   仅因后期部分改善就断言追加步数必然成功。当前双方能力相同、逃逸机初始领先，
   但有限空间也可能创造捕获机会，不能未经验证称任务不可解。

本轮未为验证这些假设改动代码或训练设置。

## 17. 下一步最值得验证的三个问题，按优先级

1. **冻结 final/best 后，随机采样执行是否仍能捕获？成功依赖怎样的轨迹？**
   优先匹配同一环境、相同 checkpoint，比较 deterministic 与 stochastic 执行，
   并保存成功/失败的逐步动作、状态和 reward components，区分真实能力、探索偶然性
   与训练窗口统计差异。该后续实验本轮未执行。
2. **原始输入尺度和边界裁剪分别怎样影响控制与梯度？**
   先核查已有激活饱和、裁剪前梯度、贴边时指令/实际位移的关系，再考虑经批准的
   单变量验证；不要同时重设 reward、observation 和动力学而失去对照。
3. **学到的行为能否跨训练 seed 复现，是否确有多机互补贡献？**
   保留本次固定协议做重复实验与轨迹分析，之后用明确授权的单机贡献对照区分
   “一机主追”、势场间接影响和有效协同；nearest 份额本身不能回答因果问题。

## 结果文件索引

- 训练原始数据：`outputs/formal_run_100k/training_metrics.csv`、`updates.csv`。
- 配置、来源与核验：同目录 `preflight.json`、`metadata.json`、`postflight_audit.json`、
  `source_snapshot/`、`training_console.log`。
- 全过程统计：`training_analysis.json`、`training_phase_summary.csv`。
- 曲线：`plots/`，包含原始数据和明确标注的滑动均值。
- final / best 评估及 3D 轨迹：`evaluation/final/`、`evaluation/best/`。
- random 对照：`random/`。
- 策略对比：`comparison.csv`、`comparison.json`。
- 行为证据：`behavior_analysis/`。
- 所有分析与审计脚本：`analysis_scripts/`；均在被 Git 忽略的实验输出目录中。

本轮只新增此报告和实验输出，没有修改研究设定或 baseline 源码。
