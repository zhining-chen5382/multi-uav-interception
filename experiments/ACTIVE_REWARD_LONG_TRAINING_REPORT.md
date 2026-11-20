# Active Reward 长训练：100k 与条件扩展至 200k

路径说明（2026-11-08）：本文链接及命令示例已更新为版本目录；冻结快照、原始日志、metadata 与权重内嵌旧路径保留，见 [目录迁移表](output_path_migration_20261108.json)。

日期：2026-10-25（Asia/Shanghai）。输出目录：`outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/`。

**最终分类：B，有限且未保持的改善；尚未学会稳定拦截。** 一次从零训练完整完成 100k，因末期多窗口最近距离恢复与初段方向纠正，按本轮条件沿同一次训练续至 200k。全程 710 个完整回合出现 6 次训练采样拦截，但 200k final/best 各 100 回合确定性评估均为 0%；最后 48199 步没有再出现训练成功。不能说完全没有学习，也不能将稀疏成功解释为稳定掌握任务。

## 1. 实验问题、冻结范围与版本

本轮检验：当前 Direct Attacker 点防御任务、active-progress reward 和 21D observation，在不改变任何研究设置时，延长训练是否能够纠正此前的侧偏、过冲和错失后继续外飞。

Git HEAD 为 `c6fe7e7c64e62425c079960b3895615b6f274113`。开始本轮时工作区已有上一轮未提交改动，不能只靠 commit hash 重建实际 baseline。已冻结 43 个源码、配置、测试和文档文件，逐文件 SHA256、开始时 Git status、全部原始/实际参数见 [preflight/freeze_manifest.json](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/preflight/freeze_manifest.json)，文件副本见 [source_snapshot](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/source_snapshot/)。结束时这 43 个文件及快照哈希全部一致。

本轮未修改 reward、observation/state、Direct Attacker、MAPPO 算法/网络/初始化、学习率、熵系数、PPO clip、rollout、epochs、minibatch、normalization、动力学、边界、物理限制、任务半径或固定初态；没有搜索其他配置、重跑挑优或 Git 写操作。仅新增本报告与输出目录内的训练包装、记录、评价和分析产物。README/SPEC 的既有 20k 文字保留为上一轮记录，本轮预算由此次用户要求与实际运行记录明确。

## 2. 实际配置

完整参数见 [environment_config.yaml](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/environment_config.yaml)、[training_config.yaml](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/training_config.yaml)；100k 时的参数单独保存在 [at_100k](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/at_100k/)。实际配置的唯一预算变化是先 100000、条件满足后 200000 total steps，输出目录始终为本次目录。

| 项目 | 冻结值 |
| --- | --- |
| 场景 / 输入 | 3 pursuers + 1 Direct Attacker，3D；actor 21D / critic 24D |
| Base / detection / attack / capture | `[0,0,0]` / 500m / 10m / 5m |
| P0/P1/P2 初始位置 | `[1.5,0,1]` / `[-0.75,1.25,1]` / `[-0.75,-1.25,1]` m |
| Attacker 初始位置 | `[497.49371855331,0,50]` m |
| 初速度 / 加速度历史 | 四机全零，固定 reset |
| dt / horizon / bounds | 0.1s / 500 steps / 三轴 `[-750,750]` m |
| 双方速度、加速度、jerk | 水平20m/s，爬升/下降8/6m/s，加速度水平/垂直3/2m/s²，jerk5m/s³ |
| Direct guidance | attack_speed20，velocity_gain1，epsilon1e-6，无规避/预测/soft steering |
| Reward | active 系数1，survival -0.2/s，capture+100，attack failure-100，collision-5 |
| Seed / device / threads | 0 / CPU / 1 |
| Actor / critic | `[128,128]` / `[256,256]` Tanh，原正交初始化与 log_std 设置 |
| Actor lr / critic lr / Adam epsilon | 0.0003 / 0.001 / 0.00001 |
| Gamma / GAE lambda / PPO clip | 0.99 / 0.95 / 0.2 |
| Rollout / epochs / minibatch | 512 / 4 / 64 team timesteps |
| Entropy / value coefficient / gradient clip | 0.01 / 0.5 / 0.5 |
| Observation/state normalization / reward scale | false / false / 1.0 |
| 原训练日志、checkpoint 间隔 / 统计窗口 | 1 update / 10 updates / 50 episodes |

Reward 保持 `active=(d_PE_prev-d_PE_now)-(d_EB_prev-d_EB_now)`，survival 每步为 `-0.2*dt`；只加权 reward 项求和，不重复加 raw 距离诊断。团队 reward 累计一次，不乘三。Capture、attack、timeout 的原判定与优先级不变。

## 3. 训练前实际检查

实际按顺序完成 pytest → Direct Attacker sanity → stationary/random reward sanity，全部通过后才开始正式训练：

| 检查 | 结果 |
| --- | --- |
| Full pytest | **231 passed in 13.28s**，关闭本机第三方插件自动加载 |
| Direct Attacker sanity | 500m 出发，282步/28.2s 进入10m攻击球，终点8.261147m |
| Stationary pursuers | 回报 **-105.703048**；active -0.063048、survival -5.64、terminal -100、collision0 |
| Random，10 episodes，seeds0–9 | 平均 **-183.347092**，均敌方攻击成功，失败回报全负，范围[-276.048882,-106.292168] |
| 数值与边界 | 12份 sanity NPZ 数值有限，无 NaN/Inf；攻击机无 hard clipping |

明细、日志、完整原始轨迹见 [preflight_results.json](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/preflight/preflight_results.json)、[Direct Attacker sanity](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/preflight/direct_attacker_sanity/)、[reward sanity](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/preflight/reward_sanity/)。**本轮 camping reference 使用本次实测 -105.703048，不是手工假定 -105.7。**

## 4. 100k 正式训练与精确阶段快照

从零训练实际完成 **100000 steps、196 updates、354 个完整回合**；完整回合共 99828 步，最后 172 步为未完回合，未伪装为完整 episode。没有加载旧模型。此时训练 0/354 拦截、354/354 敌方攻击成功、无 timeout。

现有 512 步 rollout 与 10k 等阶段不整除。本轮没有缩短 rollout 来对齐，而在输出目录的原 trainer 包装中只额外保存阶段快照；去掉唯一新增的 save 条件后，原采样/GAE/PPO 循环 AST 相同。每次额外保存均核验 Python/NumPy/Torch RNG 没有变化。原源码、原 best 选择和原周期 checkpoint 均不改。

| 阶段环境步 | 该快照权重最近完成更新至 | 已完成 updates | 口径 |
| ---: | ---: | ---: | --- |
| 10000 | 9728 | 19 | rollout 中途评价快照 |
| 20000 | 19968 | 39 | 同上 |
| 40000 | 39936 | 78 | 同上 |
| 60000 | 59904 | 117 | 同上 |
| 80000 | 79872 | 156 | 同上 |
| 100000 | 100000 | 196 | post-update final 的字节副本 |
| 120000 | 119968 | 235 | 续训 rollout 中途评价快照 |
| 150000 | 149664 | 293 | 同上 |
| 200000 | 200000 | 392 | post-update final 的字节副本 |

中途快照明确标记为评价用途，不声称能从该半个 rollout 精确 resume。100k/200k 的 final 边界均完成原最后短 rollout 的 PPO 更新后保存。见 [stage_manifest.json](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/stage_manifest.json) 和 [instrumentation static audit](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/analysis_scripts/instrumentation_static_audit.json)。

本次前39个 updates（至19968步）及前70个完整回合（至19740步），与上一轮相同 seed/config 的20k实验逐字段完全一致，见 [prefix_consistency.json](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/prefix_consistency.json)。旧20k最后额外32步的更新与本轮持续采512步不同，因此不把旧20k final 当作本轮20k阶段的同一个模型。

## 5. 为什么有条件延长至 200k

100k 没有达到稳定拦截，但末期出现了不应忽略的几何恢复：

- 相邻两个50回合窗口的平均最小距离从 **118.788959m 降至56.675318m**；末30/10回合分别为37.966232/32.954454m。
- 100k确定性阶段最近距离23.414493m，相比80k的113.845033m更近，初段更贴近进攻航线。
- Gaussian std 仍约 `[0.6134,0.5700,0.6049]`，没有探索方差耗尽证据。

反证同样保留：同期50回合回报 -383.036→-389.005，未同步改善；100k仍0%拦截、错失后继续外飞，best仍1024步。单个 checkpoint 的距离改善本身不足以认定任务学习。

综合末约15k的多个训练窗口与方向迹象，100k时暂归 **B：有限几何改善**，按用户给出的持续距离下降/方向纠正条件扩展示例继续。决定在看到后100k数据前写入 [extension_decision_100k.json](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/extension_decision_100k.json)，没有按200k结果回填理由；并未声称 reward 已持续改善或20k失败已证明仅因预算不足。

## 6. 同一训练的续跑与证据保留

继续前，将100k当时的全部模型、CSV、NPZ、图、评价、阶段记录与决策文件复制到 [at_100k](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/at_100k/)，归档240个文件逐一 SHA256 核验通过。

续跑从**本次** `stage_100000.pt` 恢复 actor/critic、优化器、Python/NumPy/Torch RNG、环境状态、加速度、环境 RNG、172步未完回合、累计奖励分量、recent窗口、best score与计数；首次新采样之前全部相等性核验通过，见 [restoration_audit.json](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/restoration_audit.json)。未加载以前20k或其他任务模型，未重跑前100k；CSV追加，无重复表头。

唯一预算覆盖为200000 total steps，没有修改 YAML 中任何研究超参数。原100k最后160步已经做过更新，因此本次是 **100k＋同状态续100k，共392updates**；不是从开始一次性指定200k的391次更新调度。两段 `rollout_steps=512` 均不变，阶段边界的短 rollout 如实保留。输出层续跑代码的采样/GAE/PPO while循环 AST 与原代码一致。

最终实际完成 **200000 steps、392 updates、710 个完整回合＋259步未完回合**。完整回合199741步，另259步未计入 episode。见 [continuation_completed.json](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/continuation_completed.json)、[verification_100k.json](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/verification_100k.json)、[verification_200k.json](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/verification_200k.json)。

## 7. Reward trend 与 camping 对比

训练回报与最小距离按回合结束时累计环境步分段，原始数据没有删去失败或异常差的回合：

| 区间 | 平均 reward | 平均回合最小距离 (m) | 训练成功回合数 |
| --- | ---: | ---: | ---: |
| 0–20k | -261.461 | 84.567 | 0 |
| 20–40k | -374.017 | 74.030 | 0 |
| 40–60k | -343.230 | 52.303 | 0 |
| 60–80k | -357.504 | 99.037 | 0 |
| 80–100k | -388.509 | 74.518 | 0 |
| 100–120k | -432.974 | 35.310 | 1 |
| 120–140k | -336.048 | 26.525 | 4 |
| 140–160k | -300.595 | 42.577 | 1 |
| 160–180k | -328.746 | 54.789 | 0 |
| 180–200k | -324.831 | 82.499 | 0 |

100k末50回合回报 -389.005→200k末50回合 -334.488，有约54.517的改善；其中碰撞项由 -60.9→-10.1，改善50.8，解释约93%的变化。Active项只改善约3.717，同期最小距离56.675→92.915m反而退化。**减少安全距离违规贡献了主要尾窗回报恢复，不能据此称拦截方向持续改善。**

全程平均 reward -344.857；704个失败完整回合均低于 camping -105.703，6个成功回合都超过 camping，但没有任何主要阶段确定性模型超过 camping。原始最优回合 +298.838，最差 -797.038，全部真实保留。

## 8. Minimum distance 与 interception trend

最小距离经历“早期较差→100k末至约140k接近改善→后期回退”，不支持持续单调下降。训练最小值1.749007m出现在成功回合；不能把这个单个最优值作为最终模型能力。200k末50回合均值92.914816m，末段已离最佳训练阶段更远。

训练共 **6/710=0.8451%** 成功；后100k结束的356个回合中6次成功、350次敌方成功，约1.6854%。这里的阶段episode统计含跨100k边界的未完回合，分析另按transition扣除其前172步，保证后100k实际环境步仍恰为100000。

| 成功累计步 | 回合 | 回合长度/时间 | 最近距离 (m) | 当时敌机距基地 (m) | Reward | Collision steps |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 102265 | 363 | 181 / 18.1s | 4.5981 | 210.2611 | +250.4513 | 10 |
| 123047 | 437 | 196 / 19.6s | 1.7490 | 180.2611 | +273.0004 | 0 |
| 125248 | 445 | 227 / 22.7s | 3.3919 | 118.2611 | +58.7375 | 30 |
| 130506 | 464 | 182 / 18.2s | 4.1911 | 208.2611 | +298.8383 | 0 |
| 134099 | 477 | 209 / 20.9s | 3.9512 | 154.2611 | +124.5382 | 24 |
| 151801 | 540 | 218 / 21.8s | 2.9420 | 136.2611 | -32.6327 | 52 |

这些成功发生在攻击球之外，是实际任务判定的拦截，不是把最近接近冒充捕获。最后一次成功后 **48199步没有再成功**，末50回合成功率0。最近50回合训练窗口的峰值为8%（4/50），未形成持续提升。

训练CSV有这些成功回合的指标，没有完整逐步轨迹；因此不凭空给训练成功者身份、三机协同几何或拦截点坐标。阶段与正式确定性评价的完整轨迹均保存，其所有回合未捕获。

## 9. Entropy、loss 与稳定性

Entropy从首update约1.7908变至最后1.2802，范围1.0456–1.8054；不是训练初期迅速降至极低。最终latent Gaussian std约 `[0.5806,0.5713,0.6161]`，best约 `[0.6013,0.5824,0.6050]`，均接近初始exp(-0.5)≈0.6065。**没有早期探索方差塌缩的直接证据**；微分熵受tanh和mean影响，不能仅靠其下降认定分布坍塌。

Actor loss范围-0.023007至-0.001105。Value loss范围7.415–1103.061，稀疏成功出现时有明显尖峰，但没有非有限值或持续数值发散；最后value loss约23.729。Approximate KL范围0.002412–0.013700，clip fraction约1.40%–16.77%。Gradient norms仍是原裁剪前口径，原max_grad_norm=0.5没有改动。

全部模型/optimizer/环境/RNG tensor与数值型CSV有限；奖励分量归和通过，缺少成功时的时间字段留空/null。工程运行稳定，不等于任务性能稳定：本轮有临时改善与后期回退，但不能把这直接诊断为某一种已证实的policy collapse。也没有用“早期熵骤降锁死错误方向”概括本次结果。

## 10. 每阶段完整确定性轨迹

每个stage至少实际运行一个seed1000的完整确定性回合。原六阶段完成后只按hash复用，续跑后新增120k、150k、200k三个回合，没有重选代表轨迹。下列全部282步、Enemy Attack Success、基地终距8.261147m、hard clipping0：

| Stage | 最近机 | 最小距离 (m) | Reward | Reward−camping | Collision steps |
| --- | --- | ---: | ---: | ---: | ---: |
| 10k | P0 | 143.459 | -251.638 | -145.935 | 0 |
| 20k | P0 | **70.982** | -699.866 | -594.163 | 92 |
| 40k | P0 | **77.932** | -835.172 | -729.469 | 86 |
| 60k | P0 | **76.065** | -338.042 | -232.339 | 0 |
| 80k | P2 | **113.845** | -305.602 | -199.899 | 0 |
| 100k | P0 | **23.414** | -485.564 | -379.861 | 14 |
| 120k | P2 | 10.289 | -430.471 | -324.768 | 0 |
| 150k | P1 | 28.754 | -172.091 | -66.388 | 0 |
| 200k | P1 | 129.966 | -286.783 | -181.080 | 0 |

“最近机”取全回合最小距离对应者，不是成功完成拦截者。快照的权重实际更新时间见第4节；不能将20k中途快照与旧20k最后短rollout更新后的final混为一谈。

主要图与记录：

- [九阶段统一物理尺度完整轨迹](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/trajectories/common_scale_stages_3d.png)。
- [九阶段基地局部对照](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/trajectories/common_base_closeup_stages_3d.png)。
- [100k完整轨迹](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/trajectories/stage_100000_trajectory_3d.png)、[120k近距但未捕获](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/trajectories/stage_120000_trajectory_3d.png)、[200k final完整轨迹](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/trajectories/stage_200000_trajectory_3d.png)。
- [逐阶段CSV](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/stages/stage_summary.csv)、[JSON](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/stages/stage_summary.json)、[行为解释](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/stages/INTERPRETATION.md)、[逐阶段reward时序](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/stages/reward_timing.json)。

## 11. 方向、主动出击与协同

所有阶段初期都有正的自身径向接近速度，追击机实际离开基地；不能说三机从一开始就完全背向敌机。10k–80k反复出现侧向/高度偏差；100k初段更接近进攻线，P0在17.6s最近时误差约 `[-1.72,-20.32,+11.51]m`，但随后继续外飞，终点三机距敌机316.6–383.7m。120k进一步出现10.29m近距接触，仍错失后远离；150k出击减慢，P2约13s才离开基地10m；200k又出现明显侧向和高度分散。

最新best在20.6s由P1达到16.2739m，误差约 `[-1.64,-7.11,+14.54]m`。三机约4.1/4.6/4.7s离开基地10m，路程约250/322/274m；末5s自身径向接近速度均为负，约-17.5/-19.5/-13.6m/s，错失后没有稳定回转。完整图见 [best_200k_trajectory_3d.png](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/trajectories/best_200k_trajectory_3d.png)。

**持续存在的是侧偏、过冲和缺乏有效修正，而不是三架始终使用同一错误向量。** P2有时与P0/P1分开；最近机也随阶段改变。三机都在运动，不存在“两机始终不动”的证据；未观察到明确互补截击或协同。没有队友移除消融，因此最近机身份不等于因果贡献。代表轨迹无边界取巧或输出动作饱和；碰撞/安全距离违规仍是真实成本。

## 12. Final / best 正式评价

100k与200k两阶段均分别实际执行final/best各 **100 episodes**，seeds1000–1099，不复用训练指标。100k数据保留在 [at_100k/evaluation](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/at_100k/evaluation/) 与原 [evaluation](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/evaluation/)；200k结果在 [evaluation_200k](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/evaluation_200k/)。

| 指标 | 100k final | 100k best | 200k final | 200k best |
| --- | ---: | ---: | ---: | ---: |
| Interception success rate | 0% | 0% | **0%** | **0%** |
| Enemy attack success / timeout | 100% / 0% | 100% / 0% | 100% / 0% | 100% / 0% |
| Average reward | -485.5641 | -121.9063 | **-286.7833** | **-510.1615** |
| Average minimum P-attacker distance (m) | 23.4145 | 20.8351 | **129.9661** | **16.2739** |
| Average terminal nearest P-attacker distance (m) | 316.5935 | 22.9357 | 187.8126 | 201.1909 |
| Average terminal attacker-base distance (m) | 8.2611 | 8.2611 | 8.2611 | 8.2611 |
| Average episode length | 282 | 282 | 282 | 282 |
| Successful capture time | N/A | N/A | N/A | N/A |
| Collision steps / episode | 14 | 0 | 0 | 42 |
| Collision episode rate | 100% | 0% | 0% | 100% |
| Collision step rate | 4.9645% | 0% | 0% | 14.8936% |
| Active reward | -309.9241 | -16.2663 | -181.1433 | -194.5215 |
| Survival / collision / terminal | -5.64 / -70 / -100 | -5.64 / 0 / -100 | -5.64 / 0 / -100 | -5.64 / -210 / -100 |

固定初态下，每个模型各100行指标完全重复，这是 **repeated deterministic evaluation，不是generalization**。四组评价均无成功时间/拦截geometry，不用0或最近接近点冒充成功。Latest best使用训练采样窗口选择，不能因其文件名就认为确定性评估回报优于final。

## 13. Best checkpoint 与是否超过 camping

100k时best确实仍为1024步/update2。续训后best已更新至 **135328 steps / update265 / 已完成481回合**，score为 `[0.08,-344.36427544129657]`，对应最近50回合中4次采样成功。没有以本轮确定性结果替换这个选择规则或另挑模型。

最新best确定性最近16.27m，仍未达到严格5m捕获条件，且42碰撞步与后续外飞使回报-510.16。**不能继续声称best停在极早期，也不能因训练窗口8%就称稳定学会。**

所有主要阶段、正式final/best和所有704个训练失败回合都低于camping；只有6个成功采样回合超过该参考。200k final的-286.78与best的-510.16仍未超过-105.70。

## 14. A / B / C 分类及训练不足解释

**归入B：过程有有限几何改善与稀疏成功，改善未保持，最终未学会稳定拦截。** 这不是A：正式确定性成功率仍0、主要模型回报低于camping、后期几何回退。也不严格满足用户C的全部条件：本轮确有方向局部纠正、6次实际成功，best已到135328步，不再停留在极早期。

延长预算确实把“20k/100k无任何训练成功”推进到102k后出现成功，因此不能否认训练时间的作用。但200k末仍不稳定、后48k无成功、确定性评价全失败，**不能再仅以“20k太短、继续加训练就会解决”解释或保证后续成功**。单次seed也不能证明任意更长训练必然无效。

本轮止于已授权的200k，不再加预算、不调reward或其他设置。保留当前设计与完整失败/部分改善记录供下一轮独立验证；不因结果不好立即改reward，也不转去修改observation。

## 15. Reward诊断：有数据支持的候选与边界

本轮不是严格C，因此不直接套用“六项全部满足、必须重做reward”的结论。以下仅是对已保存轨迹的分量分析，未实施任何新reward或新策略：

- 各阶段前5s的active均为正，不能声称active天然惩罚所有初期正确出击。100k前5s active约+17.84、survival-1、collision-70，初期负值主要由安全违规成本贡献。
- 100k到最近点active已约+195.25，错失后回吐约-505.18；best135328到最近点active约+142.40，collision-210，错失后再回吐约-336.92。接近收益、后段过冲和安全成本需分开解释。
- 六次训练成功中有一次reward为-32.63，因52个collision steps扣-260；仍高于camping。安全约束与成功目标的权衡不能只看terminal+100。
- 历史保存的单次成功追踪witness在完全相同物理/init下，按当前reward离线重计为-145.77，低于camping；它160步active全为正、前5s净+33.36，6.3s起安全违规后collision最终-490。这是安全成本/credit的诊断，不是本轮训练成功样本、替代baseline或无碰撞成功策略的最优性证明。

| 待验证问题 | 当前证据与建议边界 |
| --- | --- |
| 初期正确出击是否被负active普遍打压 | 未得到支持；正active、碰撞和时间项应单独看 |
| 为何探索比camping差、是否有负奖励谷底 | 侧偏/过冲、碰撞可吞掉接近收益；不能仅凭负绝对return推导策略梯度必然错误 |
| 是否减弱负active | 可能缓解错失后的大回吐，但也可能削弱回转纠错信号；需要独立对照 |
| 是否clip active | 会改变塑形尺度与望远镜差分性质，不宜仅为抬高曲线裁剪 |
| 是否只奖励positive active | 可能形成往返刷正进展而不承担远离成本的取巧，不能自动采用 |
| 是否调整survival | 282步仅-5.64，量级较小；同长度失败中是共同项，不是当前主导差异 |
| Terminal是否应更主导 | 碰撞/active可达数百，值得分析目标权衡；不等于本轮应自动改±100 |
| 是否换point-defense dense reward | 可先检查错失后回转、进攻线/截击几何的信用分配，再设计独立对照，不能混入角色或包围奖励 |

Gamma=.99、dt=.1使28.2s终局从初态折扣权重约0.0594；即时塑形与延迟终局的时间尺度值得记录，但本轮没有改gamma/GAE。固定282步失败中，扣除的敌机基地进展及时间项在未折扣和折扣口径均为共同项，不改变这些失败轨迹的回报排序；负绝对return本身不是优化失败的充分解释。

若后续决定优先做reward工作，建议按顺序先验证：**①错失后回吐与安全违规的信用分配；②成功终局相对塑形/安全成本的实际权衡；③代理差分是否能区分合理截击与无效接近。** 先用现有分量/轨迹建立证据，再明确单因素方案；不默认实施clipping、positive-only、改survival或terminal权重。完整九项问题、反例与风险见 [REWARD_LANDSCAPE_DIAGNOSTIC.md](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/reward_diagnostics/REWARD_LANDSCAPE_DIAGNOSTIC.md)。

输入尺度、base-relative information和agent symmetry只作为既有疑点记录，本轮未更改、未单独消融，不将其替代为本轮新的研究任务或唯一根因。

## 16. 曲线、原始数据与复现

当前根目录的 [training_metrics.csv](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/training_metrics.csv)、[updates.csv](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/updates.csv)、[analysis.json](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/analysis.json) 和NPZ为完整200k；100k版本全部保留于at_100k。缺失成功时间留空/null，NPZ保留原CSV字段与缺失值口径，无手工填造。

Episode图同时展示raw与 **trailing50 episodes mean**，前49点使用已有数据；update图不平滑。Reward图包括本次实测camping参考线。共21张训练图：

- [Reward vs steps](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/plots/reward_vs_training_steps.png)、[minimum distance](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/plots/min_distance_vs_training_steps.png)。
- [Interception rate](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/plots/interception_rate_vs_training_steps.png)、[enemy attack rate](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/plots/enemy_attack_rate_vs_training_steps.png)、[episode length](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/plots/episode_length_vs_training_steps.png)。
- [Entropy](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/plots/entropy_vs_updates.png)、[actor loss](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/plots/actor_loss_vs_updates.png)、[critic loss](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/plots/critic_loss_vs_updates.png)。
- [Reward分项](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/plots/reward_components_vs_training_steps.png)；其余KL、clip、gradient、collision、camping差图见[plots目录](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/plots/)。

主要实际命令（记录本次执行，不要求覆盖已有目录重跑）：

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:MPLBACKEND = 'Agg'
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m scripts.sanity_direct_attacker --config configs/baseline_3a1c.yaml --output-dir outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/preflight/direct_attacker_sanity
& .\.venv\Scripts\python.exe -m scripts.sanity_reward --config configs/baseline_3a1c.yaml --episodes 10 --seed 0 --output-dir outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/preflight/reward_sanity
& .\.venv\Scripts\python.exe -u outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/analysis_scripts/train_with_stage_checkpoints.py --config configs/mappo_baseline.yaml --output-dir outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000
# 100k分析与条件决策、完整归档后，才续同一次run：
& .\.venv\Scripts\python.exe -u outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/analysis_scripts/resume_same_run_to_200k.py
& .\.venv\Scripts\python.exe -m scripts.evaluate_mappo --checkpoint outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/checkpoints/final.pt --episodes 100 --output-dir outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/evaluation_200k/final
& .\.venv\Scripts\python.exe -m scripts.evaluate_mappo --checkpoint outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/checkpoints/best.pt --episodes 100 --output-dir outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/evaluation_200k/best
```

100k final/best也各实际评价100回合，保留完整metadata与配置；九stage的评价命令、NPZ复用hash及分析脚本都保存在 `analysis_scripts/` 与 `stages/`。没有为了可视化重新挑选训练配置。

## 17. 交付与限制

本轮仅新增此报告及 `outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/` 内产物。43个冻结文件不变、100k归档逐文件哈希通过、50个checkpoint的模型/优化器/环境状态有限；两阶段正式评价各final/best100回合及九stage完整回合齐全，见两份verification JSON。

报告：`experiments/ACTIVE_REWARD_LONG_TRAINING_REPORT.md`。

主要产物：`outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/{at_100k/,preflight/,checkpoints/,plots/,stages/,trajectories/,evaluation/,evaluation_200k/,reward_diagnostics/}`；训练与续训分别保留 `training_console.log`、`continuation_console.log`。

结论限于当前固定初态、单seed与这条两阶段训练轨迹，不能作为泛化或多seed稳定性证据。没有修改任何研究设置来追求成功结论，也没有在200k后继续加训。当前不需要额外信息即可查看与复核全部结果。
