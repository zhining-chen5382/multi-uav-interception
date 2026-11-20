# Active Progress Reward：修复与单次 20k 诊断

路径说明（2026-11-08）：本文链接及命令示例已更新为版本目录；冻结快照、原始日志、metadata 与权重内嵌旧路径保留，见 [目录迁移表](output_path_migration_20261108.json)。

日期：2026-10-25（Asia/Shanghai）。输出目录：`outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/`。

**结论：奖励计算与失败回报符号修复通过；camping 未消失，MAPPO 仍未学会拦截。** 全部 229 项测试通过，stationary 回报 -105.7030，10 回合 random 平均 -183.3471；一次从零 20000 步训练完整完成。Final/best 各 10 回合确定性评估均 0% 拦截、100% 敌方攻击成功。新 best 仍在 1024 步选出，低速基地驻留比旧 best 更久；final 更主动但错向且碰撞多。没有据此调参、继续修改 reward、扩展 observation 或追加训练。

## 1. 范围、代码版本与控制变量

Git HEAD：`c6fe7e7c64e62425c079960b3895615b6f274113`。开始本轮时 Git 工作区干净；本轮修改未提交。初始文件 SHA256 与 Git blob hash 见 [reward_change_baseline_manifest.json](../outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/preflight/reward_change_baseline_manifest.json)。实际训练源码、配置及测试快照共 43 个文件，见 [source_snapshot](../outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/source_snapshot/) 和 [freeze_manifest.json](../outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/freeze_manifest.json)；实验后哈希复核全部未变。未执行 Git add/commit/merge/push。

唯一研究变量是 reward definition。其余环境 YAML 内容逐项相等，`configs/mappo_baseline.yaml` 字节不变（SHA256 `cc2cfee3ebcb9d807f85cd2c07cb836bf4c3637878837b4eb71d36f44d1ae6ec`）；MAPPO、network、buffer、normalization、Direct Attacker、dynamics 等受保护源码哈希均不变。

保持：3A1C、三维、base `[0,0,0]`、detection 500m、attack radius 10m、capture radius 5m、dt 0.1s、horizon 500、数值空间三轴 `[-750,750]`、原固定位置和零初速、原 hard clipping，以及原双方速度/加速度/jerk 限制。Actor 输入仍为 **21D local observation**、critic 输入仍为 **24D global state**。没有随机初始化、预测、包围/角色奖励或 curriculum。

训练仍用 seed 0、CPU/1 thread、rollout 512、actor/critic `[128,128]`/`[256,256]` Tanh、原正交初始化、lr 0.0003/0.001、gamma 0.99、GAE 0.95、PPO clip 0.2、epochs 4、minibatch 64、entropy coefficient 0.01、value coefficient 0.5、max gradient norm 0.5。Observation/global-state normalization 仍为 false，reward_scale 仍为 1。完整有效参数见 [training_config.yaml](../outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/training_config.yaml) 和 [environment_config.yaml](../outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/environment_config.yaml)。仅按本轮要求通过 CLI 指定 20000 步和新输出目录。

## 2. 原 reward 的错位与旧 best 实际行为

旧进展项为 `d_PE_prev - d_PE_now`，无法区分距离缩短来自我方运动还是敌机主动飞向基地。即使我方静止，Direct Attacker 的进攻也会产生大量正 progress，遮蔽 -100 的终局失败信号。

本轮直接读取旧 100k 实验保存的 best 轨迹、metrics 和行为数据，没有重新训练或挑选旧模型。来源见 [旧正式报告](DIRECT_ATTACKER_MAPPO_100K_REPORT.md)、[旧 best 轨迹](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/evaluation/best/first_episode.npz)、[旧 best 3D 图](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/trajectories/best_trajectory_3d.png) 和本轮 [old_baseline_reference.json](../outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/preflight/old_baseline_reference.json)。

旧 best 0% 拦截，282 步后敌方攻击成功，但回报 **+352.666237**，分解如下：

| 旧回合分量 | 数值 |
| --- | ---: |
| 最近敌我距离进展 | +455.486237 |
| 旧时间项 | -2.820000 |
| 碰撞项 | 0 |
| 终局失败 | -100.000000 |
| 合计 | +352.666237 |

同期敌机向基地净推进 491.738853m。旧 best 三机各飞约 51–52m，基地 10m 内驻留 10.4–12.0s，后期仍离开到距基地约 50–54m；准确描述是**慢起步、同向漂移和早期低速驻留**，不是全回合严格静止。前 5s 自身朝敌机的径向速度仅约 0.34–0.35m/s，大部分距离收缩由敌机产生。这个正回报错位确实存在，但一次旧实验不能证明它是 camping 的唯一原因。

对旧保存轨迹只离线换算新公式，active=-36.252615、survival=-5.64、collision=0、terminal=-100，总计 **-141.892615**。这不是新策略表现，也没有改写旧实验数据。另用保存动作进行工程重放，确认历史 schema 的状态/reward 与原数组逐元素一致，新 schema 的状态、观测、终止及 RNG 状态也一致，仅 reward 不同。

## 3. 新公式、参数与计算口径

令 `d_PE` 为三机中最近的三维敌我距离，`d_EB` 为三维敌机-基地距离；前后量取同一步同步积分前和积分后的状态：

```text
pursuer_attacker_progress = d_PE_prev - d_PE_now
attacker_base_progress = d_EB_prev - d_EB_now
active_progress = pursuer_attacker_progress - attacker_base_progress
active_progress_reward = active_progress_coefficient * active_progress
survival = survival_penalty_per_second * dt
team_reward = active_progress_reward + survival + collision + capture + attack_failure
```

实际 YAML：

```yaml
reward:
  capture_reward: 100.0
  attack_failure_penalty: -100.0
  active_progress_coefficient: 1.0
  survival_penalty_per_second: -0.2
  collision_penalty: -5.0
```

默认配置删除 `progress_coefficient`、`time_penalty`，不叠加旧项。存活惩罚是 -0.2/s，当前每个 0.1s transition 为 -0.02，28.2s 合计 -5.64，包含终止步代表的经过时间；明显小于 ±100 的终局项。Capture、enemy attack、collision 的定义与系数不变；同一步 capture 优先，不同时扣失败项。任意我方不安全距离违规每 team step 只扣一次 -5，三份相同 reward 的团队累计只计一次。

新公式按用户指定的代理差分实现，不是在任意几何下对我方自主运动贡献的精确因果分离。保留完整旧 reward schema 仅用于旧 checkpoint/配置复现；新旧 schema 互斥，混搭、缺项或非法数值会报错，当前默认运行只使用新公式。

## 4. 奖励分解诊断

每步 `info["reward_components"]` 返回三项未加权距离诊断和五项已加权 reward：

- 原始诊断：`pursuer_attacker_progress`、`attacker_base_progress`、`active_progress`。
- Reward 项：`active_progress_reward`、`survival`、`collision`、`capture`、`attack_failure`。

**总 reward 只求和五个已加权项，三个原始诊断不参与第二次累加。** `info["episode_reward_components"]` 返回累计已加权分量副本，reset 清零。Episode CSV、evaluation metrics/summary 增加 active、survival、collision、terminal 四个累计贡献及对应均值；terminal 是 capture 与 attack_failure 的和。NPZ 增加 component names 和逐 transition 分量矩阵，保留原位置、速度、动作和 reward 数组。

旧 schema 没有 active/survival 时其新指标为 null，不用伪造的零冒充测量。日志增加不改变 RNG、PPO、GAE、checkpoint 选择或网络。

## 5. Unit tests 与完整验证

完整 pytest：**229 passed in 14.03s**，日志见 [preflight/pytest.log](../outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/preflight/pytest.log)。新增/更新测试覆盖：

| 场景 | 期望与实际 |
| --- | --- |
| PE 100→98，EB 200→198 | active=0，无主动进展正奖励 |
| PE 100→95，EB 200→198 | active=+3 |
| PE 100→99，EB 200→198 | active=-1 |
| survival=-0.2/s，dt=0.1 | 每步 -0.02，其他 dt 成比例 |
| enemy attack / capture | 分别 -100 / +100，capture 优先保持 |
| 多对我方安全距离违规 | 每步团队 collision 只扣一次 -5 |
| Reward 求和 | 排除 raw 诊断，无旧 progress/time 重复项 |
| 累计指标与 reset | 分量归和、副本安全、清零、CSV/NPZ 一致 |
| 新旧配置 | 互斥校验、历史 reward 复现 |
| 原输入与物理行为 | obs `(3,21)`、state `(24,)`，原测试继续通过 |

真实 stationary/random trajectory 从保存状态独立重算两种距离进展与逐步 reward，均吻合。全部 70 个完整训练回合四项累计之和与 return 最大误差约 **1.88e-12**。训练数据、loss、checkpoint 参数/optimizer/保存状态均有限，无 NaN/Inf。完整核验见 [experiment_verification.json](../outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/experiment_verification.json)。

## 6. Stationary pursuers sanity check

使用原固定零初速、零加速度历史和全零 pursuer action。通过真实 `env.step()` 完整运行，三机状态逐步严格不变，没有冻结敌机、移动基地或修改动力学。

| 指标 | 实际结果 |
| --- | ---: |
| 结局 / 长度 | Enemy Attack Success / 282 步、28.2s |
| PE 累计进展 | +491.675804m |
| EB 累计进展 | +491.738853m |
| Active 累计 reward | **-0.063048** |
| Survival | -5.640000 |
| Collision | 0 |
| Terminal | -100.000000 |
| 总 reward | **-105.703048** |
| 回合最小敌我距离 | 6.732431m，未达到 `<5m` |
| Hard clipping | 0 |

Active 接近零的小负残差来自原三机相对基地的三维偏置，不是所有 pursuer 都恰好位于基地点。失败没有再获得数百正回报。数据见 [stationary_metrics.json](../outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/sanity/stationary_metrics.json)、[stationary_episode.npz](../outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/sanity/stationary_episode.npz)。

## 7. Random policy sanity check

本轮实际运行 **10 个完整回合**，seeds 0–9，使用原均匀随机动作，不引用旧 random 统计。

| 指标 | 实际结果 |
| --- | ---: |
| Interception / enemy attack / timeout | 0% / 100% / 0% |
| 平均 reward | **-183.347092** |
| 平均 active reward | -35.207092 |
| 平均 survival | -5.640000 |
| 平均 collision | -42.500000 |
| 平均 terminal | -100.000000 |
| 平均长度 / 平均最小敌我距离 | 282 步 / 29.664037m |
| 失败 reward 范围 | **-276.048882 至 -106.292168** |
| 碰撞步数 / hard clipping | 85 / 0 |

10/10 失败回报均为负，独立奖励重算通过；小样本用于 sanity，不代表泛化或正式随机策略性能结论。逐回合 CSV 和全部 NPZ 位于 [sanity](../outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/sanity/)，汇总见 [reward_sanity_summary.json](../outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/sanity/reward_sanity_summary.json)。Sanity 通过后才启动本轮唯一 20k 训练。

## 8. 20k diagnostic training 完成情况

从零执行 **20000 个环境步、40 updates、70 个完整回合**；完整回合 19740 步，余下 260 步属于未结束回合，没有算作 episode。最后 rollout 为 32 步，精确到达 20000；没有跑新的 100k，也没有加载旧模型或重试其他配置。

70 个完整训练回合全部 enemy attack success、无拦截、无超时，每回合 282 步，终点敌机距基地 8.261147m。失败 return 全负，范围 **-617.974170 至 -115.694347**，平均 -261.461285；平均四项贡献为 active -95.749856、survival -5.64、collision -60.071429、terminal -100。38/70 回合发生我方安全距离违规，共 841 个 collision steps。

Final、best 及 update 10/20/30/40 共 **6 个 checkpoint** 均正常保存并加载检查。Best 仍来自 **1024 steps / update 2 / 仅 3 个完整 episodes**，原选择 score 为 `[0,-202.4691966817107]`，规则仍是“近期训练拦截率优先，再平均回报”，上限 50 回合、早期使用已有回合。它不是按本次 evaluation 重新挑选的模型，也不代表成功策略。

## 9. 整段训练趋势与曲线

| 环境步区间（按回合结束步） | 回合数 | 平均 reward | 平均回合最小距离 (m) | 拦截率 |
| --- | ---: | ---: | ---: | ---: |
| 0–5k | 17 | -223.254 | 46.889 | 0% |
| 5k–10k | 18 | -263.616 | 89.546 | 0% |
| 10k–15k | 18 | -293.456 | 124.407 | 0% |
| 15k–20k | 17 | -263.511 | 74.791 | 0% |

首/中/末 10 回合平均 reward 为 **-219.2967 / -243.9084 / -235.0738**，对应最小距离 **44.0005 / 107.6076 / 57.3448m**。后段有恢复，但没有持续改善或胜过首段。全程最接近的训练回合也仅达到 16.7237m，没有进入 `<5m`。

Actor loss 范围 -0.014263 至 -0.002706。Critic 前/后 10 updates 平均 loss 约 181.364 / 46.079，仍有尖峰，最后一次为 124.118；value 拟合下降没有转化为任务成功。Entropy 前/后 10 updates 平均约 1.7987 / 1.7334，KL 范围 0.001239–0.007500，clip fraction 1.30%–10.37%。没有数值失控或近零方差坍塌证据，但任务表现未学会。记录的 gradient norms 是裁剪前数值，不能用它们判断原有 0.5 裁剪失效。

原始 [training_metrics.csv](../outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/training_metrics.csv)、[updates.csv](../outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/updates.csv) 和无 pickle 的 NPZ 副本均保留。共保存 **20 张曲线**，episode 图同时展示 raw 和 **trailing 10 episodes mean**，前 9 点使用已有数据；原训练日志的统计窗口仍是 50，没有修改超参数。Update 图不平滑。

- [Reward](../outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/plots/reward_vs_training_steps.png)、[分项贡献](../outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/plots/reward_components_vs_training_steps.png)、[最小距离](../outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/plots/min_distance_vs_training_steps.png)。
- [拦截率](../outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/plots/interception_rate_vs_training_steps.png)、[敌方成功率](../outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/plots/enemy_attack_rate_vs_training_steps.png)、[回合长度](../outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/plots/episode_length_vs_training_steps.png)。
- [Actor loss](../outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/plots/actor_loss_vs_updates.png)、[Value loss](../outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/plots/critic_loss_vs_updates.png)、[Entropy](../outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/plots/entropy_vs_updates.png)。

全部曲线及数值、缺失值口径和数据 SHA256 见 [plots](../outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/plots/) 与 [analysis.json](../outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/analysis.json)。

## 10. Final / best 确定性评估

每个模型实际运行默认 **10 episodes**，seeds 1000–1009，使用各 checkpoint 的新奖励环境快照，未覆盖任务参数。每种模型的 10 行指标均相同，是固定初始化的 repeated deterministic performance，不能解释为 10 个独立场景的泛化。

| 指标 | Final | Best |
| --- | ---: | ---: |
| Interception / enemy attack / timeout | 0% / 100% / 0% | 0% / 100% / 0% |
| 平均 reward | **-762.755853** | **-121.906307** |
| 平均长度 | 282 步 | 282 步 |
| 成功拦截时间 | N/A | N/A |
| 平均回合最小敌我距离 | 72.432501m | 20.835127m |
| 终点最近敌我距离 | 158.785236m | 22.935689m |
| 终点敌机距基地 | 8.261147m | 8.261147m |
| Active reward | -152.115853 | -16.266307 |
| Survival | -5.640000 | -5.640000 |
| Collision | **-505.000000** | 0 |
| Terminal | -100.000000 | -100.000000 |
| 每回合 collision steps | **101（35.82%）** | 0 |
| Hard clipping | 0 | 0 |

Final 的大负回报主要包括安全距离违规 101 步带来的 -505；训练使用随机采样动作、确定性评估使用 tanh(mean)，分布不同，不能要求其碰撞率与训练均值相同。它是实际策略问题，不是用改系数来掩盖的工程异常。

详细 CSV、summary、配置、metadata 和首回合完整 NPZ 见 [evaluation/final](../outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/evaluation/final/) 与 [evaluation/best](../outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/evaluation/best/)。两者无成功，所以 interception geometry 与成功时间为 null，不把最近接近点称为拦截点。

## 11. 3D 完整轨迹与行为变化

旧 best 使用已保存的 seed 1000；新 best/final 均取本轮 evaluation 第一回合 seed 1000，没有筛选较好轨迹。三维图标出基地、10m 攻击球、四机起终点、完整路径、结局、步数和距离。

- [旧 best 完整轨迹](../outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/trajectories/before_best_trajectory_3d.png)。
- [新 20k best 完整轨迹](../outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/trajectories/best_trajectory_3d.png)。
- [新 20k final 完整轨迹](../outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/trajectories/final_trajectory_3d.png)。
- [统一物理尺度的前后对照及基地局部视图](../outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/trajectories/before_after_common_scale_3d.png)。

| 逐机指标，顺序 P0/P1/P2 | 旧 best | 新 best | 新 final |
| --- | --- | --- | --- |
| 全回合路程 (m) | 52.30 / 51.14 / 51.24 | 32.11 / 28.92 / 26.14 | 207.82 / 207.72 / 207.85 |
| 基地 10m 内时间 (s) | 10.4 / 12.0 / 11.4 | **13.2 / 14.8 / 14.7** | 3.8 / 3.9 / 4.3 |
| 其中速度 ≤1m/s 时间 (s) | 6.5 / 6.7 / 6.7 | **10.8 / 11.8 / 12.4** | 0.9 / 0.9 / 0.9 |
| 前 5s 自身朝敌机径向速度 (m/s) | 0.349 / 0.347 / 0.342 | **0.162 / 0.161 / 0.156** | 1.948 / 1.927 / 1.957 |
| 终点距基地 (m) | 53.74 / 50.12 / 51.39 | 33.60 / 28.87 / 26.67 | 166.11 / 169.12 / 169.63 |

**Camping 在 best 中没有消失，低速驻留反而更明显。** 三机仍缓慢同向漂移，前 5s 主动径向速度更低，路程更短；新 best 最近接近时主要高度误差约 +20.18m，没有对准敌机轨迹。负 return 不等于学会了主动出击。

Final 三机明显离开基地、前期主动靠近，但在最近接近时仍有约 +53.23m 侧向与 +48.99m 高度偏差；后段均远离敌机，末态高度约 82.9–87.7m，而敌机仅约 0.744m。三机同向聚集、互相过近，没有有效修正或协同截击。

## 12. 一机主追、协同、异常与 reward hacking

Final 的 P0 在全部 post-step 状态中最近；best 的 P0 最近份额约 95.4%，但两种策略三机都在移动。Final 三机路程几乎相同，不能描述成“一机追、两机不动”。更准确是**最近接触由一机主导，团队同向偏离、角色贡献不互补**。共享奖励、平行轨迹或多个非零运动本身不构成协同；本轮没有队友移除消融。

三条代表轨迹 hard clipping 均为 0，动作分量 `abs(a)>0.95` 的比例均为 0；未见边界取巧或动作输出饱和。Final 的 101 个 collision steps 是我方安全距离违规，应和敌我 capture 区分，不能因没有 hard clip 就忽略。

**旧的“enemy attack success 仍获得数百正 return”现象，在本轮 stationary、random、全部完整训练回合和两模型评估中均已消除。** 但 camping 仍存在，不能宣称所有 reward hacking 或奖励目标问题都已消除。

一个有用的对照是：新奖励下 stationary 为 -105.70，仍高于新 best 的 -121.91 和 final 的 -762.76。训练尚未找到成功轨迹时，“不乱飞、不碰撞、留在敌机进攻线附近”的失败策略仍可能优于错向出击。这个事实不能用再次调 reward 来隐去。

## 13. 新公式的数学限制与此次结论

未折扣的 active 回合和会望远镜消去中间距离：

```text
sum(active_progress) = (d_PE_initial - d_PE_terminal)
                       - (d_EB_initial - d_EB_terminal)
```

当前固定初态 `d_PE_initial≈498.408235m`、`d_EB_initial=500m`。敌方成功意味着 `d_EB_terminal<10m` 且 `d_PE_terminal>=5m`，故默认系数 1 下 active 总和 `<3.408235`；加 -100 终局和非正 survival/collision 后，失败回报必负。这个界依赖当前初态、半径、系数，不是任意任务的普遍保证。

更关键的是：**Direct Attacker 不受我方动作影响；所有 282 步失败轨迹中的敌机基地进展时间序列相同。** 相对于旧 reward，本例每个失败回合都扣掉相同的 491.738853 基地进展，以及新旧时间项差额 2.82，共同平移约 -494.558853。因而，在这一组同长度失败轨迹之间，新公式改变 return 的基线和符号，**并不改变它们的整回合回报排序**。它会影响实际有限样本 value/advantage 的优化过程，也让成功与失败的信号更易检查，但没有数学保证它单独就会消除 camping。

目前只能认定本轮指定公式正确实现、旧虚高正回报被消除；不能认定奖励修复已带来任务学习，或通过一次 seed=0 的 20k 对照证明 reward 是唯一失败原因。前后训练预算为 100k/20k，奖励量纲也不同；不得将旧 +352 与新负数直接当作策略好坏的量化改善。当前任务结论仍为 **Not learned**。

## 14. 是否建议下一步修改 observation

**建议下一轮单独开展观测相关对照，但不据本次结果直接替换 baseline 或保证扩维有效。** 当前 actor 缺少显式基地相对信息，且旧报告已发现原始数百米输入造成第一层 Tanh 大量饱和；这分别是信息内容与输入尺度两种假设，应分开验证，不能同时新增基地向量、开启 normalization、改 reward 和调学习率。

优先保留本轮 reward 与其他全部控制变量，单独验证输入尺度/激活问题；随后独立检验 base-relative information 能否改善出击方向、拦截高度与错失后的修正。当前固定案例曾有仅用原相对信息的可行追踪证据，因此不能把缺少基地观测断言为物理不可达或唯一根因。本轮 **没有修改 21D observation、24D state 或任何 normalization 开关**，也没有执行后续实验。

## 15. 文件修改、运行方式与产物

| 文件 | 本轮改动 |
| --- | --- |
| `configs/baseline_3a1c.yaml` | 仅替换 progress/time 的 reward 参数 |
| `envs/rewards.py` | 新增独立 active reward 算术与有限性检查 |
| `envs/pursuit_env.py` | 调用新 reward、累计分量；保留互斥历史 schema |
| `utils/evaluation.py` | 累计分量 metrics、summary 与 NPZ 逐步分解 |
| `scripts/train_mappo.py` | 仅增加分量日志和 partial episode 诊断，无 PPO 改动 |
| `scripts/random_rollout.py` | 增加累计分量输出 |
| `scripts/sanity_reward.py` | 新增 stationary/random 实际运行及独立重算 |
| `tests/test_active_progress_reward.py` | 新增 reward unit/integration tests |
| `tests/conftest.py`、`test_point_defense.py`、`test_mission_evaluation.py` | 更新默认奖励预期与日志测试；保留历史测试含义 |
| `README.md`、`BASELINE_SPEC.md` | 同步定义、单位、兼容性、命令与限制 |
| 本报告 | 记录真实结果与待验证问题 |

主要实际命令：

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:MPLBACKEND = 'Agg'
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m scripts.sanity_reward --config configs/baseline_3a1c.yaml --episodes 10 --seed 0 --output-dir outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/sanity
& .\.venv\Scripts\python.exe -u -m scripts.train_mappo --config configs/mappo_baseline.yaml --total-steps 20000 --output-dir outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000
& .\.venv\Scripts\python.exe -m scripts.evaluate_mappo --checkpoint outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/checkpoints/final.pt --episodes 10 --output-dir outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/evaluation/final
& .\.venv\Scripts\python.exe -m scripts.evaluate_mappo --checkpoint outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/checkpoints/best.pt --episodes 10 --output-dir outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/evaluation/best
```

已有训练目录拒绝覆盖；上述是本轮命令记录，不表示需要重复训练。分析脚本均位于 `outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/analysis_scripts/`，可从保留的 CSV/NPZ 重新生成图表。

主要产物：

- 报告：`experiments/ACTIVE_PROGRESS_REWARD_REPORT.md`。
- Sanity：`outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/sanity/`。
- 训练 CSV/NPZ、console log、配置与 checkpoint：`outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/`。
- 二十张训练曲线：`outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/plots/`。
- 新旧完整轨迹及共同比例对照：`outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/trajectories/`。
- 逐机行为 JSON/CSV、诊断定义和解释：`outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/behavior/`。

目前不需要用户补充信息即可查看与复核这些结果；任何后续观测或研究设定改动另行确定范围。
