# Direct Attacker：MAPPO 100k 正式基线实验

路径说明（2026-11-08）：本文链接及命令示例已更新为版本目录；冻结快照、原始日志、metadata 与权重内嵌旧路径保留，见 [目录迁移表](output_path_migration_20261108.json)。

实验日期：2026-10-25（Asia/Shanghai）。输出目录：`outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/`。

**结论：Not learned。** 单次从零训练完整完成 100000 个环境步；训练期间 354 个完整回合均未拦截。Final 与 best 各实际执行 100 个确定性评估回合，拦截率均为 0%，敌方攻击成功率均为 100%；本次重新运行的 100 回合 random 对照拦截率为 2%。训练回报总体下降，没有证据表明现有 MAPPO 已学会点防御拦截。

## 1. 实验目的与范围

检验现有 3A1C MAPPO 能否在已验证的 point-defense / Direct Attacker 固定任务中，从随机初始化的网络学到有效拦截，并保留失败过程用于分析。

本轮只有一次 seed=0、100k 的正式训练，没有加载旧 checkpoint、复用旧任务模型、调参重试或挑选多个训练配置。环境、Direct Attacker、初始化、21D/24D 接口、reward、物理约束、边界、MAPPO 网络和算法全部冻结。新增内容仅为本报告，以及输出目录内的实验数据、图表、快照和分析脚本。未执行 Git 写操作。

## 2. 代码版本与训练前检查

- Git HEAD：`23bf55576cca27944b4ee2ae5c53a70295c7dac3`。
- 开始实验时工作区已存在未提交改动，因此该 hash 不能单独代表本次实际代码。已将实际使用的 40 个源码、配置、测试和定义文件保存至 [source_snapshot](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/source_snapshot/)，逐文件 SHA256 与初始 Git status 见 [freeze_manifest.json](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/freeze_manifest.json)。训练后复核这 40 个文件字节未变。
- 训练前完整 pytest：**203 passed in 13.52s**。使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 排除本机第三方 pytest 插件干扰；Matplotlib 使用 `Agg`。
- 运行环境：Python 3.8.8、NumPy 1.24.4、PyTorch 2.4.1+cpu、PyYAML 6.0.3、Matplotlib 3.7.5、pytest 8.3.5；CPU，Torch 单线程。
- 新运行的 Direct Attacker sanity check 通过：初始距基地 500.000 m，第 281 步为 10.261147 m，第 282 步进入攻击球，距离 8.261147 m，耗时 28.2 s。进入时速度约 20.000 m/s，距基地没有增加的步，无 hard clipping、无超时，数值及物理约束均通过检查。见 [sanity metrics](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/preflight/sanity/metrics.json) 和 [sanity 原始轨迹](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/preflight/sanity/trajectory.npz)。

实际执行的主要命令如下。仅明确传入既定 100000 步和专用输出目录，没有覆盖任何研究超参数；正式评估按本次要求将回合数从配置默认 10 提升为 100。

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:MPLBACKEND = 'Agg'
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m scripts.sanity_direct_attacker --config configs/baseline_3a1c.yaml --output-dir outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/preflight/sanity
& .\.venv\Scripts\python.exe -u -m scripts.train_mappo --config configs/mappo_baseline.yaml --total-steps 100000 --output-dir outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000
& .\.venv\Scripts\python.exe -m scripts.evaluate_mappo --checkpoint outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/checkpoints/final.pt --episodes 100 --output-dir outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/evaluation/final
& .\.venv\Scripts\python.exe -m scripts.evaluate_mappo --checkpoint outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/checkpoints/best.pt --episodes 100 --output-dir outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/evaluation/best
& .\.venv\Scripts\python.exe outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/analysis_scripts/evaluate_random.py --config configs/mappo_baseline.yaml --episodes 100 --seed 1000 --output-dir outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/evaluation/random
```

训练 stdout/stderr 保留于 [training_console.log](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/training_console.log)；完整核验结果见 [experiment_verification.json](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/experiment_verification.json)。

已有数据的训练曲线、3D 图和输入诊断可通过以下分析命令重新生成，无需再训练：

```powershell
& .\.venv\Scripts\python.exe outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/analysis_scripts/analyze_training.py --run-dir outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000
& .\.venv\Scripts\python.exe outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/analysis_scripts/analyze_behavior.py --run-dir outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000
& .\.venv\Scripts\python.exe outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/analysis_scripts/analyze_input_activation.py
```

## 3. Task definition

- 三架 pursuer 防御基地 `[0,0,0]`，一架 Direct Attacker 从 500 m 检测球上的固定点进攻。
- 每步同步推进四机后判断：任意敌我距离严格 `<5 m` 为拦截成功；否则敌机距基地严格 `<10 m` 为敌方攻击成功。若同一步两条件都满足，拦截优先；前两者均不成立且达到 500 步才超时。
- Direct Attacker 不避让 pursuer，不随机转向，不做预测；按基地视线方向生成 20 m/s 期望速度，通过现有速度误差控制器及动力学约束执行。
- 这是三维、无障碍、固定初始化任务。`[-750,750]^3` 为数值工作区，z 的负半轴不代表地形；保留 hard position clipping 和原速度处理，不引入墙面策略。
- 各 actor 使用原始 21D 局部观测；共享 actor，centralized critic 使用原始 24D 全局状态。没有观测噪声、随机初始化、显式包围奖励、curriculum 或轨迹预测。

## 4. 全部环境关键参数

配置原件为 `configs/baseline_3a1c.yaml`；实际运行副本见 [environment_config.yaml](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/environment_config.yaml)，训练前副本见 [environment_parameters.yaml](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/preflight/environment_parameters.yaml)。

| 参数 | 实际值 |
| --- | --- |
| dimension / pursuers / attackers | 3 / 3 / 1 |
| dt / max_episode_steps | 0.1 s / 500 |
| bounds.low / bounds.high | `[-750,-750,-750]` / `[750,750,750]` m |
| base_position | `[0,0,0]` m |
| detection_radius / attack_radius / capture.radius | 500 / 10 / 5 m |
| P0 / P1 / P2 初始位置 | `[1.5,0,1]` / `[-0.75,1.25,1]` / `[-0.75,-1.25,1]` m |
| attacker 初始位置 | `[497.49371855331,0,50]` m |
| 四机初始速度 / 加速度历史 | 全部为零 / reset 为零 |
| 双方 max_horizontal_speed | 20 m/s |
| 双方 max_climb_speed / max_descent_speed | 8 / 6 m/s |
| 双方 max_horizontal_acceleration / max_vertical_acceleration | 3 / 2 m/s² |
| 双方 max_jerk | 5 m/s³ |
| attacker.policy.mode / attack_speed | direct / 20 m/s |
| attacker.policy.velocity_gain / distance_epsilon | 1.0 s⁻¹ / 0.000001 |
| pursuer_safety_distance | 2 m |
| capture_reward / attack_failure_penalty | +100 / -100 |
| time_penalty / collision_penalty / progress_coefficient | -0.01 / -5 / 1.0 |
| YAML smoke rollout 默认 seed / episodes | 0 / 3；本轮正式 random 单独运行 100 回合 |

动作是三维归一化加速度，shape `(3,3)`、范围 `[-1,1]`，按原水平/垂直加速度尺度映射并经过原 jerk、加速度和速度限制。Actor 观测为自身速度 3D、敌机相对位置 3D、敌机相对速度 3D、两队友相对位置 6D、两队友相对速度 6D；critic state 为 `[P0,P1,P2,attacker]` 各自位置和速度，共 24D。加速度历史不在这两个输入中。

共享每步 reward 保持原定义：成功 +100、敌方攻击成功 -100、时间 -0.01、出现任意我方安全距离违规时 -5（每个 team timestep 计一次，不按碰撞对重复），以及 `1.0 × (上一步最近敌我距离 - 当前最近敌我距离)`。本文回报按 team reward 计一次，不将三份相同 reward 相加。此进展项可因敌机自行接近而增加；回合最小距离与终点最近距离是不同指标。

## 5. 全部 MAPPO 超参数

原配置为 `configs/mappo_baseline.yaml`；实际生效的完整参数见 [training_config.yaml](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/training_config.yaml)，与训练前 [effective_training_parameters.yaml](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/preflight/effective_training_parameters.yaml) 一致。

| 参数 | 实际值 |
| --- | --- |
| env_config / seed / device / torch_num_threads | baseline_3a1c.yaml（相对 configs） / 0 / cpu / 1 |
| output_dir | outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000（CLI 指定） |
| total_env_steps / rollout_steps | 100000 / 512 |
| log_interval_updates / checkpoint_interval_updates | 1 / 10 |
| capture_rate_window | 50 个完整 episodes |
| actor_hidden_sizes / critic_hidden_sizes | `[128,128]` / `[256,256]` |
| hidden activation / initialization | Tanh / orthogonal weights、zero biases（原实现） |
| initial_log_std / min_log_std / max_log_std | -0.5 / -5.0 / 2.0 |
| hidden_gain / actor_output_gain / critic_output_gain | 1.4142135623730951 / 0.01 / 1.0 |
| actor_learning_rate / critic_learning_rate | 0.0003 / 0.001 |
| adam_epsilon | 0.00001 |
| gamma / gae_lambda / clip_ratio | 0.99 / 0.95 / 0.2 |
| update_epochs / minibatch_size | 4 / 64 team timesteps（192 actor samples、64 values） |
| entropy_coefficient / value_loss_coefficient | 0.01 / 0.5 |
| max_grad_norm | 0.5 |
| advantage_normalization / normalization_epsilon | true / 0.00000001 |
| normalization.observation / global_state / reward_scale | false / false / 1.0 |
| evaluation 默认 episodes / seed | 10 / 1000；本次每模型 CLI 指定 100 episodes |
| visualization.seed / dpi / figure_size | 1000 / 160 / `[10,8]` |

训练采样 squashed Gaussian 动作；确定性评估使用 `tanh(mean)`。训练与评估的动作分布不同，不能把训练随机采样回报直接视为确定性模型回报。

## 6. 训练完成情况与 checkpoint

实际训练 **100000 环境步、196 次 PPO updates、354 个完整 episodes**；每个完整回合均为 282 步，共 99828 步。余下 172 步属于最后一个未结束回合，未冒充完整 episode。最后一次 rollout 只有 160 步以精确到达 100000。完整回合均为敌方攻击成功，无拦截、无超时。

已保存 19 个每 10 updates 的周期 checkpoint（update 10 至 190），以及 [final.pt](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/checkpoints/final.pt)、[best.pt](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/checkpoints/best.pt)，共 21 个；逐一加载核验 actor/critic、optimizer tensor 和保存的环境状态，均有限，无 NaN/Inf。日志中的训练指标均正常记录，所有数值型 CSV 单元均有限。未成功时 capture/interception time 留空或 JSON `null`，没有填为 0。

Best 保存于 **update 2 / 1024 steps / 已完成 3 episodes**。原代码按“近期训练拦截率优先，再比较近期平均回报”选取，窗口上限为 50、早期不足 50 时使用已有回合；当时 score=`[0.0,292.4142316248802]`。整个训练捕获率都为零，因此这是早期小样本训练回报启发式选出的 best，既不是按本轮 evaluation 挑选，也不意味着它学会了任务。

## 7. 训练数据与曲线

原始数据：[training_metrics.csv](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/training_metrics.csv)（354 行）及 [updates.csv](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/updates.csv)（196 行）。另保存 [training_data.npz](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/training_data.npz)、[updates_data.npz](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/updates_data.npz)，保留全部 CSV 字段及缺失值信息；生成过程和 SHA256 记录于 [training_analysis.json](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/training_analysis.json)。

Episode 曲线同时画原始值与 **trailing 50 episodes rolling mean**；前 49 个点只使用当时已存在的数据，未填造未来样本。横轴为回合结束时累计环境步。Update 曲线直接展示原始 update 指标，不做平滑。

| 指标 | 图文件 |
| --- | --- |
| episode reward | [reward_vs_training_steps.png](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/plots/reward_vs_training_steps.png) |
| interception success rate | [interception_rate_vs_training_steps.png](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/plots/interception_rate_vs_training_steps.png) |
| enemy attack success rate | [enemy_attack_rate_vs_training_steps.png](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/plots/enemy_attack_rate_vs_training_steps.png) |
| episode length | [episode_length_vs_training_steps.png](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/plots/episode_length_vs_training_steps.png) |
| minimum pursuer-attacker distance | [min_distance_vs_training_steps.png](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/plots/min_distance_vs_training_steps.png) |
| terminal attacker-base distance | [terminal_base_distance_vs_training_steps.png](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/plots/terminal_base_distance_vs_training_steps.png) |
| actor loss | [actor_loss_vs_updates.png](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/plots/actor_loss_vs_updates.png) |
| critic/value loss | [critic_loss_vs_updates.png](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/plots/critic_loss_vs_updates.png) |
| entropy | [entropy_vs_updates.png](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/plots/entropy_vs_updates.png) |

同一 [plots 目录](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/plots/) 还保存 timeout rate、collision steps、approximate KL、clip fraction、actor/critic gradient norm 六张图，共 15 张。图像均基于本次原始 CSV，生成脚本为 [analyze_training.py](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/analysis_scripts/analyze_training.py)。

## 8. 整个训练过程的趋势

| 按回合结束步分段 | 完整回合数 | 平均 reward | 平均回合最小敌我距离 (m) | 拦截 / 敌方攻击成功率 |
| --- | ---: | ---: | ---: | --- |
| 0–20k | 70 | 226.2477 | 108.0898 | 0% / 100% |
| 20k–40k | 71 | 205.8791 | 118.6828 | 0% / 100% |
| 40k–60k | 71 | 161.9914 | 142.6269 | 0% / 100% |
| 60k–80k | 71 | 152.8737 | 141.9356 | 0% / 100% |
| 80k–100k | 71 | 140.7803 | 122.0810 | 0% / 100% |

首 50 / 中间 50（episodes 153–202）/ 末 50 回合平均回报为 **234.8121 / 165.2967 / 151.6882**，首末相差 -83.1239。对应平均最小距离为 99.1578 / 144.9049 / 112.8223 m，后段有部分恢复，但没有持续接近 5 m。全训练最小距离的最好单回合也只有 14.6623 m。全部回合的长度固定为 282 步、终点敌机距基地固定为 8.261147 m、timeout 为零；这些是敌机不受干扰进攻的结果，不代表训练已收敛成功。

完整回合平均 reward 177.4169；169/354 回合出现安全距离违规，共 3341 个 collision steps，占完整回合步数 3.3468%。后期碰撞减少，但回报和任务成功率没有改善。

Actor loss 在 -0.017949 至 -0.000172 之间，首/末 update 为 -0.008678 / -0.008299；surrogate loss 变动不能替代任务进展。Value loss 首 20 / 末 20 updates 平均为 187.9471 / 34.6399，整体下降但有尖峰，范围 5.6317–422.9053，最后一个 update 为 119.2486；critic 更会拟合已有回报不意味着 actor 学会拦截。

Entropy 为 squashed Gaussian 的微分熵 Monte Carlo 估计，五个训练段均值约 **1.7840 → 1.7423 → 1.6849 → 1.6300 → 1.5371**，没有接近零方差的直接证据。Approximate KL 范围 0.002486–0.010751，clip fraction 范围 1.888%–13.428%，没有明显突发的大幅策略更新。

Pre-clipping critic gradient norm 的五段均值约 168 / 266 / 367 / 459 / 724，最后 update 达 3332.66；actor 最后为 9.76。代码原有 `max_grad_norm=0.5` 正常执行，日志记录的是裁剪前范数，不能据此判断裁剪失效。数值运行稳定；性能则长期无成功、回报退化，不能称为稳定学会，也不能仅凭下降就断言熵坍塌。

## 9. Final model 正式评估

使用 100000-step `final.pt`，实际执行 100 回合确定性评估（seeds 1000–1099），不从训练回合复用指标。数据位于 [evaluation/final](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/evaluation/final/)。

结果：0/100 拦截、100/100 敌方攻击成功、0/100 timeout；平均 reward 161.7954，长度 282 步 / 28.2 s，平均回合最小敌我距离 58.3119 m，终点最近敌我距离 233.7928 m，终点敌机距基地 8.261147 m。成功拦截时间无样本，保存为 `null`。无我方 collision steps，无 hard clipping。

## 10. Best model 正式评估

使用 update 2 / 1024-step `best.pt`，同样实际执行 100 回合确定性评估（seeds 1000–1099）。数据位于 [evaluation/best](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/evaluation/best/)。

结果：0/100 拦截、100/100 敌方攻击成功、0/100 timeout；平均 reward 352.6662，长度 282 步 / 28.2 s，平均回合最小敌我距离 20.2497 m，终点最近敌我距离 42.9220 m，终点敌机距基地 8.261147 m。成功拦截时间为 `null`；无我方 collision steps，无 hard clipping。它比 final 更接近敌机，但离严格 `<5m` 的目标仍有明显差距。

**重复确定性表现不等于泛化。** 两模型各自 100 行 metrics 除 episode/seed 外完全相同。另对 seed 1001、1099 重放完整回合，其 states/actions/rewards 与保存的 seed 1000 逐元素相同；固定初态、确定性敌机和确定性 actor 导致相同轨迹。这里没有 100 个独立初始场景，不能给出泛化成功率或利用重复次数制造置信度。

## 11. 正式 Random Baseline 对照

本次重新运行 100 个完整 random episodes、28171 个实际环境步，seeds 1000–1099；使用现有 `env.sample_actions()` 的均匀随机动作与相同 evaluation 路径，未使用旧 3 回合结果。100 条状态轨迹均不同，全部保存于 [evaluation/random/episodes](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/evaluation/random/episodes/)。随机评估脚本为 [evaluate_random.py](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/analysis_scripts/evaluate_random.py)，逐项复核见 [independent_data_audit.json](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/evaluation/random/independent_data_audit.json)。

| 指标 | Final，100 episodes | Best，100 episodes | Random，100 episodes |
| --- | ---: | ---: | ---: |
| interception success rate | 0% | 0% | 2% |
| enemy attack success rate | 100% | 100% | 98% |
| timeout rate | 0% | 0% | 0% |
| average episode reward | 161.7954 | 352.6662 | 305.1595 |
| average episode length (steps) | 282.00 | 282.00 | 281.71 |
| average interception time，仅成功 (s) | N/A | N/A | 26.75（2 回合） |
| average episode minimum pursuer-attacker distance (m) | 58.3119 | 20.2497 | 42.0406 |
| average terminal nearest pursuer-attacker distance (m) | 233.7928 | 42.9220 | 50.1316 |
| average terminal attacker-base distance (m) | 8.2611 | 8.2611 | 8.8411 |
| collision episode rate | 0% | 0% | 49% |
| collision steps，总数 | 0 | 0 | 886 |
| collision step rate | 0% | 0% | 3.1451% |
| average pairwise pursuer distance (m) | 142.3246 | 3.2023 | 46.1881 |
| hard boundary clip count | 0 | 0 | 0 |

Collision 指我方任意一对距离 `<2m`，并非敌我拦截。Random 回报标准差 71.7178、范围 94.5539–591.5613；每回合最小距离标准差 20.5721 m。Random 的两次偶然成功不能建立稳健拦截能力，也不宜将 2% 与固定初态重复的 0% 解释为统计显著差异。

**MAPPO 没有明显优于 random。** Best 有更高回报和更小最近距离，但未转化为任何任务成功；final 回报和最小距离均劣于 random 均值。没有训练回报明显改善的证据。

## 12. 完整 3D trajectory

代表回合统一取每种策略正式评估的第 1 回合（seed 1000），不筛选较好轨迹。三图均包含完整 episode 和最后 5 s 局部视图，标注基地、10 m attack sphere、四机轨迹与起终点、outcome、steps、最小敌我距离、终点敌机距基地。

- [Random 3D trajectory](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/trajectories/random_trajectory_3d.png)：282 步，最近 30.7714 m，敌方攻击成功。这是代表回合值，不同于 100 回合均值 42.0406 m。
- [Final 3D trajectory](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/trajectories/final_trajectory_3d.png)：282 步，最近 58.3119 m，敌方攻击成功。
- [Best 3D trajectory](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/trajectories/best_trajectory_3d.png)：282 步，最近 20.2497 m，敌方攻击成功。

原始轨迹分别保存于 `evaluation/{random,final,best}/first_episode.npz`；原始行为对照见 [behavior_comparison_raw.png](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/behavior/behavior_comparison_raw.png)，逐机时间序列和诊断见 [behavior_summary.json](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/behavior/behavior_summary.json)。

Final 在前 5 s 主动向敌机前出，自身径向接近速度均值约 3.52–4.09 m/s；但随后侧偏、散开，没有形成有效提前截击。P0 在 17.5 s 最接近时仍有约 56.54 m 的 y 偏差，并继续沿错误侧向加速。末 5 s 三机自身径向接近速度均为负，末态离敌机 233.79–435.03 m。动作已有结构，不能简单称为仍在均匀随机飞行；主要失败是方向对准和错失后的修正。

Best 三机动作小、近似同向缓慢漂移，全回合各飞约 51–52 m，自身朝敌机的平均径向速度约 1.09–1.12 m/s。P1 在 26.5 s 最近，虽在 x 方向相遇，y/z 误差仍约 -18.17/+8.52 m；大部分距离收缩来自敌机主动飞来，没有形成可用拦截。

## 13. Cooperation、一机主追与取巧检查

| 逐机表现 | P0 | P1 | P2 |
| --- | ---: | ---: | ---: |
| Final 全程路程 (m) | 435.641 | 447.096 | 342.845 |
| Final 各机最小敌我距离 (m) | 58.312 | 94.239 | 144.310 |
| Final 作为最近机的 post-step 占比 | 74.8% | 0% | 25.2% |
| Best 各机最小敌我距离 (m) | 22.160 | 20.250 | 24.119 |

**没有明确协同。** Final 的最近接触主要由 P0 提供，另两机有效接近有限；三机均在运动，P1 路程甚至最长，不能说“两机不动”或严格的“一机主追、两机停留后方”。更准确是三机均前出，P0 最接近，整体方向/转向失配。Best 近似平行运动，末态从敌机看三机夹角仅约 1.79°–4.88°，未出现互补方向的包夹。Final 的末态大夹角发生在数百米之外，不是有效包围。未做移除队友消融，因此没有量化各机因果贡献。

三种策略代表轨迹与正式 random 全部 100 回合均无 hard clipping；两确定性模型轨迹重复，离工作区边界至少约 252.5 m。没有利用边界限制敌机的现象。

Final 的三机在基地 10 m 内仅停留约 3.1–3.8 s；Best 停留 10.4–12.0 s，其中速度 `<=1m/s` 的时间约 6.5–6.7 s，随后缓慢离开至距基地约 50–54 m。两者均失败，没有成功的“原地等敌机撞上来”漏洞。

Final/best 确定性动作分量 `abs(a)>0.95` 占比均为 0，无输出动作饱和证据。Final P0/P1 约 51.1%/53.5% 步达到水平速度上限，这是动力学限速，和 actor 输出饱和不同。Final Gaussian std 仍约 `[0.583,0.629,0.625]`，没有接近零方差的策略坍塌证据；任务表现确实退化。

## 14. Interception geometry

Final/best 均无成功回合，成功拦截位置、完成者及其余两机当时距离均为 `null`。图中最近接近位置没有冒充 interception point。

正式 random 中的两次成功如下；“拦截位置”列为判定成功时的敌机位置，敌我距离仍按 `<5m` 判定，不表示两点完全重合。

| Seed | Steps / time | 完成者 | 敌机位置 `[x,y,z]` (m) | 敌机距基地 (m) | `[P0,P1,P2]` 距敌机 (m) |
| --- | --- | --- | --- | ---: | --- |
| 1004 | 277 / 27.7 s | P0 | `[18.186917,0,1.644858]` | 18.261147 | `[4.627888,36.063101,65.317503]` |
| 1020 | 258 / 25.8 s | P2 | `[56.032449,0,5.067678]` | 56.261147 | `[74.439012,89.859866,4.266971]` |

其余两机距离较远，这两次是随机偶遇成功，没有协同拦截的证据；完整数据保留在相应 episode NPZ 和 [random evaluation_metrics.csv](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/evaluation/random/evaluation_metrics.csv)。

## 15. 是否学会任务

明确分类为 **Not learned**。全部训练回合 0% 拦截；两模型确定性评估 0% 拦截、100% 敌方成功；reward 总体退化；没有有效提前截击或协同；也没有优于 random 的任务表现。Best 的较高 dense reward 和 20.25 m 最近距离可描述为“更接近”，不足以认定 Partial learning 的有效拦截能力。Final 有主动前出动作，但任务判定不因此放宽。

此结论限于当前代码、固定初态、seed 0、100k 预算；不是所有 MAPPO 配置或更长训练必然失败的证明。没有多 seed 方差或泛化评估。

## 16. Failure mode 与原因排序

**观测到的最大 failure mode：主动运动与进攻航线未对准，错失后没有有效修正；团队未形成互补拦截，优化过程也没有将几何接近转化为 `<5m` 成功。** 以下是按现有证据排序的待验证假设，不是已确立因果关系。

### 16.1 Optimization / representation：优先检查输入尺度和优化信号

配置中的 observation/global-state normalization 均为 false；actor 的 input_scale 全为 1。初始敌机相对 x 坐标约 496–498 m、相对高度约 49 m，而队友相对位置最大约 2.5 m、自身速度为 0。离线读取实际 checkpoint 和首回合 NPZ，重建 actor 输入并复核保存动作一致，测量如下：

| Actor 激活诊断 | Final | Best |
| --- | ---: | ---: |
| 首层 Tanh 饱和比例，整回合，`abs(preactivation)>3` | 95.08% | 88.07% |
| 首层饱和比例，reset 状态 | 96.09% | 97.92% |
| 第二层饱和比例，整回合 | 3.50% | 2.31% |
| 输出动作 `abs(a)>0.95` 比例 | 0% | 0% |
| 最大确定性动作绝对值 | 0.8231 | 0.0508 |

同配置 seed 0 新初始化 actor **仅做一次初态前向计算**，未运行环境、更新或保存新模型，首层饱和已达 98.70%，平均 Tanh 导数约 0.000387。结合 critic 裁剪前梯度增长，输入尺度/首层低敏感性是优先排查的具体优化候选机制。该阈值为诊断定义，不是配置改动；首层饱和不等于整网或动作输出饱和，也未证明它是唯一根因。明细见 [input_activation_analysis.json](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/input_activation_analysis.json)。

探索熵随训练下降且训练从未采到成功；有限预算、探索分布和 PPO 信号可能共同影响学习。但 KL/clip fraction 没有明显失控，单次 seed 不能据此断言学习率错误、100k 必然不足或换超参数必然有效。

### 16.2 Reward / credit assignment：距离进展与点防御成功未充分对齐的可能

敌机自行向基地飞来即可产生正的最近距离进展；best 在 0 拦截时仍有 352.6662 回报，说明高回报不能作为拦截证据。共享最近距离奖励主要反映最近一机，另两机的有用贡献难以从这一标量区分；碰撞惩罚又可能让简单共同追踪与安全分离产生权衡。需要分别分析各项奖励、时间和逐机贡献。

这些现象支持检查 reward/credit assignment，尚不能证明当前 reward 的全局最优策略偏好失败、PPO 已找到该最优解，或某一个系数单独导致失败。本轮保留原 reward，不添加包围项。

### 16.3 Observation：点防御信息与隐藏动力学状态

21D actor 没有显式 base-relative information，可能不利于学习“在攻击球外拦截”与跨进攻方向泛化；jerk 限制使用的前一步加速度也不在 21D/24D 输入中。二者均值得单独验证，但当前固定初态不能把缺少基地信息直接认作失败的必要原因。

为区分“优化失败”与“固定案例物理不可达”，另执行了**一次零训练、无搜索的简单追踪诊断**，只用现有 21D 中自身速度及敌机相对位置，以 YAML 中 20m/s、gain=1 生成速度误差加速度，再交给完全不变的 `env.step()`。该诊断在 160 步 / 16.0 s 成功：P0 距敌机 3.2438 m，P1/P2 各 5.4375 m，敌机距基地 252.2611 m，hard clip=0。数据见 [physical_feasibility/diagnostic.json](../outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/behavior/physical_feasibility/diagnostic.json) 和原始 trajectory NPZ。

它只证明当前固定案例在现有约束和相对信息下存在可行追踪，不是 MAPPO 学习结果、正式 random 对照或优质协同策略。它有 98 个 collision steps（61.25%），回报仅 103.5644；成功但碰撞多的控制器回报低于一些失败轨迹，进一步提供分析奖励权衡的动机，却不能证明无碰撞成功策略也会得低回报。

### 16.4 Environment / task：没有发现不可达或边界帮助的证据

Sanity 和 replay 一致、数值及物理约束通过；单次可行控制证据排除了“该固定任务必然物理不可拦截”。基地攻击半径 10 m 大于捕获半径 5 m，单纯待在基地点通常会先触发敌方攻击，主动前出有必要。不能据当前结果擅自降低速度、放大捕获半径或延后敌方成功判定。

Fixed initialization 限制探索覆盖和泛化，同一共享 actor 对相似输入容易给相似动作；它可能形成局部同向轨迹，但当前没有通过干预建立因果关系。无 hard clipping 或地形漏洞证据，因此边界不应是本轮优先解释。

## 17. 下一步最值得验证的三个问题

1. **原始输入尺度是否使 PPO 难以获得有效动作修正？** 先核对各输入组、actor/critic 激活和逐层梯度，再在后续获准的独立实验中只切换已有固定 normalization 开关，保持任务、reward、网络和预算不变。当前首层饱和证据使它比盲目延长训练或多维调参更优先。
2. **当前共享 reward 如何区分有效截击、敌机自行接近与安全分离？** 先做各奖励分量和成功/失败几何的分解，确认捕获收益、距离进展、碰撞与 credit assignment 的关系；如需改 reward，再独立做单因素对照，不把 observation 或算法改动混入。
3. **在控制输入尺度之后，显式基地相对信息是否改善点防御决策？** 对照是否需要基地向量/距攻击球余量才能学到外出拦截和错失后修正，同时观察三机贡献。先在同一固定案例验证机制；随机初始化和泛化是另行授权的研究步骤。

## 18. 是否建议修改 baseline，以及交付位置

**不建议现在同时修改 observation、reward 和 PPO 超参数，也不建议凭本次结果直接增大训练预算或降低任务难度。** 下一轮优先验证输入缩放与优化信号；已有 normalization 开关的受控实验优于先改 21D observation 定义。Reward 与 base-relative observation 都有合理疑点，应按上述顺序分别验证；学习率、熵系数等超参数暂没有充分证据要求立即调整。本轮没有执行这些后续实验或修改。

主要交付：

- 本报告：`experiments/DIRECT_ATTACKER_MAPPO_100K_REPORT.md`。
- 配置、代码版本、SHA256、检查结果：`outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/{freeze_manifest.json,source_snapshot/,metadata.json,experiment_verification.json}`。
- 训练：`training_console.log`、`training_metrics.csv`、`updates.csv`、`training_data.npz`、`updates_data.npz`、`checkpoints/{final,best}.pt`。
- 曲线：`outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/plots/`，共 15 张，原始数据和 rolling window 标注均保留。
- 正式各 100 回合结果：`outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/evaluation/{final,best,random}/` 下 CSV、summary、metadata 和首回合 NPZ；random 另有全部 100 条 NPZ。
- 三策略完整 3D 图：`outputs/v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000/trajectories/{random,final,best}_trajectory_3d.png`。
- 行为、输入激活与单次可达性诊断：`behavior/`、`input_activation_analysis.json`；可复查分析脚本位于 `analysis_scripts/`。

本次保留真实失败结果，没有修改已有源码、配置、tests、README 或 BASELINE_SPEC；不需要额外确认即可查看全部产物。后续研究性改动应另行明确实验范围。
