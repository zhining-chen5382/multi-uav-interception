# 项目续接记录：2026-11-07

路径说明（2026-11-08）：本文链接及命令示例已更新为版本目录；冻结快照、原始日志、metadata 与权重内嵌旧路径保留，见 [目录迁移表](output_path_migration_20261108.json)。

本记录用于恢复项目上下文。2026-11-07 实际重新检查了仓库、配置、已有实验数据和模型文件，并运行完整 pytest。此前主要实验完成于 2026-10-25；本次只补记录与核验，没有运行新训练或改变研究设定。

**当前状态：环境、MAPPO 训练、评估与视频流程已跑通；最新实验已经完成 200000 步，但还没有学会稳定拦截。接下来最有价值的工作是定位近距离错失机制，并检查奖励目标与模型筛选指标。**

## 1. 当前任务和设置

当前是三维点防御：三架我方 MAPPO 拦截机保护固定基地，一架规则型 Direct Attacker 从约 500 米外直飞基地。旧追逃任务、旧 evader 边界实验保留用于历史复现。

| 项目 | 当前值 |
| --- | --- |
| 配置 | [环境 YAML](../configs/baseline_3a1c.yaml)、[MAPPO YAML](../configs/mappo_baseline.yaml) |
| 基地 / 敌机初始位置 | `[0,0,0]` / `[497.49371855331,0,50]` 米 |
| 三机初始位置 | `[1.5,0,1]`、`[-0.75,1.25,1]`、`[-0.75,-1.25,1]` 米 |
| 初始化 | 固定位置，四机零初速度、零加速度历史 |
| 成功 / 失败 | 任一敌我距离严格 `<5 m` 捕获；未捕获且敌机距基地严格 `<10 m` 则敌方攻击成功 |
| 同步触发 | 捕获优先于敌方攻击成功；terminal 优先于 timeout |
| 时间 / 数值空间 | `dt=0.1 s`，最多 500 步；三轴 `[-750,750] m` |
| 双方物理限制 | 水平速度 20 m/s；爬升/下降 8/6 m/s；水平/垂直加速度 3/2 m/s²；jerk 5 m/s³ |
| 敌机制导 | Direct Attacker，期望速度 20 m/s，速度跟踪增益 1；无规避、预测或 soft boundary steering |
| Action / observation / state | 三机各 3D 归一化加速度；actor 各 21D 局部观测；critic 24D 全局状态 |
| 网络 | 共享 actor `[128,128]`、critic `[256,256]`，Tanh |
| 训练设置 | seed 0、CPU、1 线程；rollout 512，epochs 4，minibatch 64 |
| 学习率 / 折扣 | actor `3e-4`、critic `1e-3`；gamma `0.99`、GAE lambda `0.95` |
| PPO / normalization | clip `0.2`，entropy 系数 `0.01`；observation/state normalization 关闭，reward scale 1 |

工作空间只作数值保护，hard position clipping 保留；当前没有地面模型。捕获只检查离散更新后的状态，没有步内连续接触检测。

完整定义见 [BASELINE_SPEC.md](../BASELINE_SPEC.md)。README/SPEC 中“本轮只做 20k、不跑 100k”的文字属于上一次 reward 修复任务的范围记录；后续已授权的长训练结果见下文与独立报告。当前配置中的默认训练预算仍是 100000，最新实际 run 通过运行参数和续跑达到 200000；没有把 YAML 静默改成 200k。

## 2. 已完成到哪里

| 阶段 | 产物 / 意义 |
| --- | --- |
| 环境骨架、动力学、MAPPO、评估与可视化 | 模块与基本流程已实现，有自动测试 |
| 历史追逃与边界修复 | 历史问题与对照保留，不能当作当前点防御成绩 |
| 点防御与 Direct Attacker | 攻击机使用速度跟踪，正常进入基地攻击球 |
| Active progress reward 修复 | 扣除敌机自行向基地接近的距离进展，存活惩罚改为按秒计 |
| 20k 诊断 | `outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/`；IDE 打开的 CSV 属于此较早实验 |
| 100k → 条件续至 200k | 最新实验位于 `outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/`，原目录名为 `active_reward_100k`，实际训练至 200k |
| 已保存模型筛查与视频 | `outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/checkpoint_screening/` |

最新长训练是一次从零开始的 run：先完成 100k，依据当时记录的末期几何改善沿同一状态续至 200k，没有换配置重训挑最好的一次。100k 时的完整数据单独归档在 `at_100k/`。两段训练共 392 updates，不能当作从一开始指定 200k 的更新调度。

| 节点 | Updates | 完整回合 | 训练采样成功 |
| --- | ---: | ---: | ---: |
| 100000 步 | 196 | 354 | 0 |
| 200000 步 | 392 | 710 | 6（约 0.845%） |

200k 的完整回合共 199741 步，另有 259 步未完回合。最后一次训练拦截发生在累计 151801 步，之后 48199 步没有再成功。训练过程中有阶段性接近改善和稀疏成功，后期回退；没有稳定任务掌握或明确协同的证据。历史核验未发现 NaN/Inf 或持续数值发散，但数值稳定不代表策略性能稳定。

详见 [长训练报告](ACTIVE_REWARD_LONG_TRAINING_REPORT.md)。

## 3. 模型筛查与三个不同的 best

筛查覆盖主目录 50 个和 100k 归档 27 个 checkpoint 文件，共 77 条路径，去重为 50 个不同策略。每个策略实际完成一次确定性评估，全部未捕获；回报第一名额外实际运行 100 回合，仍全部失败。

固定初始化和确定性动作重复同一轨迹，改变 seed 不构成泛化测试。训练时随机动作下的 6 次成功，与保存模型确定性评估的全部失败，必须同时保留。

| 模型含义 | 训练步数 / update | 最近距离 | 结束时最近距离 | 无折扣回报 | 确定性捕获 |
| --- | --- | ---: | ---: | ---: | --- |
| 原训练窗口选出的 `checkpoints/best.pt` | 135328 / 265 | 16.274 m | 201.191 m | -510.161 | 否 |
| 筛查中回报最高 `best_deterministic.pt` | 1024 / 2 | 20.835 m | 22.936 m | -121.906 | 否 |
| 筛查中距离最近 `closest_approach.pt` | 158368 / 310 | **8.629 m** | 204.139 m | -303.110 | 否 |
| 最终 `checkpoints/final.pt` | 200000 / 392 | 129.966 m | 187.813 m | -286.783 | 否 |

这几个模型网络结构一致，权重与训练时刻不同。原训练 `best.pt` 依据最近 50 个训练回合的捕获率、再平均回报挑选，不能单凭名称认定它确定性评估最好。筛查生成的是另外的模型副本，没有覆盖原 best。

最值得观察的近距模型为 **158368 步 / update 310**，原文件是 `checkpoints/update_000310.pt`。在第 206 步（20.6 秒）由 pursuer 1 接近敌机至 8.629391 米，仍在 5 米捕获范围之外，随后错过并拉远。终局第 282 步敌机到达基地，最近敌我距离已增至 204.139 米。

相关文件：

- [筛查报告](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/checkpoint_screening/SCREENING_REPORT.md)、[全部模型回报排序](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/checkpoint_screening/ranking.csv)、[全部模型最近距离排序](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/checkpoint_screening/closest_approach_ranking.csv)。
- [最近距离模型副本](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/checkpoint_screening/closest_approach.pt)、[完整回合视频](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/checkpoint_screening/videos/closest_approach_full_episode.mp4)、[3D 图](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/checkpoint_screening/closest_approach_trajectory_3d.png)。
- [回报最高模型副本](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/checkpoint_screening/best_deterministic.pt)、[完整回合视频](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/checkpoint_screening/videos/best_deterministic_full_episode.mp4)。

两段视频均为 1 倍速，完整仿真 28.2 秒；保留初态后，视频 283 帧、10 fps、28.3 秒。真实结局均为敌方攻击成功。

## 4. 上次讨论停在 reward 问题

用户观察到：158368 步模型更接近拦截，回报却比 1024 步模型低，因此怀疑当前 reward 设计有问题。这个现象已经核对，但尚未确定下一版奖励公式，也没有据此修改代码。

当前奖励在 [envs/rewards.py](../envs/rewards.py) 中计算：

```text
active_progress = (d_PE_previous - d_PE_current)
                  - (d_EB_previous - d_EB_current)
reward = active_progress + survival + collision + capture + attack_failure
```

`d_PE` 为当时三机中最近的敌我距离，`d_EB` 为敌机距基地距离。active 系数 1，survival `-0.2/s`，碰撞每个违规步团队扣 5，捕获 +100，敌方攻击成功 -100。

对这两个同长度、同初态、同敌机轨迹、均失败且无碰撞的回合，未折扣 active 差分逐步抵消，因此总回报可写成 `共同常数 - 终局最近敌我距离`。中间达到过 8.63 米，不会独立留在整回合总分中。158368 步模型到最近点时累计 +145.920，后续 -449.030，总计 -303.110。

| 实际奖励分量 / 口径 | 158368 步近距模型 | 1024 步回报最高模型 |
| --- | ---: | ---: |
| Active 累计 | -197.470 | -16.266 |
| Survival | -5.640 | -5.640 |
| Collision / capture / attack failure | 0 / 0 / -100 | 0 / 0 / -100 |
| CSV 无折扣总回报 | -303.110 | -121.906 |
| 同一轨迹按 gamma=0.99 重算完整折扣回报 | **+9.297** | **-5.873** |

**这里有两个不同问题：CSV 总回报不能单独代表拦截能力；训练 reward 与最终任务目标是否对齐，仍需诊断。** 折扣后两条轨迹排序反转，不能只根据无折扣回报说 PPO 更偏好留在基地附近。它也不能证明奖励已合理、critic 已学对或 PPO 更新方向正确；实际更新还涉及 rollout、bootstrap、GAE 与价值估计。

本次从原始 NPZ 再次计算了上述数字，原始指标与检查结果见 [本次核验 JSON](../outputs/_project_项目核对与管理资料/20261107_resume/verification.json)。

## 5. 当前已知限制与尚未证明的原因

- Actor 只有自身速度和相对敌机/队友状态，不能唯一恢复基地相对方位和距离。
- Jerk 需要加速度历史，但既定 21D observation / 24D state 不包含该历史。
- 主要轨迹存在侧向/高度偏离、过冲和错失后缺少有效修正；这些现象尚未归因到唯一原因。
- 未观察到明确互补截击或协同；最近机身份不能直接代表其他飞机没有贡献。
- 当前只在固定初始化、单训练 seed 下验证，不能宣称泛化能力。
- 没有仅因失败自动增加包围奖励、噪声、预测、随机初始化、障碍物、curriculum，或修改捕获半径/物理限制。

## 6. 建议接下来的三个任务（尚未执行）

1. **先解释 8.63 米为什么错过。** 围绕 update 310、第 206 步前后的窗口，检查相对位置、相对速度、物理加速度、jerk/限幅及转向时机，区分横向误差、高度误差、过冲和修正不足。先利用已有 NPZ 与冻结模型做诊断。
2. **检查学习目标与模型筛选。** 并列捕获率、回合最近距离、终局距离、无折扣/折扣回报，进一步检查 critic 和 advantage 对接近/错失的评价。现有成功优先的筛查规则保留；不要再把文件名 best 或单一 CSV 回报解释成拦截能力。
3. **依据证据再定一个研究改动。** 在 reward、任务可观测性、控制时序等候选中选一个明确假设做单因素验证，保留原 baseline 参照。本次没有决定或启动新的改法、训练预算或多配置搜索。

建议下一轮从第 1 项开始：目标是回答“差的 3.63 米来自哪里、为何没有及时纠正”，并同时保留 reward 时间分量，形成可审查的诊断结果。

## 7. 本次重新核验、版本与复现入口

- 本次完整 pytest：**231 passed，exit code 0**；关闭第三方插件自动加载。日志：[pytest.log](../outputs/_project_项目核对与管理资料/20261107_resume/pytest.log)。
- 当前 HEAD：`c6fe7e7c64e62425c079960b3895615b6f274113`。已有未提交源码/配置修改仍在，commit hash 单独不足以复现实验。实际训练源代码还应以 [43 文件冻结清单](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/preflight/freeze_manifest.json) 与 [源码快照](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/source_snapshot/) 为准。
- 本次检查：43 个冻结文件与训练时一致；77 条原始 checkpoint 路径的 SHA-256 与筛查记录一致；两段视频 SHA-256 与视频 manifest 一致。
- 本次只新增本记录，以及 `outputs/_project_项目核对与管理资料/20261107_resume/` 下的核验脚本、测试日志、JSON。训练、配置、模型与原始结果均未改动；没有 Git add/commit/merge/push。

在仓库根目录重跑测试：

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
& .\.venv\Scripts\python.exe -m pytest -q
```

如需重新观察最近距离模型，可使用新的输出目录实际运行一次评估；这是可选命令，本次没有执行：

```powershell
& .\.venv\Scripts\python.exe -m scripts.evaluate_mappo --checkpoint outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/checkpoint_screening/closest_approach.pt --episodes 1 --seed 1000 --output-dir outputs/resume_closest_u310_evaluation
```

评估默认使用 checkpoint 内的环境快照。避免覆盖已有输出，也不要把 `outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000` 或旧追逃 v01-r03 的结果混入最新训练。

阅读顺序：本记录 → [长训练报告](ACTIVE_REWARD_LONG_TRAINING_REPORT.md) → [模型筛查报告](../outputs/v05_ActiveProgress奖励/v05-r02_20261025_seed000/checkpoint_screening/SCREENING_REPORT.md) → [技术规格](../BASELINE_SPEC.md)。
