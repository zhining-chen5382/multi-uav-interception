# v06-r01：Actor / Critic 输入归一化 20k 对照

路径说明（2026-11-08）：本文链接及命令示例已更新为版本目录；冻结快照、原始日志、metadata 与权重内嵌旧路径保留，见 [目录迁移表](output_path_migration_20261108.json)。

日期：2026-11-08。**实际完成一次从零开始的 20000 步训练；数值检查通过，但没有学会拦截。**
相同预算下，没有证据表明这次输入缩放改善了任务表现。原始数据与模型全部保留，不因结果不好追加训练。

## 1. 实验目的与唯一变化

检验混合量纲的物理输入是否造成 Tanh 饱和，以及开启已有固定尺度缩放后能否改善学习。
对照是历史 **v05-r01** 的 20k、seed 0 运行，不是训练预算不同的 v05-r02 200k 模型。

| 项目 | 对照 v05-r01 | 本次 v06-r01 |
| --- | --- | --- |
| 训练参数快照 | `outputs/v05_ActiveProgress奖励/v05-r01_20261025_seed000/training_config.yaml` | `outputs/v06_输入归一化/v06-r01_20261108_seed000/training_config.yaml` |
| `normalization.observation` | false | **true** |
| `normalization.global_state` | false | **true** |
| Reward scale | 1.0 | 1.0 |
| 环境步数 / seed | 20000 / 0 | 20000 / 0 |
| 初始化 / 已加载模型 | 从零 / 无 | 从零 / 无 |

除两个开关和输出路径外，有效训练配置逐项相等；环境 YAML 快照内容完全相同。
Actor/Critic 同时开启缩放，所以只能评价联合效果，不能区分两者各自的因果贡献。
未修改 reward、观测信息内容、物理状态、动作、网络结构、动力学、初始化、敌机策略或 PPO 更新。

配置入口：[mappo_normalized.yaml](../configs/mappo_normalized.yaml)。原始
[mappo_baseline.yaml](../configs/mappo_baseline.yaml) 仍关闭两个开关，没有被覆盖。

## 2. 代码与数据来源

- Git HEAD：`c6fe7e7c64e62425c079960b3895615b6f274113`，训练元数据明确记录 `git_dirty: true`。
  Commit hash 不是本工作区的完整快照；训练前另保存核心源码及配置 SHA256。
- 实际训练开始：`2026-11-08T06:59:52.496463+00:00`，北京时间 14:59:52。
- Python 3.8.8、NumPy 1.24.4、Torch 2.4.1+cpu；CPU、1 个 Torch thread。
- [训练元数据](../outputs/v06_输入归一化/v06-r01_20261108_seed000/metadata.json)、
  [训练前计划及保护清单](../outputs/v06_输入归一化/v06-r01_20261108_seed000/preflight/experiment_plan.json)、
  [执行命令、退出码与耗时](../outputs/v06_输入归一化/v06-r01_20261108_seed000/training_execution.json)。
- 旧训练、模型、评估和视频保留原位置。本次只更新统一导航，没有替换历史实验数据。
- 本轮没有 Git add、commit、merge 或 push。

## 3. 实际使用的环境和 MAPPO 参数

环境参数来自本次保存的 [environment_config.yaml](../outputs/v06_输入归一化/v06-r01_20261108_seed000/environment_config.yaml)：

| 环境项 | 值 |
| --- | --- |
| 飞行器 / 任务 | 3 pursuers + 1 Direct Attacker，3D 点防御，无障碍 |
| 基地 / 敌机初态 | `[0,0,0]` / `[497.49371855331,0,50]` m |
| 三机初态 | `[1.5,0,1]`, `[-0.75,1.25,1]`, `[-0.75,-1.25,1]` m |
| 速度 / 加速度初态 | 全零，固定初始化 |
| Workspace | 每轴 `[-750,750]` m，hard position clip 保留 |
| dt / 最大回合步数 | 0.1 s / 500 |
| Capture / attack / safety radius | 5 / 10 / 2 m，严格小于判定 |
| 双方水平 / 爬升 / 下降最大速度 | 20 / 8 / 6 m/s |
| 双方水平 / 垂直最大加速度，jerk | 3 / 2 m/s²，5 m/s³ |
| Direct Attacker | attack_speed=20，velocity_gain=1，distance_epsilon=1e-6 |
| Observation / critic state / action | 原始 21D / 24D / 每机 `[-1,1]^3` 加速度 |

共享团队奖励保持原式，每个 transition 只累计一次：

```text
R_t = [(d_PE,t - d_PE,t+1) - (d_EB,t - d_EB,t+1)]
      - 0.2 * dt
      - 5 * I(任意我方机对距离 < 2 m)
      + 100 * I(capture)
      - 100 * I(enemy_attack_success)
```

`d_PE` 是当时三机中最近者到敌机的距离。捕获优先于敌方进攻成功，二者互斥。
训练目标折扣 gamma=0.99；CSV 记录未折扣的真实回合回报。

完整超参数见 [training_config.yaml](../outputs/v06_输入归一化/v06-r01_20261108_seed000/training_config.yaml)：

| MAPPO 项 | 实际值 |
| --- | --- |
| 预算 / rollout steps | 20000 / 512；最后一次 rollout 32 步 |
| Actor / Critic hidden sizes | `[128,128]` / `[256,256]`，Tanh |
| 初始 / 最小 / 最大 log_std | -0.5 / -5 / 2 |
| Hidden / actor output / critic output gain | sqrt(2) / 0.01 / 1 |
| Actor / critic LR，Adam epsilon | 0.0003 / 0.001，1e-5 |
| Gamma / GAE lambda / PPO clip | 0.99 / 0.95 / 0.2 |
| Epochs / team minibatch size | 4 / 64 |
| Entropy / value loss coefficient | 0.01 / 0.5 |
| Max gradient norm | 0.5 |
| Advantage normalization / epsilon | true / 1e-8 |
| Observation / global state scaling | true / true |
| Reward scale | 1.0 |
| 日志 / 周期 checkpoint | 每 1 / 10 updates |
| 训练 best 统计窗口 | 最近 50 个完整回合 |
| 确定性评估 | 每模型 10 回合，seeds 1000–1009 |

## 4. 缩放实现与训练前检查

复用 [build_input_scales](../algorithms/mappo/normalization.py) 和
[SharedActor.forward / CentralizedCritic.forward](../algorithms/mappo/networks.py)：

```text
Actor(o) = 原有 MLP(o / actor_input_scale)
Critic(s) = 原有 MLP(s / critic_input_scale)

自身速度尺度      = [20,20,8]
相对位置尺度      = [1500,1500,1500]
相对速度尺度      = [40,40,16]
Critic 每机 p/v   = [750,750,750,20,20,8]，重复 4 次
```

尺度从 workspace bounds 和双方速度限制推导，不使用运行均值、运行方差或输入 clipping。
环境及 rollout buffer 仍保存原始物理量；缩放只在网络内部执行。
`input_scale` 是 network buffer，会随 checkpoint 保存、加载；评估没有漏掉缩放，也没有重复缩放。

以**同一个初始物理状态、同一组 seed-0 初始权重**比较第一层，结果如下。
这里将 `|preactivation|>3` 作为描述性饱和阈值，不是新增训练参数：

| 初始检查 | 未缩放 | 开启缩放 |
| --- | ---: | ---: |
| Actor 输入最大绝对值 | 498.244 | 0.332 |
| Critic 输入最大绝对值 | 497.494 | 0.663 |
| Actor 第一层 `|z|>3` 比例 | 98.698% | 0% |
| Critic 第一层 `|z|>3` 比例 | 96.484% | 0% |
| Actor 平均 Tanh 导数 | 0.000387 | 0.998271 |
| Critic 平均 Tanh 导数 | 0.016301 | 0.996554 |

这支持“原始输入会造成初始第一层饱和”，不支持“消除饱和即可学会任务”。
数据见 [input_scaling_audit.json](../outputs/v06_输入归一化/v06-r01_20261108_seed000/preflight/input_scaling_audit.json)。

训练前 full pytest：**246 passed in 17.83s**。
新增 4 项 [归一化隔离与数值测试](../tests/test_normalization_experiment.py)：
原始环境状态/转移/奖励不变；同 seed 可训练参数完全一致；scale 和开关随 checkpoint 恢复；真实短 rollout 更新有限。
另新增 2 项版本配置/学习指纹测试，避免 next-run 为 v06 提供未缩放配置。
测试中的小网络只用于快速验证，没有用于正式 20k 训练。
[完整测试日志](../outputs/v06_输入归一化/v06-r01_20261108_seed000/preflight/pytest.log)。

## 5. 训练实际完成情况与全过程趋势

- 完成 **20000 / 20000 环境步、40 updates**，训练命令退出码 0，墙钟约 23.62 秒。
- **70 个完整回合，共 19740 步**；最后还有一个 260 步的未结束回合，没有冒充完整回合计入 CSV。
- 完整回合全部 282 步敌方攻击成功：训练捕获 **0/70**，timeout 0。
- 训练过程中没有 NaN/Inf；所有 CSV 数值、保存的网络张量和评估轨迹数值检查通过。
- `best.pt` 在 **1024 步 / update 2** 保存；`final.pt` 在 20000 步正常保存。
  原始 best 选择规则未改：最近训练捕获率优先，再按训练窗口平均回报，且保存的是更新后策略。
  它不保证是确定性评估最优。

比较首 10 与末 10 个完整回合；loss/entropy 则比较首/末 10 次更新：

| 指标 | v05-r01 原始输入：首 → 末 | v06-r01 缩放输入：首 → 末 |
| --- | ---: | ---: |
| 平均 episode reward | -219.297 → -235.074 | **-222.621 → -253.510** |
| 平均 minimum distance，m | 44.001 → 57.345 | **51.035 → 85.845** |
| 平均 episode length | 282 → 282 | 282 → 282 |
| 训练捕获率 | 始终 0% | 始终 0% |
| 平均碰撞步数 | 13.3 → 8.1 | 12.8 → 8.7 |
| Actor loss | -0.00718 → -0.01028 | -0.00279 → -0.00244 |
| Value loss | 181.364 → 46.079 | 236.934 → 84.513 |
| Entropy | 1.79867 → 1.73337 | 1.77853 → 1.68759 |
| Approx KL | 0.00406 → 0.00509 | 0.00219 → 0.00227 |
| Clip fraction | 0.04167 → 0.06068 | 0.01886 → 0.02290 |
| Actor gradient norm，裁剪前 | 1.539 → 1.980 | 0.442 → 0.612 |
| Critic gradient norm，裁剪前 | 447.980 → 414.004 | 39.911 → 47.468 |

v06 的全部 70 回合平均回报 -261.817，对照 -261.461，未显示整体提高。
训练中最近一次达 15.448 m，对照 16.724 m；这是带探索的某一回合，不是确定性模型成功。
回报和距离均有波动，中段变差、后段局部回升，但没有形成持续的任务学习趋势。
Value loss 下降表示对本次更新目标的拟合变化，不能当作任务进步；两个 run 的采样分布和 target 也已不同。
梯度尺度更小、没有数值发散，但确定性三机动作更同质；不能仅凭有限数值就宣称没有行为退化。

## 6. 同预算确定性评估

每个 best/final 用完整的 10 回合评估，未使用新环境或降低难度：

| 模型 | checkpoint 步数 | 捕获率 | 平均 reward | 最近距离 m | 终局最近距离 m | 每回合碰撞步数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v05-r01 best | 1024 | 0/10 | -121.906 | **20.835** | 22.936 | 0 |
| v06-r01 best | 1024 | 0/10 | -212.435 | 109.732 | 113.464 | 0 |
| v05-r01 final | 20000 | 0/10 | -762.756 | 72.433 | 158.785 | 101 |
| v06-r01 final | 20000 | 0/10 | -277.599 | 100.499 | 178.629 | 0 |

全部平均长度 282 步，敌方攻击成功率 100%，timeout 0%；成功捕获时间均为 null。
敌机终局距基地约 8.261 m。新 best/final 的首回合 hard clipping 都为 0。
固定初始化下，同一模型 10 回合指标完全相同；这些重复用于流程/复现检查，不是 10 个独立场景或泛化证据。

**Final 回报改善不能解释为更会拦截**：旧 final 的碰撞罚分为 -505，新 final 为 0；
而 active-progress 累计由 -152.116 变差至 -171.959，最近距离也从 72.433 变差至 100.499 m。
两者 survival 都是 -5.64、terminal 都是 -100，因此总回报净提高 485.156，主要来自碰撞分消失。

数据：[新 best 汇总](../outputs/v06_输入归一化/v06-r01_20261108_seed000/evaluation/best/evaluation_summary.json)、
[新 final 汇总](../outputs/v06_输入归一化/v06-r01_20261108_seed000/evaluation/final/evaluation_summary.json)、
[逐项对照 JSON](../outputs/v06_输入归一化/v06-r01_20261108_seed000/comparison_summary.json)。

## 7. 周期 checkpoint 筛查与轨迹行为

额外对新 run 已保存的所有 checkpoint 做一次确定性筛查，**没有额外训练**：
6 个文件（best、final、4 个 periodic）对应 5 个不同 Actor 策略；update 40 与 final 相同。
全部 5 个策略均失败。按最小距离选出的是 final：100.499 m；按未折扣回报选出的是
update 30 / 15360 步：-206.221，最近 107.250 m。训练 best 没被替换。
[全部筛查 CSV](../outputs/v06_输入归一化/v06-r01_20261108_seed000/checkpoint_screening/checkpoints.csv)、
[选择规则与摘要](../outputs/v06_输入归一化/v06-r01_20261108_seed000/checkpoint_screening/screening_summary.json)。

新 best 在 26.9 s 最近 109.732 m，三机位置 y 约 -98 m；新 final 在 23.1 s 最近 100.499 m，
三机 y 约 95–100 m。敌机始终沿 y=0 直飞基地，三机的主要失败是共同偏离其航线，没有形成有效接触。
两种新策略的动作分量 `|action|>0.95` 比例均 0，未出现动作饱和或贴 workspace 边界。

两两动作方向平均 cosine：新 best 约 0.999999、新 final 约 0.99949–0.99997；
旧 best 为 0.92549–0.94065。只描述本次轨迹上的动作相似性，不能单独用来评价所有协同质量。
结合轨迹，本次没有明显角色分工或有效多机协同证据，反而更接近三机共同偏航。
共享参数、相近初始观测仍可能产生同质动作；输入缩放本身没有添加角色或通信机制。

历史 200k 的 8.629 m 近距模型属于 v05-r02，与本次预算不同，仍是历史最接近的已筛查
MAPPO 确定性轨迹；不能用本次 100.499 m 结果替代它，也不能把那次未捕获当作成功。

## 8. Critic 诊断与结论边界

另对保存的确定性轨迹计算完整 Monte Carlo discounted return-to-go，gamma=0.99：
新 best 的 V(s0)=-8.627，轨迹 G0=-17.270；新 final 的 V(s0)=-31.478，G0=-9.578。
精确数值保存在 comparison_summary.json；这不是训练日志中的 GAE/lambda-return，
也不是随机训练策略的无偏 value 评估，不能据此宣称归一化已经修好或损坏 Critic。
训练历史没有逐 transition 保存实际 lambda-return，无法事后凭空补造其解释方差。

本次支持的结论：固定尺度缩放消除了初态第一层的大面积饱和，并改变了梯度尺度；
**20k、seed 0 的实际任务表现没有改善，Baseline 仍未学会拦截**。
本次不支持：归一化必然无效、奖励一定是唯一原因、只抬高某失败轨迹回报就能修好学习，
或应立即改变物理难度。20k 单 seed 预算不足以作这些判断。

后续最有价值的检查是：先离线核对成功/近距错过/静止/随机轨迹的奖励信号，
再单独设计是否需要 Actor-only / Critic-only 或更长预算的对照；这些都尚未执行，需另定实验。

## 9. 图表与复现

全部 14 张图在 [plots/](../outputs/v06_输入归一化/v06-r01_20261108_seed000/plots/)：
保留 CSV 原始曲线，平滑线明确使用 **trailing 10 个完整回合 / 10 updates**，不足 10 点不画均值。
四张 3D 图直接绘制已保存的轨迹，没有重新仿真或手工补造数据。

- [Reward vs steps](../outputs/v06_输入归一化/v06-r01_20261108_seed000/plots/reward_vs_steps.png)
- [Capture vs steps](../outputs/v06_输入归一化/v06-r01_20261108_seed000/plots/capture_vs_steps.png)
- [Episode length vs steps](../outputs/v06_输入归一化/v06-r01_20261108_seed000/plots/length_vs_steps.png)
- [Minimum distance vs steps](../outputs/v06_输入归一化/v06-r01_20261108_seed000/plots/min_distance_vs_steps.png)
- [Actor/value loss](../outputs/v06_输入归一化/v06-r01_20261108_seed000/plots/actor_value_loss_vs_updates.png)
- [Entropy](../outputs/v06_输入归一化/v06-r01_20261108_seed000/plots/entropy_vs_updates.png)
- [归一化 best 3D](../outputs/v06_输入归一化/v06-r01_20261108_seed000/plots/normalized_best_trajectory_3d.png)
- [归一化 final 3D](../outputs/v06_输入归一化/v06-r01_20261108_seed000/plots/normalized_final_trajectory_3d.png)

本轮实际执行的训练与评估入口如下；已完成的输出不可覆盖。
若将来另获授权重跑，应先通过 experiment_index 获取新 run 目录，而不是照抄本次输出路径：

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q --basetemp tmp/norm_full_01
.\.venv\Scripts\python.exe -B -u -m scripts.train_mappo --config configs/mappo_normalized.yaml
.\.venv\Scripts\python.exe -B -m scripts.evaluate_mappo --checkpoint outputs/v06_输入归一化/v06-r01_20261108_seed000/checkpoints/best.pt --episodes 10 --seed 1000 --output-dir outputs/v06_输入归一化/v06-r01_20261108_seed000/evaluation/best
.\.venv\Scripts\python.exe -B -m scripts.evaluate_mappo --checkpoint outputs/v06_输入归一化/v06-r01_20261108_seed000/checkpoints/final.pt --episodes 10 --seed 1000 --output-dir outputs/v06_输入归一化/v06-r01_20261108_seed000/evaluation/final
```

分析可独立复算，不训练、不更改旧数据：

```powershell
.\.venv\Scripts\python.exe -B -m scripts.replay_normalization_analysis --stage 20k --run
.\.venv\Scripts\python.exe -B -m scripts.experiment_index --check
```

## 10. 本轮文件变化

- 新增配置 `configs/mappo_normalized.yaml`；新增归一化测试文件，扩展版本管理测试。
- 只扩展 `scripts/experiment_index.py` 的版本配置和学习指纹管理，没有改 MAPPO 算法。
- 更新 README、BASELINE_SPEC、实验注册表和两个生成导航；新增本报告。
- 旧奖励提案的暂存编号 v06 顺延 v07，旧诊断报告附编号说明；奖励提案仍未实现。
- 新训练、模型、评估、筛查、曲线、分析脚本和校验均位于独立的 `outputs/v06_输入归一化/v06-r01_20261108_seed000/`。
- 核心环境、动力学、奖励、网络和训练脚本保持训练前内容；最终保护检查见
  [verification.json](../outputs/v06_输入归一化/v06-r01_20261108_seed000/verification.json)。
  20 个受保护核心文件也全部匹配历史长训练的冻结哈希；1890 个既有输出文件中，
  1889 个内容未变，唯一变化是预期更新的 `outputs/README.md` 生成导航。259 个本地文档链接校验通过。
