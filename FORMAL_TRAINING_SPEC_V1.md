# 正式训练规范 v1.0

本文汇总用户确认的独立1v1 PPO正式实验协议，2026-11-16补齐16维归一化和加速度定义。生产配置为 `configs/interception_1v1.yaml`、`configs/ppo_formal_v1.yaml`，入口为 `scripts/run_formal_experiment.py`。

## 追加预算决定，2026-11-16

用户在初始300k实验进行期间要求五个Seed都补到累计500k。保留每个Seed已经执行的300k训练，并从各自300k `final.pt` 完整继承网络、Adam、Scheduler、环境和RNG，追加200k。原300k范围内的学习率履历保留，300k之后Actor LR固定3e-5、Critic LR固定1e-4。奖励、观测、动力学、网络及其他PPO参数继续使用确认值。

最终累计预算为每Seed500,000、五次合计2,500,000，追加量合计1,000,000。新配置为 `configs/ppo_formal_v1_500k.yaml`，入口为 `scripts/run_formal_500k.py`。源300k结果保留，500k续训使用独立目录，并在500k全部完成后按同一固定协议测试。下文300k描述对应初始阶段与学习率衰减horizon。

## 1. 场景与物理模型

保留原中线P0，初始位置[1.5,0,1]、速度零。基地[0,0,0]，敌机0°初始位置[497.49371855331,0,50]、速度零。每个episode独立采样U(-30°, +30°)，以基地为中心绕Z轴旋转，+X为0°、逆时针为正，保持高度和距基地距离。

复用原Direct Attacker、物理动作映射及动力学。双方最大水平速度20m/s，最大爬升/下降速度8/6m/s，最大水平/垂直加速度3/2m/s²，最大三维jerk为5m/s³。dt=0.1s，最多500步即50s，工作空间各轴[-750,+750]m，保留数值硬裁剪。三维无障碍，理想化状态感知。

## 2. 16维观测与固定物理尺度

所有向量采用世界坐标系。Actor与Critic使用相同观测，严格按下表排列。

| 索引 | 字段 | 除数 |
| --- | --- | --- |
| 0–2 | 自身速度XYZ | [20,20,8] |
| 3–5 | 自身applied acceleration XYZ | [3,3,2] |
| 6–8 | 敌机位置减自身位置XYZ | [1500,1500,1500] |
| 9–11 | 敌机速度减自身速度XYZ | [40,40,14] |
| 12–14 | 敌机位置减基地位置XYZ | [500,500,500] |
| 15 | 剩余时间除以最大回合时间 | 1 |

逐维使用 `x_norm=x/scale`。不做均值中心化、不使用动态归一化、不静默截断。超尺度时记录字段、原值及归一化值。训练、验证、测试使用完全相同的参数及函数，并保存在Checkpoint中。

第3–5维读取环境维护的 `accelerations[0]`，即jerk限制后的applied acceleration。该状态作为下一步动力学jerk限制的历史加速度。速度限幅后的 `(v_next-v_previous)/dt` 保留为诊断量。正式入口要求 `normalization.acceleration_kind=applied`。

## 3. 网络与动作分布

Actor/Critic为独立网络，输入16，隐藏层128×128，Tanh激活。Actor输出三维Gaussian mean，Critic输出标量value。沿用原正交初始化，隐藏层gain为√2，Actor/Critic输出gain为0.01/1，bias为零。

三个动作维度使用独立、与状态无关的可学习log_std，初始std=0.5，log_std范围[-5,0]。训练随机采样，保存pre-tanh动作，Tanh映射到[-1,1]。Log probability包含稳定的Jacobian修正，熵使用变换后分布的可重参数化MC估计。确定性评估使用tanh(mean)。

## 4. PPO与Adam

| 参数 | 已确认设置 |
| --- | --- |
| Actor LR | 3e-4线性衰减到3e-5 |
| Critic LR | 1e-3线性衰减到1e-4 |
| Scheduler | 每Run的300k aggregate environment steps |
| Adam | betas=(0.9,0.999)，eps=1e-5，weight_decay=0，AMSGrad=False |
| 环境/rollout | 8个独立环境，每环境512步，共4096样本 |
| Minibatch / epochs | 256 / 最多4 |
| Gamma / GAE Lambda | 0.99 / 0.95 |
| Clip / Target KL | 0.2 / 0.015 |
| Entropy Coefficient | 固定0.01 |
| Value Loss Coefficient | 0.5 |
| Gradient Clipping | max norm=0.5 |
| Advantage Normalization | rollout归一化，epsilon=1e-8 |
| Value Clipping / Normalization | 关闭 |

每minibatch更新后计算old-policy到current-policy的潜在对角Gaussian解析KL，严格大于0.015时停止本rollout剩余更新。记录每minibatch KL及每epoch clip fraction。沿用既有Lv=0.5*mean((V-return)²)，Critic objective=0.5*Lv。不手动衰减std或entropy。

## 5. 奖励、终止与Bootstrap

距离取本步完成后的三维敌我距离。

```text
r = -0.01 * distance * dt - 0.2 * dt
    +500 * I(capture) -100 * I(defense failure)
```

仅删除我方机间碰撞惩罚。捕获采用敌我距离严格小于5m，失守采用敌机距基地严格小于10m，同步触发时捕获优先。捕获、失守和50s超时均为真正终止，GAE不bootstrap。Rollout暂时结束但episode继续时正常bootstrap。

## 6. 五次训练、验证与模型选择

Seeds依次为1001、1002、1003、1004、1005，顺序执行。每Run从零初始化网络、Adam、Scheduler与RNG，训练300,000 aggregate environment steps，总预算1,500,000。8环境采用同步向量采样与批量策略推理，每环境拥有独立状态及局部RNG。

每10k步验证相同100个线性均匀角度，含±30°端点。验证保留未完成rollout，记录实际采样步数、模型最近更新步数和pending样本。连续三次捕获率≥80%仅标记稳定性，继续完整300k。

Best依次比较捕获率高、防御失败率低、成功平均拦截时间短、平均最小距离小，完全相同保留已有Best。验证与测试root seed分别为2001、3001。

## 7. 最终测试与统计

五次训练全部完成后，冻结每Run验证Best，在相同500个新均匀随机角度上确定性测试。测试场景与训练、验证独立，测试期间不更新参数。

报告每Seed捕获率、防御失败率、超时率、平均成功拦截时间、平均最小距离及回报，报告五次均值、样本标准差和95%均值区间。跨运行采用t(4)区间，每模型另记录Wilson捕获率区间。

角度分区为[-30,-20)、[-20,-10)、[-10,0)、[0,10)、[10,20)、[20,30]，每个场景只计一次。整体达标要求五模型平均测试捕获率≥80%。

## 8. 验收、恢复与交付

开训前通过环境/观测/奖励/终止、分布/熵/KL、8环境独立性、PPO早停、完整恢复、步数边界、验证时点、完整pytest和生产设置Smoke Test。

保存网络、双Adam、Scheduler、log_std、全局及环境RNG、物理状态、当前观测、未完成rollout、回合累积量与验证选择状态。保存完整配置、Git及软件版本、实际源码快照和每回合角度。

每次训练结束自动生成全部原始CSV和训练曲线，固定0°逐Checkpoint XY轨迹演化图及CSV，浅到深表示实际训练步数，生成同场景best/final完整视频、manifest和全帧解码校验。保存各Run报告及五Run综合报告。所有结果使用独立目录。

## 9. 独立Reward A/B实验，2026-11-17

用户授权在同一1v1 PPO协议下开展受控奖励对照。A组复用`outputs/formal_ppo_v1/v14_单机PPO_原始奖励/01_从零训练至300k/`中1001、1002、1003各自300k历史结果。B组使用`configs/ppo_reward_ab.yaml`和`configs/interception_1v1_best_so_far.yaml`，三个Seed各从零训练300000 aggregate steps，8环境，每10k用原100角度集验证。最终Best和Final使用原500个测试角度分别评估。实际输出为`outputs/formal_ppo_v1/v15_历史最小距离奖励/01_从零训练至300k/`。

B组只在原四项奖励上增加

```text
best_distance_so_far(t) = min(d(0), ..., d(t))
near_shaping(t) = 100 * (exp(-best_distance_so_far(t+1) / 20)
                         - exp(-best_distance_so_far(t) / 20))
```

`best_distance_so_far`为环境正式状态，复用原`episode_min_distance`作为单一存储来源。reset从实际初始距离初始化，Checkpoint显式保存，B组恢复必须存在且与存储的历史最小距离一致。远离、等距、再次靠近但未超过旧纪录时新增项为0。只有更小纪录才产生新区间奖励。

环境配置中的`reward.best_so_far_shaping`显式包含`length_scale_m=20.0`和`beta=100.0`。未配置时继续使用A组原四项奖励及原快照格式。B组在每回合CSV和评估轨迹中分别记录原奖励回报、near累计量及best-distance状态。

Actor/Critic输入仍为16维，历史最小距离保留在环境内部。奖励由历史状态决定，这一历史信息没有追加到观测中。学习率、可学习std、PPO、网络、归一化、初始化、动力学、终止和选模排序沿用本规范300k协议。

本轮范围为三个Seed各300k。是否开展后续完整五Seed实验，依据本轮全部三个Seed的训练、验证、测试及退化情况判断。A/B使用同Seed同300k预算，训练捕获率由各自实际完整回合重新计算。两个奖励定义下同时报告任务指标和原四项共同回报。

## 10. 独立线段成本实验，2026-11-17

用户指定直接尝试k=0.1。实验配置为`configs/ppo_line_cost_k0p1.yaml`与`configs/interception_1v1_line_cost_k0p1.yaml`，入口为`scripts/run_line_cost_experiment.py`。

本轮采用上一轮B组的三个Seed 1001、1002、1003，各从零训练300000 aggregate steps，8环境，±30°均匀来袭，每10k使用相同100场景验证。Actor/Critic为16维输入、128×128隐藏层，gamma=0.99、GAE lambda=0.95，学习率与可学习std采用第3、4节参数。此规模由上一轮实验沿用并在执行前说明。

新增项为`line_cost=-k*d_segment_post*dt`。`d_segment_post`是本步积分后我方位置到基地—敌机三维闭线段的最短距离，投影比例截断在[0,1]。敌机与基地重合时退化为点距离。该项在原四项与β100、λ20m历史最小距离shaping基础上累加，捕获和失守终止步计入成本。配置字段为`reward.line_cost_coefficient`。

训练和评估分别记录原四项、near、line与总回报，环境快照保存累计line分量。恢复完整保留pending rollout。输出为`outputs/formal_ppo_v1/v16_线段偏离成本_k0p1/01_从零训练至300k/`，每Seed交付Best/Final完整0°视频与逐Checkpoint轨迹演化。

受控比较复用上一轮B组300k结果与相同500角度场景。该场景集已参与奖励离线标定，本次测试用于诊断对照。基地附近停留、最大前出半径、路径长度、正负方向捕获率与退化情况均列入报告。初始及历史Checkpoint按真实来源保存。

## 11. 主动接近near配比训练，2026-11-18

用户授权训练离线标定的两组候选，β100/k0.01和β200/k0.01，λ20m。沿用最近三个Seed协议，每组1001、1002、1003各从零300000 aggregate steps，8环境，共六次训练1800000步，顺序执行。独立输出为`outputs/formal_ppo_v1/v17_主动接近归因_k0p01/01_主动接近配比训练/`。

环境配置的`reward.best_so_far_shaping.mode: motion_attributed`启用归因。令实际历史最小距离为b_t，我方位置p_t，敌机位置a_t。

```text
d_cf = norm(a_(t+1) - p_t)
b_cf = min(b_t, d_cf)
b_next = min(b_t, norm(a_(t+1) - p_(t+1)))
near = beta * max(0, exp(-b_next / 20) - exp(-b_cf / 20))
line = -0.01 * d_segment_post * dt
```

该归因衡量实际位移，包括惯性位移。静止、未突破纪录及未优于冻结位置反事实时near为0。历史最小距离取真实轨迹，绕回已有距离区间不重复获奖。缺少mode的旧配置沿用既有历史最小距离差分。

原四项奖励、16维观测、网络、动力学、初态、动作、归一化、PPO、学习率及std沿用本规范。保持+500捕获、−100失守与捕获优先。每10k评估相同100角度，Best与Final各测试相同500角度。场景已参与离线标定，测试用于诊断对照。

Checkpoint保存实际best-distance、near校正前后累计与当前诊断。生产配置Smoke验证含pending rollout的逐位恢复。训练、评估及轨迹重建使用一致公式。每Run交付Best/Final视频、解码记录与全部实际Checkpoint演化数据，每组另生成三Seed并排密集XY及分阶段图。

β100失败near累计理论上界约77.88分。β200在标定轨迹中失败累计低于约80分，其理论上界约155.76分，正式训练记录实际失败分项极值。

## 12. 历史k=0.1单模型续训至500k，2026-11-19

用户选择历史k=0.1实验Seed1001的300k `final.pt`，该模型在原500场景中捕获165/500（33%），授权继续到累计500000 aggregate environment steps，实际只新增200000步、只运行Seed1001。源为`outputs/formal_ppo_v1/v16_线段偏离成本_k0p1/01_从零训练至300k/seed_1001/checkpoints/final.pt`，独立输出为`outputs/formal_ppo_v1/v16_线段偏离成本_k0p1/02_Seed1001续训至500k/`。

完整继承Actor、Critic、双Adam、环境、RNG、回合累计量与验证选模状态。保留已执行的300k学习率衰减，后续Actor LR固定3e-5、Critic LR固定1e-4。继续使用旧版未做motion attribution的历史最小距离near（β100、λ20m）和k=0.1线段成本；奖励、观测、动力学、std和其余PPO参数不变。

沿用每10k固定100角度验证与原Best排序，并继承历史Best（150k、验证30%）。Best若未被后续策略超越，允许仍来自300k以前，报告明确注明。源33%模型作为300k基准；续训后的Best/Final用原相同500角度分别评估，交付完整视频、真实checkpoint密集XY演化、CSV与来源核验。此为选中模型的单Seed续训诊断，与多Seed从零训练可靠性实验分别记录。

底层完整恢复校验保留源配置中的三个Seed列表以避免改变研究参数，执行脚本仅显式调用Seed1001，实际执行范围另行记录。配置与日志写入新实验目录，不覆盖原300k数据。
