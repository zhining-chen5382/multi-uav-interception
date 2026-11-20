# v06-r01：归一化模型从 20k final 续训至总计 100k

日期：2026-11-08。**已完成总计 100000 环境步。Final 的最近距离由 100.50 m 改善至 42.89 m，
但 best/final 确定性评估仍均未捕获，尚未形成稳定拦截策略。**

## 1. 实验目的与设置

根据用户要求，从上一轮归一化 **20k final** 继续训练，检验增加训练量能否把主动接近转化为成功拦截。
本次追加 **80000 步**，不是重新从零训练，也不是额外训练 100000 步。
仍属于同一个 **v06-r01**，没有新增 seed、搜索配置或更换奖励。

与冻结的 20k 有效配置相比，只有：

```text
training.total_env_steps: 20000 -> 100000
output_dir: 原20k目录 -> 原目录/continuation_100k
```

配置为 [mappo_normalized_100k.yaml](../configs/mappo_normalized_100k.yaml)，实际保存于
[training_config.yaml](../outputs/v06_输入归一化/v06-r01_20261108_seed000/continuation_100k/training_config.yaml)。
Actor observation 和 Critic global state 固定缩放仍开启，reward_scale=1。
原始环境、reward、21D/24D 输入、3D acceleration action、网络、初始化及敌机策略均未改动。

关键参数继续保持：seed=0、CPU/1 thread、rollout=512、Actor=[128,128]、Critic=[256,256]、Tanh；
actor/critic LR=0.0003/0.001、Adam eps=1e-5、gamma=0.99、GAE lambda=0.95、clip=0.2、epochs=4、
minibatch=64、entropy coefficient=0.01、value coefficient=0.5、max gradient norm=0.5、advantage normalization=true。
初始 log_std=-0.5、范围[-5,2]，其他初始化增益、日志与 checkpoint 间隔也保持原值。
完整环境参数见 [environment_config.yaml](../outputs/v06_输入归一化/v06-r01_20261108_seed000/continuation_100k/environment_config.yaml)
及 [20k 报告](NORMALIZATION_20K_REPORT.md)。

## 2. 代码版本、恢复过程与测试

- HEAD：`c6fe7e7c64e62425c079960b3895615b6f274113`；工作区 dirty，不能把 HEAD 当作完整工作区快照。
- Python 3.8.8、NumPy 1.24.4、Torch 2.4.1+cpu。
- 续训实际开始 UTC：`2026-11-08T07:23:40.923581+00:00`，北京时间 15:23:40。
- [训练前计划](../outputs/v06_输入归一化/v06-r01_20261108_seed000/continuation_100k/preflight/continuation_plan.json)
  锁定源 final SHA256、60 个已有 20k 文件以及 20 个核心文件的哈希。
- 完整 pytest：**246 passed in 19.85s**，见
  [日志](../outputs/v06_输入归一化/v06-r01_20261108_seed000/continuation_100k/preflight/pytest.log)。

原训练 CLI 尚无通用 resume 参数，因此本次使用输出目录内的
[续训封装](../outputs/v06_输入归一化/v06-r01_20261108_seed000/continuation_100k/resume_to_100k.py)。
它在输出目录生成恢复初始化的训练副本，**整个 sampling/GAE/PPO while loop 与原训练脚本 AST 完全一致**，
没有修改核心算法或原 `scripts/train_mappo.py`。

恢复内容包括：Actor/Critic 参数和 scale buffers、两套 Adam state、Python/NumPy/Torch RNG、
环境位置/速度/加速度、环境 RNG、step count、未结束回合累计量、最近 50 回合窗口和 best score。
所有对象构造完成后才恢复 RNG；不调用 reset 或重新设 seed。旧 checkpoint 未保存的边界计数，
通过追捕机可达界与攻击机 260 步精确重积分证明为零，再恢复。

恢复自检：独立恢复两次，每次执行 3 步带探索动作；动作、latent、log probability、value、
状态、加速度、奖励、info 和 RNG 完全相同，没有执行优化器更新。正式采样前再次核对恢复状态。
见 [self-test](../outputs/v06_输入归一化/v06-r01_20261108_seed000/continuation_100k/preflight/restore_self_test.json)、
[正式恢复审计](../outputs/v06_输入归一化/v06-r01_20261108_seed000/continuation_100k/preflight/restoration_audit.json)。

原20k最后一次32步 rollout 已完成更新；续训 buffer 从空开始，不重复这32步。
新增 156 次 512 步更新和一次 128 步更新，因此累计 **40+157=197 updates**。
从零不间断100k通常为196次，本实验不能伪装成完全相同的更新日程。

## 3. 实际训练结果与全过程趋势

- 总计 **100000 步、197 updates、354 完整回合**。
- 完整回合共 99738 步，结尾未结束回合 262 步；二者相加恰为100000。
- 追加80000步，实际续训约 **96.33秒**；正常退出，final 与 best 均正常保存。
- 训练捕获 **2/354=0.565%**，最后50个完整回合捕获率 **2%**，训练50回合窗口最高也为2%。
- 捕获以外的完整回合均为敌方攻击成功，无 timeout；没有 NaN/Inf。

两次训练捕获来自随机探索策略，而非保存模型的确定性策略：

| Episode | 累计环境步数 | 捕获时间 | 最近距离 | 回合 reward | 碰撞步数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 88 | 24777 | 24.3 s | 3.747 m | +96.062 | 16 |
| 325 | 91560 | 23.1 s | 4.609 m | +89.440 | 22 |

捕获触发了既定 `<5m` 条件，但两回合均有我方不安全间距，不能描述为安全协同拦截。
两次成功间隔约66.8k步，没有形成持续成功的训练阶段。训练只记录回合指标，
没有逐步保存这两个成功回合的完整轨迹，不能为它们伪造仿真视频或事后重建路径。

首50与末50个完整回合 / 更新的对照：

| 指标 | 首50 | 末50 |
| --- | ---: | ---: |
| 平均 episode reward | -267.641 | -252.748 |
| 平均 minimum distance | 91.004 m | 48.252 m |
| 平均碰撞步数 | 12.18 | 11.66 |
| 捕获率 | 0% | 2% |
| Actor loss | -0.003177 | -0.003339 |
| Value loss | 182.613 | 69.628 |
| Entropy | 1.72024 | 1.78700 |
| Approx KL | 0.002793 | 0.002525 |
| Clip fraction | 0.02911 | 0.02503 |

中间20k–50k阶段平均回报约-326.54，50k–100k约-251.28，存在后段改善，但不是单调学习。
首20k的平均最近距离90.20m，后80k结束的回合均值77.31m，末50回合进一步降至48.25m。
回合按结束步数归属阶段；第71回合跨越20k边界，不能把其完整回合长度误当作全部新增步数。

Value loss有波动，范围3.48–486.82；entropy范围1.512–1.862，没有数值爆炸或熵归零。
这支持数值运行稳定，但不证明行为已收敛。稀疏的探索成功没有转化为可靠的确定性捕获。

## 4. Best / final 确定性评估

使用 checkpoint 自带环境快照，不降低难度；100k best/final 各 **50回合**，seeds1000–1049。
固定初始化下，同一模型的50次指标完全相同，是重复复现，不代表50个独立场景或泛化测试。

| 模型 | 权重环境步数 | 捕获 | Reward | 最近距离 | 终局最近距离 | 每回合碰撞步数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 冻结20k best | 1024 | 0/10 | -212.435 | 109.732 m | 113.464 m | 0 |
| 冻结20k final | 20000 | 0/10 | -277.599 | 100.499 m | 178.629 m | 0 |
| 100k training best | 91680 | 0/50 | -414.306 | 85.634 m | 90.335 m | 45 |
| **100k final** | **100000** | **0/50** | **-262.843** | **42.891 m** | **163.872 m** | **0** |

全部平均282步，敌方攻击成功率100%，timeout0%，成功捕获时间null。
100k best/final 的首条完整轨迹均无 hard boundary clipping、动作分量 `|action|>0.95` 比例均为0。

**本次final相较20k final确有更有效的接近**：最近距离减少57.61m，约57.3%，
两者都没有碰撞，回报改善14.76分，不能归结为碰撞罚分消失。
它也比100k training best更接近目标，且没有后者的45步碰撞。
不过final仍未进入5m捕获区，因此“更好的接近”与“学会任务”应分别记录。

`best.pt` 的选择标准保持不变：训练最近50回合捕获率优先，再按窗口平均回报；
它不是确定性验证最优。100k best来自update180/91680步，紧随第二次随机训练捕获，
窗口为2%捕获、平均回报-225.395；评估时得到的是更新后的确定性策略，不能用训练窗口成绩代替其表现。

数据：[best汇总](../outputs/v06_输入归一化/v06-r01_20261108_seed000/continuation_100k/evaluation/best/evaluation_summary.json)、
[final汇总](../outputs/v06_输入归一化/v06-r01_20261108_seed000/continuation_100k/evaluation/final/evaluation_summary.json)。

## 5. 保存模型筛查与轨迹观察

额外对整条20k→100k链上已保存的23个checkpoint文件按Actor参数和scale去重，
共21个不同策略，每个运行1个确定性回合；**全部未捕获**，没有额外训练。
最小距离和最高无折扣回报都来自 **update140 / 71200步**：最近21.662m，回报-120.632，碰撞0。
该策略主要留在基地后方，最小距离出现在敌机已经攻击成功的终局，
不能仅因距离/总分更好就将它描述为更成功的主动拦截策略。
原生训练best未替换；筛查结果独立保存在
[screening_summary.json](../outputs/v06_输入归一化/v06-r01_20261108_seed000/continuation_100k/checkpoint_screening/screening_summary.json)。

Final在23.4s由P0达到最近42.891m。当时：

```text
P0 = [99.088, -26.579, 42.718] m
E  = [103.837, 0, 9.391] m
P0-E = [-4.749, -26.579, 33.326] m
```

沿x方向已经相当接近，但横向约26.6m、高度约33.3m的误差使其未接触目标。
随后敌机飞向基地、我方继续外飞，终局最近距离增至163.872m。
这符合“在尝试接近但方向仍不准确”，还不能将误差都归因于训练量。

三机动作仍高度相似：final两两动作平均cosine约0.9966–0.9993；
best约0.9998–1.0000。没有观察到明确角色分工或有效夹击；同向运动本身不证明协同成功。
Best主要偏至正y和高z，且有持续不安全间距；final虽改善接近，仍需纠正横向和高度。
数字见 [behavior_summary.json](../outputs/v06_输入归一化/v06-r01_20261108_seed000/continuation_100k/behavior_summary.json)。

## 6. 判断与限制

增加训练量产生了一些可见进展：两次随机训练捕获、后段距离下降、final的主动接近改善。
但100k final/best和所有已保存策略的确定性评估仍失败，**当前不能认为已经学会稳定拦截**。
数值稳定与行为成功是不同结论。仍是单seed、固定敌机直线进攻与固定初态，没有泛化证据。

这轮没有改reward、obs、dynamics或捕获条件，也没有因结果不佳续到更大预算。
历史未归一化100k评估JSON仅作补充记录，详见analysis_summary.json；
其root权重后来已变为200k，分析没有将它们误当100k模型。
两个实验还存在197与196次更新的日程差异，因此不作严格的单因素因果结论。

## 7. 保存位置、运行方式与文件变化

- 新增配置 `configs/mappo_normalized_100k.yaml`，没有覆盖20k配置。
- 核心环境、reward、网络和训练脚本保持不变；续训封装、自检和分析脚本只放在新输出子目录。
- 更新本报告、README、BASELINE_SPEC、注册表与生成索引。同一run的20k/100k阶段分开列出。
- 60个旧20k文件和20个核心文件均保持原哈希，最终记录见
  [verification.json](../outputs/v06_输入归一化/v06-r01_20261108_seed000/continuation_100k/verification.json)。
- 本轮没有Git add、commit、merge或push。

本轮实跑命令；`--run`只能对未开始的续训目录执行一次，不要覆盖现有完成结果：

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q --basetemp tmp/norm_100k_full_01
.\.venv\Scripts\python.exe -B outputs/v06_输入归一化/v06-r01_20261108_seed000/continuation_100k/resume_to_100k.py --self-test
.\.venv\Scripts\python.exe -B -u outputs/v06_输入归一化/v06-r01_20261108_seed000/continuation_100k/resume_to_100k.py --run
```

完整控制台及实际退出码保存在 [training_console.log](../outputs/v06_输入归一化/v06-r01_20261108_seed000/continuation_100k/training_console.log)
和 [training_execution.json](../outputs/v06_输入归一化/v06-r01_20261108_seed000/continuation_100k/training_execution.json)。
评估使用原 `scripts.evaluate_mappo`，分别指定best/final、`--episodes 50 --seed 1000`，输出到新子目录的evaluation中。

13张实际数据图保存在 [plots/](../outputs/v06_输入归一化/v06-r01_20261108_seed000/continuation_100k/plots/)；
训练曲线保留原始点，均值线明确为trailing10回合或10updates，并标出20k续训边界。

- [Reward](../outputs/v06_输入归一化/v06-r01_20261108_seed000/continuation_100k/plots/reward_vs_steps.png)
- [Capture](../outputs/v06_输入归一化/v06-r01_20261108_seed000/continuation_100k/plots/capture_vs_steps.png)
- [Minimum distance](../outputs/v06_输入归一化/v06-r01_20261108_seed000/continuation_100k/plots/min_distance_vs_steps.png)
- [Actor/value loss](../outputs/v06_输入归一化/v06-r01_20261108_seed000/continuation_100k/plots/actor_value_loss_vs_updates.png)
- [Final 3D轨迹](../outputs/v06_输入归一化/v06-r01_20261108_seed000/continuation_100k/plots/final_trajectory_3d.png)
- [Best 3D轨迹](../outputs/v06_输入归一化/v06-r01_20261108_seed000/continuation_100k/plots/best_trajectory_3d.png)
- [完整机器可读分析](../outputs/v06_输入归一化/v06-r01_20261108_seed000/continuation_100k/analysis_summary.json)
