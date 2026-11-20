# v05 / v06 同为300k：训练、拦截与行为差异对照

日期：2026-11-08。**两个版本均实际完成总计300000环境步。v05动作和轨迹差异明显更大，
v06确定性动作高度同质；但两者best/final各100回合均未捕获，尚未学会稳定拦截。**

## 1. 本次做了什么

用户最初要求200k，随后在实际训练启动前改为两个版本都到总300k。
200k准备目录保留`SUPERSEDED_BY_300K.md`，没有启动该准备方案，不算新run或已完成模型。

| 项目 | v05-r02 | v06-r01 |
| --- | --- | --- |
| 输入缩放 | Actor false / Critic false | Actor true / Critic true |
| 本轮起点 | 既有200k final | 既有100k final |
| 本轮新增环境步数 | 100000 | 200000 |
| 最终总环境步数 | **300000** | **300000** |
| 最终updates | 588 | 588 |
| 完整episodes | 1065 | 1064 |
| 未结束回合步数 | 149 | 252 |
| 累计训练捕获 | 6 / 1065（0.563%） | 4 / 1064（0.376%） |
| 本轮新增训练捕获 | 0 | 2 |
| 本轮续训墙钟 | 约163.5s | 约307.8s |

两条任务并行执行，均从自身final恢复，没有重新初始化，也没有从训练best续训。
原20k/100k/200k产物保持不变，新训练结果写在各run的`continuation_300k/`。
两个版本均未超过300k，没有额外seed或配置搜索，没有Git add/commit/merge/push。

## 2. 环境、配置与代码版本

两个环境快照逐项完全相同：固定初态3追捕机+1 Direct Attacker、3D点防御、基地原点、
敌机从约500m直接飞向基地、capture半径5m、attack半径10m、dt0.1s、最大500步、workspace三轴±750m。
动作仍为每机3D加速度，水平/垂直加速度、速度及jerk限制没有改变。
Reward仍是active progress、按秒存活惩罚、碰撞惩罚和原terminal reward；没有新增角色或包围奖励。

学习设定继续保持：seed0、CPU/1个Torch thread、rollout512、共享Actor128/128 Tanh、
集中式Critic256/256 Tanh、actor/critic LR 0.0003/0.001、gamma0.99、GAE lambda0.95、
PPO clip0.2、4epochs、minibatch64、entropy系数0.01、value系数0.5、gradient clipping0.5、
advantage normalization开启、reward_scale1。所有其余网络初始化参数也保持源checkpoint配置。

新增配置仅将各自源配置的总预算和输出路径改为本轮值：

- [mappo_v05_300k.yaml](../configs/mappo_v05_300k.yaml)
- [mappo_normalized_300k.yaml](../configs/mappo_normalized_300k.yaml)

HEAD为`c6fe7e7c64e62425c079960b3895615b6f274113`，工作区dirty；因此另冻结20个核心文件的SHA256，
并记录1152个旧实验文件哈希。Python3.8.8、NumPy1.24.4、Torch2.4.1+cpu。
完整参数、起点checkpoint SHA与保护清单见
[experiment_plan.json](../outputs/_project_项目核对与管理资料/20261108_v05_v06_300k_comparison/preflight/experiment_plan.json)。

## 3. 恢复、测试与公平性边界

恢复包括Actor/Critic及input_scale buffers、两套Adam moments和step、Python/NumPy/Torch RNG、
环境位置/速度/加速度及RNG、未结束回合累计量、recent50窗口和历史best score。
v05源partial已有52步不安全间距、累计碰撞罚分-260，恢复时保留，没有清零。
缺失的边界计数先由可达界和攻击机原动力学重积分证明为0，再恢复。

两版本各自独立恢复两次、各走3步诊断，动作、logprob、value、状态、奖励和RNG逐项一致；
诊断没有更新参数。正式采样前再次校验。输出目录内的恢复封装沿用生产trainer，
**完整sampling/GAE/PPO while循环AST一致**；核心训练脚本未修改。

Full pytest：**251 passed in19.01s**。测试在用户改预算之前完成，之后仅新增配置和隔离的恢复封装，
被测核心源码未变，未重复消耗测试。日志与出处见
[pytest.log](../outputs/_project_项目核对与管理资料/20261108_v05_v06_300k_comparison/preflight/pytest.log)、
[pytest_provenance.json](../outputs/_project_项目核对与管理资料/20261108_v05_v06_300k_comparison/preflight/pytest_provenance.json)。

**相同588updates不意味着更新日程完全相同**：

| 版本 | 阶段末短rollout | 实际每套Adam各参数step |
| --- | --- | ---: |
| v05 | 100k末160、200k末160、300k末160 | 18756 |
| v06 | 20k末32、100k末128、300k末320 | 18752 |

除输入缩放之外，两个run的阶段边界和随机状态访问也不同，且只有seed0。
因此这是同任务、同预算的实际历史run对照，不能当作完全匹配日程的严格单因素因果实验。

## 4. 统一确定性评估

四组均用原checkpoint自带环境快照、相同初态和seeds1000–1099，**每模型实际100回合**。
没有environment override，未降低任务难度。固定初始化下，同一模型100次轨迹相同；
这些重复检验复现，不代表100个独立场景或泛化成绩。

| 模型 | 权重训练步数 | 捕获率 | 平均reward | 最近距离m | 终局最近距离m | 每回合碰撞步数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v05 training best | 135328 | 0/100 | -510.161 | **16.274** | 201.191 | 42 |
| v05 300k final | 300000 | 0/100 | -432.085 | **75.649** | 333.114 | 0 |
| v06 training best | 189088 | 0/100 | -467.515 | 57.271 | 368.545 | 0 |
| v06 300k final | 300000 | 0/100 | -560.088 | **102.641** | 461.117 | 0 |

全部平均长度282步，敌方攻击成功率100%，timeout0%，成功捕获时间null。
四条保存轨迹均无hard clipping，动作分量`|action|>0.95`比例均0。

Best选择规则没有改：训练recent50捕获率优先，再按窗口平均reward。
v05 best沿用原135328步策略（历史窗口8%捕获），并不是本轮新学出的模型；
v06 best来自189088步（窗口4%捕获），保存的是PPO更新后的模型。
这些训练窗口成绩不能替代其确定性评估结果。

## 5. 三机差异化：你的观察得到了数值支持

平均动作方向余弦以三对无人机、各有效step计算；零向量不参与且另记覆盖率。
本次四模型没有零动作。值接近1表示动作方向相近，但差异更大本身不表示协同更有效。
机间距采用每步更新后位置，reset不计入均值。

| 指标 | v05 best | v06 best | v05 final | v06 final |
| --- | ---: | ---: | ---: | ---: |
| 平均两两动作方向cosine | 0.58015 | 0.9999905 | **0.36204** | **0.9999621** |
| 平均机间距m | 30.122 | 2.525 | **112.869** | **2.594** |
| 终局平均机间距m | 157.679 | 2.694 | 296.105 | 2.656 |
| 每步三机动作分量std后取时间均值[x,y,z] | [.16793,.18400,.10481] | [.000562,.000200,.000233] | [.19669,.19572,.23182] | [.001068,.000315,.000334] |

v05 final的P0/P1/P2：

- 路程分别447.25 / 424.89 / 352.54m；各机最近距离75.65 / 96.34 / 147.11m。
- 最近敌机的step份额65.60% / 10.28% / 24.11%。
- 轨迹明显分散，包含不同的爬升/下降和侧向偏离；没有形成能阻止敌机到基地的分工。

v06 final的P0/P1/P2：

- 路程分别471.664 / 471.699 / 471.719m，几乎相同。
- 各机最近距离103.65 / 105.19 / 102.64m，三机保持约2.6m间距一起偏离目标。
- 最近step份额54.96% / 0% / 45.04%；这种份额不等于个体credit或实际贡献估计。

所以本次可以确认：**v05行为更差异化，v06确定性均值动作明显同质化；两者均无有效协同拦截证据。**
不能将v05三条不同路径称为已学到角色，也不能把v06均值动作相近直接解释为探索熵归零。
目前没有显式通信、角色分配或个体贡献实验。

可能需要后续检验的是：固定尺度将几米的队友几何差异压缩后，是否影响共享Actor的行为区分。
这是待验证解释；本次Actor和Critic一起缩放，不能分离两者作用，也不能据单seed证明因果。

## 6. 全过程趋势与是否值得继续单纯加步数

| 指标 | v05：首50→末50完整回合 | v06：首50→末50完整回合 |
| --- | --- | --- |
| 平均reward | -258.96 → -339.89 | -267.64 → -516.36 |
| 平均最近距离m | 86.32 → 60.74 | 91.00 → 48.34 |
| 平均终局最近距离m | 97.69 → 197.32 | 107.77 → 352.89 |
| 末50捕获率 | 0% | 0% |

v05最后一次训练捕获在151801步，此后148199步无捕获；v06最后一次在177078步，
此后122922步无捕获。双方200k–300k阶段均没有训练捕获。
回合最低距离局部改善，并未转化为接触；更常见的是接近后继续飞远，终局距离与reward恶化。

首10→末10 updates：v05 value loss181.36→37.19、entropy1.799→1.280；
v06 value loss236.93→43.99、entropy1.779→1.522。
所有日志和保存张量有限，未出现NaN/Inf；这只说明数值运行正常。
Loss降低没有带来任务成功；当前证据更支持行为停滞/退化，而非已经收敛到有效拦截。

本轮增加预算没有解决任务，不能认为再加步数必然有效。后续应先定位学习信号与同质化原因，
再决定是否追加预算或进行新的消融；本轮没有实施任何研究修改。

## 7. 周期checkpoint次级筛查

在主best/final对照之外，只筛查本轮两个`continuation_300k/checkpoints`目录，每个不同Actor策略运行1次确定性回合。
该筛查不覆盖全部历史checkpoint，也不替换训练best/final：

| 版本 | 文件/不同策略/捕获 | 本轮目录最近模型 | 最近距离m | 该模型reward | 最高reward模型 |
| --- | --- | --- | ---: | ---: | --- |
| v05 | 21/21/0 | 135328步（沿用的training best） | 16.274 | -510.161 | 204096步，-161.990，最近57.206m |
| v06 | 41/41/0 | 162976步/update320 | 19.428 | -496.375 | 127136步，-207.069，最近107.771m |

旧v05历史158368步的8.629m近距模型仍保留，属于此前筛查结果，不是本次300k final，
也没有因为新模型失败而把它混入主对比来选择性展示。

## 8. 结果入口与复现

本次统一入口：
[20261108_v05_v06_300k_comparison](../outputs/_project_项目核对与管理资料/20261108_v05_v06_300k_comparison/)。

- [完整机器可读分析](../outputs/_project_项目核对与管理资料/20261108_v05_v06_300k_comparison/comparison_summary.json)
- [行为指标CSV](../outputs/_project_项目核对与管理资料/20261108_v05_v06_300k_comparison/behavior_metrics.csv)
- [19张曲线和3D图](../outputs/_project_项目核对与管理资料/20261108_v05_v06_300k_comparison/plots/)
- [Final动作相似度与机间距随时间](../outputs/_project_项目核对与管理资料/20261108_v05_v06_300k_comparison/plots/final_behavior_vs_time.png)
- [Final各机到敌机距离](../outputs/_project_项目核对与管理资料/20261108_v05_v06_300k_comparison/plots/final_per_agent_target_distance_vs_time.png)
- [v05 final完整视频](../outputs/_project_项目核对与管理资料/20261108_v05_v06_300k_comparison/videos/v05/final_full_episode.mp4)
- [v06 final完整视频](../outputs/_project_项目核对与管理资料/20261108_v05_v06_300k_comparison/videos/v06/final_full_episode.mp4)
- [视频帧数、模型SHA核验](../outputs/_project_项目核对与管理资料/20261108_v05_v06_300k_comparison/videos/video_verification.json)
- [周期模型筛查摘要](../outputs/_project_项目核对与管理资料/20261108_v05_v06_300k_comparison/checkpoint_screening/screening_summary.json)
- [旧数据、配置、checkpoint、registry最终核验](../outputs/_project_项目核对与管理资料/20261108_v05_v06_300k_comparison/verification.json)

训练曲线保留CSV原始点，滑动均值明确使用trailing50完整回合或10updates，不补造中间数据。
两段视频直接绘制保存的评估NPZ，包含初态和全部282步，283帧/10fps/28.3秒，未重跑仿真或插值。
主报告和各模型评估均使用同一个固定场景，不作泛化结论。

两个实际续训封装位于各自`continuation_300k/resume_to_300k.py`；
完整命令、退出码、源checkpoint SHA和耗时见对比目录及各续训目录的`training_execution.json`。
运行封装会拒绝覆盖已经开始的训练目录；原始训练、GAE和PPO代码保持不变。
统一评估使用原 `scripts.evaluate_mappo`，四组均`--episodes 100 --seed 1000`。

复算分析（不训练，不修改原始模型与CSV；会重生成本次分析JSON和图表）：

```powershell
.\.venv\Scripts\python.exe -B outputs/_project_项目核对与管理资料/20261108_v05_v06_300k_comparison/analyze_comparison.py
.\.venv\Scripts\python.exe -B -m scripts.experiment_index --check
```

本轮新增两份300k配置、隔离的训练/评估/分析/筛查/视频与核验产物、本报告，
并同步README、BASELINE_SPEC、注册表和生成索引。研究设定与所有旧模型/指标未改。
