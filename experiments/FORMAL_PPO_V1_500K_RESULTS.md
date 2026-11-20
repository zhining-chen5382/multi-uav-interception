# 单机PPO五Seed累计500k正式实验结果

结果核对完成于2026-11-17，实验起始日期为2026-11-16。

## 结论

五个Seed均完成累计500,000 aggregate environment steps，总计2,500,000步、615次PPO更新、8852个完整训练回合，随机训练捕获50次。
五个验证Best在同一组500个新随机角度上冻结测试，捕获率均值10.24%，跨Seed样本标准差11.17个百分点，95% Student-t(4)均值区间为[0%,24.11%]。本轮未达到平均捕获率80%的目标。
300k阶段平均测试捕获率为9.28%，补到500k后增加0.96个百分点。四个Seed的历史Best仍来自300k以内，Seed1001的新Best为430k。

## 实验协议与实际参数

任务为1v1三维点防御，Direct Attacker每回合来袭角独立采样U(-30°, +30°)，基地为原点，0°沿+X，逆时针为正。P0初态[1.5,0,1]，敌机0°初态[497.49371855331,0,50]，两机初速度均为零。
观测为自身速度3、applied acceleration3、敌机相对位置3、敌机相对速度3、敌机相对基地位置3及剩余时间比例1。除数严格为[20,20,8,3,3,2,1500,1500,1500,40,40,14,500,500,500,1]，采用逐维除法，越界记录日志。applied acceleration读取环境维护的jerk历史状态。
Actor/Critic独立，16输入、128×128 Tanh隐藏层。策略使用三个可学习log_std，初始std=0.5、log_std范围[-5,0]、Tanh动作和Jacobian修正。PPO采用8独立环境、每环境512步、minibatch256、最多4 epochs、gamma0.99、GAE lambda0.95、clip0.2、解析KL阈值0.015、entropy coefficient0.01、value coefficient0.5。Adam betas=(0.9,0.999)、eps=1e-5、weight_decay=0、AMSGrad=False。
捕获半径严格小于5m，基地失守严格小于10m，捕获优先。dt=0.1s，50s超时作为真正终止。奖励为-0.01*当前敌我距离*dt-0.2*dt+500*捕获-100*失守。
五个Seed为1001—1005，初始300k分别从零开始。用户在训练过程中追加到500k，各自完整恢复300k final，包含双网络、双Adam、RNG、环境及回合累积状态。
追加训练采用保留既有学习率履历的实施决定。0—300k沿原线性衰减，Actor 3e-4到3e-5、Critic 1e-3到1e-4，300k后维持终值。该决定已在执行前说明，未重新安排已经执行的学习率。
每Seed初始阶段为73个4096样本批次加992尾批，追加阶段为48个4096样本批次加3392尾批，共123次更新。每10k确定性验证相同100个线性均匀角度，共50次。连续三次达到80%只用于稳定性标记，五组均未达到该标记。
验证root seed为2001，测试root seed为3001。测试500角度与训练、验证独立，五模型共享，测试期间冻结参数。用户追加500k的决定先于300k中间测试结果生成，追加阶段参数按已锁定方案执行。

## 五Seed结果

下表测试使用验证选出的Best，每模型500回合。Final列使用500k末次的100个固定验证场景。

| Seed | 完成回合 | 训练捕获 | Best步数 | Best测试捕获率 | Final验证捕获率 | 失守率 | 成功平均时间s | 平均最小距离m | 平均回报 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1001 | 1771 | 14 | 430000 | 4.8% | 0.0% | 95.2% | 19.762 | 52.292 | -154.269 |
| 1002 | 1773 | 16 | 130000 | 8.4% | 0.0% | 91.6% | 23.248 | 31.665 | -130.008 |
| 1003 | 1770 | 13 | 260000 | 29.2% | 1.0% | 70.8% | 23.249 | 11.379 | -2.753 |
| 1004 | 1770 | 7 | 120000 | 8.8% | 0.0% | 91.2% | 20.014 | 46.018 | -125.454 |
| 1005 | 1768 | 0 | 10000 | 0.0% | 0.0% | 100.0% | 无成功回合 | 73.860 | -191.729 |

总计256/2500个模型—场景组合捕获，防御失败率89.76%，超时率0%。成功回合合并平均拦截时间22.366s。跨Seed平均最小距离43.043m，平均回报-120.843。
捕获时间仅统计成功回合，Seed1005没有成功回合。总体时间描述使用实际256个成功回合；自动JSON另列四个有成功模型的条件时间均值21.568s。
95%均值区间以五次训练为统计单位，并以本次共享500场景为条件。每个模型另记录Wilson二项区间，详见test_results_by_seed.csv。

## 方向分区与失败行为

| 角度区间 | 每模型场景数 | 五Seed平均捕获率 |
| --- | --- | --- |
| [-30,-20) | 91 | 11.87% |
| [-20,-10) | 88 | 39.55% |
| [-10,0) | 100 | 5.60% |
| [0,10) | 67 | 0% |
| [10,20) | 73 | 0% |
| [20,30] | 81 | 0% |

全部256次测试捕获发生在负角度，正角度为0/1105。Seed1003在[-20,-10)区间88/88捕获，但正角度三个区间均失败，当前能力具有显著方向不对称。
实际训练中正角度比例分别约49.07%、49.63%、51.36%、49.10%、50.74%。初态atan2方位、500m半径和50m高度核验通过，训练采样覆盖未见符号错误。
Seed1001追加200k新增6次随机训练捕获，430k验证达到4%，之后再次回落。Seed1002新增13次捕获，但新增验证最高5%，历史130k的8%保留。Seed1003新增7次捕获，追加验证最高8%，历史260k的26%保留。Seed1004和1005追加阶段均无训练捕获，新增验证均为0%。
五个Final验证捕获率分别为0%、0%、1%、0%、0%。训练出现局部角度拦截，继续更新后能力多次丢失。
固定0°视频中十个Best/Final均失守。Seed1003 Best最近距离7.954m，最近时敌我位置差约[-0.448,-7.306,-3.114]m，主要剩余误差位于横向和高度。其500k Final最近距离扩大到67.985m。

演化记录中Seed1001的0步随机初始化Actor在固定0°单例捕获，28.2s、最近3.971m、回报415.252。当步敌机距基地8.261m，按已确认的捕获优先规则判成功。该记录覆盖固定0°单个场景，后续训练未持续保留该单例能力。
Seed1004和1005 Final出现方向及高度漂移。1004末位置约[17.487,242.728,182.924]，1005约[201.29,312.18,-153.70]。这些位置处于既定数值工作空间内。

## 数值稳定与策略保持

全部训练和测试记录数值有限。未发现NaN/Inf、hard boundary clipping、归一化越界或超时。最终raw std约0.50—0.54。
Seed1004追加阶段变换后entropy由约1.222降到0.027，1005由约0.310降到-0.177。连续分布微分熵可以为负，测试已验证其Jacobian计算。该变化伴随性能和空间方向退化。
0° Final中1004/1005的x动作超过0.95的帧比例分别约3.55%/12.77%，未出现超过0.99的帧。其余数据没有支持全程动作饱和或raw std归零的结论。
Critic loss降低与捕获能力未能同步保持，尤其Seed1005始终无捕获。上述观察不能唯一确定退化原因，本轮没有据此调整奖励、探索、网络或其他PPO参数。

## 工程验收与恢复

生产观测专项31项、预算扩展专项20项及恢复编排等测试完成，最终完整pytest为701 passed in113.06s。生产配置检查及两次生产设置Smoke通过，真实中断恢复的双网络、双Adam、Scheduler、环境、未完成Buffer、RNG及CSV逐位一致。
Seed1001完成500k后发生一次后处理进程中断，日志没有正常退出标记，原因未能从已有日志确定。复用该500k模型，恢复后处理到postprocessing_002；其余Seed按顺序继续。通过批准的独立后台进程完成，最终process_completion.json的exit_code为0。
实际源300k Checkpoint哈希、全部日志前缀及初始恢复状态保持一致。十段完整视频全帧解码通过，每段283帧。最终逐一读取2500条NPZ，核对初态、字段归一化、applied来源、运动约束、回报和最近距离，全部通过。
代码HEAD为c6fe7e7c64e62425c079960b3895615b6f274113，工作区含未提交修改，实际源码快照及SHA保存于实验目录和execution_recovery_001。历史注册表的48个模型路径与Checkpoint哈希核验通过。

## 工程修改清单

- FORMAL_TRAINING_SPEC_V1.md、BASELINE_SPEC.md、README.md写入已确认约定和实验入口。
- configs/interception_1v1.yaml填入全部除数、applied定义和确认来源。configs/ppo_formal_v1_500k.yaml记录500k预算及300k衰减horizon。
- utils/formal_ppo_config.py和utils/formal_ppo_training.py约束正式入口的applied来源，训练器增加显式预算扩展及完整恢复。
- algorithms/ppo/agent.py支持保留原衰减周期并在扩展预算内维持终值，默认预算行为保持回归验收。
- envs/interception_env.py补充实际加速度字段来源说明。
- scripts/run_formal_experiment.py支持对应预算说明和源码快照，新增scripts/run_formal_500k.py及scripts/resume_formal_500k.py完成顺序续训与中断恢复。
- 新增tests/test_formal_production_contract.py、test_formal_budget_extension.py、test_formal_500k_entry.py、test_formal_recovery.py，更新test_formal_ppo_evaluation.py的未确认配置负例。

## 结果入口

- [自动综合报告](E:/projects/research/multi-uav-interception/outputs/formal_ppo_v1/v14_单机PPO_原始奖励/02_续训至500k/FORMAL_EXPERIMENT_REPORT.md)
- [逐Seed测试CSV](E:/projects/research/multi-uav-interception/outputs/formal_ppo_v1/v14_单机PPO_原始奖励/02_续训至500k/test_results_by_seed.csv)
- [角度分区CSV](E:/projects/research/multi-uav-interception/outputs/formal_ppo_v1/v14_单机PPO_原始奖励/02_续训至500k/test_angle_bins.csv)
- [最终交付核验](E:/projects/research/multi-uav-interception/outputs/formal_ppo_v1/v14_单机PPO_原始奖励/02_续训至500k/execution_recovery_001/final_artifact_audit_20261117.json)
- [完成状态](E:/projects/research/multi-uav-interception/outputs/formal_ppo_v1/v14_单机PPO_原始奖励/02_续训至500k/execution_recovery_001/process_completion.json)

视频统一为固定0°场景。测试表采用500个随机角度，原始轨迹、视频manifest与解码检查随各Run保存。

| Seed | 报告 | XY演化 | Best视频 | Final视频 |
| --- | --- | --- | --- | --- |
| 1001 | [报告](E:/projects/research/multi-uav-interception/outputs/formal_ppo_v1/v14_单机PPO_原始奖励/02_续训至500k/seed_1001/postprocessing_002/TRAINING_REPORT.md) | [XY](E:/projects/research/multi-uav-interception/outputs/formal_ppo_v1/v14_单机PPO_原始奖励/02_续训至500k/seed_1001/postprocessing_002/trajectory_evolution/trajectory_evolution_xy.png) | [Best](E:/projects/research/multi-uav-interception/outputs/formal_ppo_v1/v14_单机PPO_原始奖励/02_续训至500k/seed_1001/postprocessing_002/videos/best/best_full_episode.mp4) | [Final](E:/projects/research/multi-uav-interception/outputs/formal_ppo_v1/v14_单机PPO_原始奖励/02_续训至500k/seed_1001/postprocessing_002/videos/final/final_full_episode.mp4) |
| 1002 | [报告](E:/projects/research/multi-uav-interception/outputs/formal_ppo_v1/v14_单机PPO_原始奖励/02_续训至500k/seed_1002/postprocessing_001/TRAINING_REPORT.md) | [XY](E:/projects/research/multi-uav-interception/outputs/formal_ppo_v1/v14_单机PPO_原始奖励/02_续训至500k/seed_1002/postprocessing_001/trajectory_evolution/trajectory_evolution_xy.png) | [Best](E:/projects/research/multi-uav-interception/outputs/formal_ppo_v1/v14_单机PPO_原始奖励/02_续训至500k/seed_1002/postprocessing_001/videos/best/best_full_episode.mp4) | [Final](E:/projects/research/multi-uav-interception/outputs/formal_ppo_v1/v14_单机PPO_原始奖励/02_续训至500k/seed_1002/postprocessing_001/videos/final/final_full_episode.mp4) |
| 1003 | [报告](E:/projects/research/multi-uav-interception/outputs/formal_ppo_v1/v14_单机PPO_原始奖励/02_续训至500k/seed_1003/postprocessing_001/TRAINING_REPORT.md) | [XY](E:/projects/research/multi-uav-interception/outputs/formal_ppo_v1/v14_单机PPO_原始奖励/02_续训至500k/seed_1003/postprocessing_001/trajectory_evolution/trajectory_evolution_xy.png) | [Best](E:/projects/research/multi-uav-interception/outputs/formal_ppo_v1/v14_单机PPO_原始奖励/02_续训至500k/seed_1003/postprocessing_001/videos/best/best_full_episode.mp4) | [Final](E:/projects/research/multi-uav-interception/outputs/formal_ppo_v1/v14_单机PPO_原始奖励/02_续训至500k/seed_1003/postprocessing_001/videos/final/final_full_episode.mp4) |
| 1004 | [报告](E:/projects/research/multi-uav-interception/outputs/formal_ppo_v1/v14_单机PPO_原始奖励/02_续训至500k/seed_1004/postprocessing_001/TRAINING_REPORT.md) | [XY](E:/projects/research/multi-uav-interception/outputs/formal_ppo_v1/v14_单机PPO_原始奖励/02_续训至500k/seed_1004/postprocessing_001/trajectory_evolution/trajectory_evolution_xy.png) | [Best](E:/projects/research/multi-uav-interception/outputs/formal_ppo_v1/v14_单机PPO_原始奖励/02_续训至500k/seed_1004/postprocessing_001/videos/best/best_full_episode.mp4) | [Final](E:/projects/research/multi-uav-interception/outputs/formal_ppo_v1/v14_单机PPO_原始奖励/02_续训至500k/seed_1004/postprocessing_001/videos/final/final_full_episode.mp4) |
| 1005 | [报告](E:/projects/research/multi-uav-interception/outputs/formal_ppo_v1/v14_单机PPO_原始奖励/02_续训至500k/seed_1005/postprocessing_001/TRAINING_REPORT.md) | [XY](E:/projects/research/multi-uav-interception/outputs/formal_ppo_v1/v14_单机PPO_原始奖励/02_续训至500k/seed_1005/postprocessing_001/trajectory_evolution/trajectory_evolution_xy.png) | [Best](E:/projects/research/multi-uav-interception/outputs/formal_ppo_v1/v14_单机PPO_原始奖励/02_续训至500k/seed_1005/postprocessing_001/videos/best/best_full_episode.mp4) | [Final](E:/projects/research/multi-uav-interception/outputs/formal_ppo_v1/v14_单机PPO_原始奖励/02_续训至500k/seed_1005/postprocessing_001/videos/final/final_full_episode.mp4) |
