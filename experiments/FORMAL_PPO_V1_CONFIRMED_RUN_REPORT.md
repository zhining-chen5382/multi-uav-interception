# 单机 PPO 正式实验执行记录

2026-11-16用户确认16维归一化与applied acceleration定义，正式协议见 `FORMAL_TRAINING_SPEC_V1.md`。

## 最终状态，2026-11-17核对

五个Seed均完成累计500,000步，总计2,500,000步、615次更新、8852个训练回合。最终进程完成状态为exit_code=0，十段完整视频与2500条测试轨迹核验通过。最终Best测试捕获率为4.8%、8.4%、29.2%、8.8%、0%，均值10.24%，未达到80%目标。

完整最终结果和视频入口见 [FORMAL_PPO_V1_500K_RESULTS.md](FORMAL_PPO_V1_500K_RESULTS.md)。下文保留本轮按时间形成的实施、追加预算和恢复记录。

## 生产约定

固定除数为 `[20,20,8,3,3,2,1500,1500,1500,40,40,14,500,500,500,1]`。按世界坐标字段顺序逐维除法归一化，记录超尺度值。观测3–5维读取jerk限制后的 `accelerations[0]`，它也是下一步jerk限制使用的历史加速度状态。

生产环境配置为 `configs/interception_1v1.yaml`，已写入确认标志、来源、完整缩放表和 `acceleration_kind=applied`。正式配置检查与单Run训练校验均拒绝realized加速度定义。

本轮其他已确认训练与任务参数沿用 `configs/ppo_formal_v1.yaml`。五次分别从零开始，按Seed1001–1005顺序，每次300k aggregate steps。验证每10k使用相同100个固定角度，五次全部完成后以各自Best共享500个新随机角度测试。

## 验收状态

- 生产配置检查通过，ready=true。
- 新增生产观测专项31项通过，4.74s。
- 完整pytest为674 passed，180.90s，全部通过。
- 正式入口已启动，先执行生产Smoke，通过后自动开始五次训练。

专项覆盖16个逐维缩放、世界坐标字段顺序、正负值、四种限速情况下的applied与速度差分区别、下一步jerk的同一持久状态、越界日志、零值、历史统计无影响、防双重归一化、训练/验证/测试一致性和完整环境状态恢复。

## 结果位置

用户在初始训练进行期间追加预算，五个Seed最终各累计500k。保持既有300k学习率履历，300k后固定已确认终值，从各自final完整追加200k。目标累计2,500,000，追加1,000,000。

500k流程使用新配置 `configs/ppo_formal_v1_500k.yaml` 与入口 `scripts/run_formal_500k.py`，源码扩展及恢复测试验收通过后执行。500k结果将位于 `formal_20261116_continuation_500k/`，完成后的综合报告作为本轮最终结果。

500k扩展验收已完成，新增预算扩展专项20项、总入口专项3项，完整pytest共697项通过，129.91s。五个真实300k final来源检查通过。生产续训Smoke从300k恢复，比较不中断运行至312288与310k暂停后恢复至312288，1,808条未完成rollout样本、双网络、双Adam、Scheduler、环境、RNG和CSV逐位一致，源Checkpoint哈希保持。

五个300k阶段及其自动中间评估已完成，总计370次更新、5297个训练回合、24次随机训练捕获。用户追加500k的指令先于中间测试结果生成，追加训练仍按已锁定参数执行。300k中间Best测试捕获率依次为0%、8.4%、29.2%、8.8%、0%，均值9.28%。500k续训已启动，其最终结果待完成后汇总。

- `_acceptance_confirmed_20261116/` 保存本轮验收和执行日志。
- `_normalization_confirmed_20261116/normalization_tests.log` 保存31项专项结果。
- `formal_20261116_acceptance_smoke/` 保存正式配置的中断恢复对照。
- `formal_20261116/seed_1001/` 至 `seed_1005/` 保存五次运行。
- 每Run的 `postprocessing_001/` 保存曲线、XY演化、完整视频和自动报告。
- 每Run的 `final_test/` 保存500场景冻结Best测试。
- `formal_20261116/FORMAL_EXPERIMENT_REPORT.md` 保存最终综合报告。

以上路径均以 `outputs/formal_ppo_v1/` 为根。

执行过程与完成状态见本轮 `formal_console.log` 和 `process_completion.json`，完整结果在五次训练及最终测试结束后汇总。

## 500k执行中断与恢复

Seed1001已完整达到500,000步、123次更新、1771个训练回合和50次验证。执行进程在该Run后处理期间中断，原控制台没有异常堆栈或正常完成标记，检查时没有存活Python进程。中断原因未能从现有日志确定。

只读核对确认final.pt完整且所有数值有限，300k前缀及源文件哈希匹配。追加200k训练捕获6次，累计14次。验证Best为430k、4%，500k末次验证为0%。原postprocessing_001中的两个视频及解码记录已存在，顶层报告和完成manifest缺失。

新增 `scripts/resume_formal_500k.py`，复用已完成训练模型，为中断后处理创建新编号目录。其余Seed从各自300k final继续，未完成Run从实际最大步数Checkpoint恢复至新目录。原目录及中断产物保留。恢复配置与原500k实验配置逐项相等，来源Checkpoint哈希再次核对。

恢复编排及预算、入口专项共27项通过，11.37s，其中新增4项覆盖已完成训练跳过、后处理恢复、最大实际步数选择和原final继承。完整回归结果记录在 `_extension_acceptance_20261116/recovery_full_pytest.log`。

完整恢复回归为701 passed，113.06s。沙箱内独立子进程随工具会话结束退出后，已通过批准的沙箱外隐藏后台进程恢复执行，控制台为 `_extension_acceptance_20261116/recovery_002_stdout.log`，错误日志为同目录 `recovery_002_stderr.log`，完成标记由Python进程直接写入 `formal_20261116_continuation_500k/execution_recovery_001/process_completion.json`。Seed1001后处理在postprocessing_002完成，后续Seed按顺序继续。
