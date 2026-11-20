# 训练后轨迹演化图：实现与 v08/v09 结果

本轮只实现训练后分析并读取已有模型做确定性推理，没有启动正式长训练，也没有更新任何原模型权重或训练指标。

## 当前两套结果

两个实验统一使用原始 0° 来袭方向、seed 1000、敌机初位 `[497.49371855331,0,50]`，我方初态和其他物理参数不变。
所有 checkpoint 的环境快照、网络/PPO/归一化设置均一致，三机仍仅通过局部 21D observation 确定性行动。
图上为 XY 投影，捕获和表中距离仍按实际 3D 状态计算；平面交叉不代表捕获。

| 实验 | 实际评估步数 | XY 主图 / 指标 |
| --- | --- | --- |
| v08-r01 300k | 5120, 10240, 20480, 40960, 61440, 81920, **92160**, 98816, 148480, 199680, 250880, 300000 | [主图](../outputs/v08_捕获奖励500/v08-r01_20261108_seed000/trajectory_evolution/trajectory_evolution_xy.png) · [CSV](../outputs/v08_捕获奖励500/v08-r01_20261108_seed000/trajectory_evolution/trajectory_evolution_metrics.csv) |
| v09-r01 累计 300k | 0, 10000, 20000, **30000**, 40000, 60000, 80000, 100000, 150000, 200000, 250000, 300000 | [主图](../outputs/v09_来袭方向课程/v09-r01_20261109_seed000/continuation_300k/trajectory_evolution/trajectory_evolution_xy.png) · [CSV](../outputs/v09_来袭方向课程/v09-r01_20261109_seed000/continuation_300k/trajectory_evolution/trajectory_evolution_metrics.csv) |

v08 没有保存 0 步模型，因此用最早真实 5120 步模型并在 manifest 记录，未按旧 seed 重构或伪造初始权重。
92160 步模型是显式额外纳入的已知成功 checkpoint，本场景复现 17.3s 捕获、最小距离 3.525m；抽样的其余策略均失败，说明中期成功未被后期持续保持。
v09 的 12 个抽样策略在原始 0° 场景下均失败；30k 模型最近 31.034m，300k 模型最近 208.020m。
这与此前 +3.13° 视频的成功不矛盾：那是另一个明确选择的来袭角；30k best 在固定 100 角度评估中的整体捕获率仍是 9%。
不能把不同模型各一回合的结果合并成某个模型的成功率或泛化指标。

每个输出目录还包含 `trajectory_evolution_checkpoints_xy.png`：使用相同坐标范围逐模型展示，减少叠加遮挡。
主图四条线型区分 P0/P1/P2/Attacker，浅到深编码实际训练步数，标出基地、起终点和真实捕获事件，并附实际步数/结局/事件时间/最小距离表。

## 生成逻辑

1. 只扫描本 output 的 `checkpoints/*.pt` 和 `stage_*/checkpoints/*.pt`，不混入其他 run、视频或源码快照。
2. 从 checkpoint `training_state.env_steps` 读取实际步数，目标步数取最近可用点；并列取较早，去重后加入首末和保存的 best；额外步骤必须真实存在。
3. 相同步数的别名必须拥有相同 Actor/Critic 权重。跳过非首阶段 initial 和 resume_initial，因为它们是交接诊断，可能在同一计数恢复较早 best，不代表新的训练策略。
4. 所有策略使用同一份环境快照、相同角度/初始状态和 seed；`eval()` + `torch.inference_mode()` + deterministic Actor，运行至原捕获、攻击成功或超时条件。
5. 保存完整初始至终止轨迹和指标 CSV，再渲染 XY 主图及逐 checkpoint 小图；不平滑、插值、拼造轨迹。
6. 对照原模型/配置/指标 SHA256；finally 恢复 Python、NumPy、Torch/CUDA RNG、CPU 线程数及 cuDNN 标志。

CSV 字段：实际 `step`、Capture/Fail/Timeout、`episode_length`、`event_time`（秒）、`min_pursuer_attacker_distance`（3D 米）、`final_attacker_base_distance`（3D 米）、团队回报、checkpoint 路径与 SHA256、完整轨迹路径。

## 自动调用与配置

普通 MAPPO 和课程训练入口均在最终 checkpoint/终止摘要保存后调用独立报告函数。该函数不更改原训练返回值，不参与梯度、reward、observation、dynamics、best 选择或课程 gate。
普通 trainer 新增真实 0 步 `initial.pt` 保存，未来训练可从真正初始策略开始分析；既有 v08 不补造。

配置：`configs/trajectory_evolution.yaml`，默认启用，输出子目录 `trajectory_evolution`。
目标步骤、额外步骤、固定角度、seed、CPU 线程、图大小/DPI、色图与浅深范围均独立记录。
`enabled: false` 可关闭；也可在训练配置 `post_training.trajectory_evolution_config` 指定另一份分析 YAML。
默认情景为 0°，课程的随机训练分布保持原样。
报告失败显式 warning 并写 `trajectory_evolution_error.json`，不丢失已完成的训练；目前仅支持点防御快照。
已有演化输出拒绝覆盖，手工复跑时应指定新的 `--output-dir`。

```powershell
# 历史 v08；额外纳入真实成功模型
.\.venv\Scripts\python.exe -B -m scripts.trajectory_evolution --run-dir outputs/v08_捕获奖励500/v08-r01_20261108_seed000 --extra-steps 92160
# 当前 v09 累计 300k
.\.venv\Scripts\python.exe -B -m scripts.trajectory_evolution --run-dir outputs/v09_来袭方向课程/v09-r01_20261109_seed000/continuation_300k
# 复跑同一个 output 时用新的目标目录，原结果不会覆盖
.\.venv\Scripts\python.exe -B -m scripts.trajectory_evolution --run-dir <训练目录> --output-dir <新目录>
```

## 修改文件与验证

| 文件 | 内容 |
| --- | --- |
| `configs/trajectory_evolution.yaml` | 独立分析参数 |
| `utils/trajectory_evolution.py` | 发现、抽样、冻结推理、指标及完整来源核验 |
| `utils/trajectory_evolution_plot.py` | XY 主图、表、逐 checkpoint 小图 |
| `scripts/trajectory_evolution.py` | 历史训练的独立生成入口 |
| `utils/post_training.py` | 训练结束回调和失败记录 |
| `scripts/train_mappo.py` | 保存真实初始模型，结束后调用分析 |
| `scripts/train_direction_curriculum.py` | 全课程终止后调用一次分析 |
| `tests/test_trajectory_evolution.py` | 18 项工程与集成测试 |
| README、BASELINE_SPEC、AGENTS、实验索引 | 入口、交付流程及当前结果 |

完整 pytest 实际 **372 passed in 76.09s**（原 354 + 新增 18）；日志：[pytest.log](../outputs/_project_项目核对与管理资料/20261109_trajectory_evolution/pytest.log)。
覆盖真实步数/去重/缺失 0 步、额外模型、统一初态与全回合、三种终局、CSV/NPZ 三维距离与事件时间、禁止 update、含双 Adam 的源文件哈希、RNG/线程/cuDNN 恢复、两入口结束后回调、禁用及失败保留 checkpoint。
实际生成 v08/v09 共 24 个冻结回合，两套主图及逐模型图已检查。没有 Git commit、merge 或 push。
