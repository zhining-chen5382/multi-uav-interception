# Formal PPO v1 实施与验收记录

日期为 2026-11-16。目标为独立 1v1 单机 PPO，从零训练五个 seed，每次 300,000 aggregate environment steps。

## 当前执行状态

独立实现及工程 fixture 验收已完成，完整 pytest 为 **643 passed in 125.56s**。正式配置的归一化约定仍缺少可追溯的确认值，正式配置 smoke 和正式训练受阻。

| Seed | 正式训练状态 | 正式累计步数 | 500 场景最终测试 |
| --- | --- | --- | --- |
| 1001 | 未启动 | 0 | 未执行 |
| 1002 | 未启动 | 0 | 未执行 |
| 1003 | 未启动 | 0 | 未执行 |
| 1004 | 未启动 | 0 | 未执行 |
| 1005 | 未启动 | 0 | 未执行 |

正式预算执行量为 0 / 1,500,000。工程测试中的短程模型及轻量 300k 计数替身均保存在验收目录。

## 正式开训阻断项

已检索源码、配置、文档及用户提供的中文《正式训练规范 v1.0》附件。附件要求结合实际物理边界核实缩放系数，未列出最终确认的 16 个数值。当前可读取资料中没有找到确认表。

需要补充按以下字段顺序排列的 16 个物理除数。

| 索引 | 世界坐标字段 | 数量 |
| --- | --- | --- |
| 0–2 | 自身速度 XYZ | 3 |
| 3–5 | 自身实际加速度 XYZ | 3 |
| 6–8 | 敌机位置减自身位置 XYZ | 3 |
| 9–11 | 敌机速度减自身速度 XYZ | 3 |
| 12–14 | 敌机位置减基地位置 XYZ | 3 |
| 15 | 剩余时间除以最大回合时间 | 1 |

原动力学返回 jerk 限制后的 `applied_acceleration`。限速后的速度差分 `(v_next-v_previous)/dt` 为 `realized_acceleration`。二者在触发速度限制时可能不同，还需明确观测选用哪个定义。

`configs/interception_1v1.yaml` 中 `normalization.confirmed=false`，`scales` 和 `acceleration_kind` 保留空值。正式入口在创建输出目录和训练模型前检查这些字段。测试中的缩放值均标明 fixture 来源。

## 环境与 PPO

`envs/interception_env.py::InterceptionEnv` 提供独立 1v1 接口。

- P0 初始位置 `[1.5, 0, 1]`，初速为零。
- 0° 敌机初始位置 `[497.49371855331, 0, 50]`，初速为零，基地为原点。
- 每回合从 U(-30°, +30°) 独立采样来袭角，绕基地逆时针为正，保持原高度及距基地距离。
- 复用原 `envs/dynamics.py` 和 `DirectAttackerPolicy`，动作仍映射为受原约束执行的三维加速度，保留数值边界硬裁剪。
- Actor/Critic 使用相同 16 维观测，按固定物理尺度除法归一化，逐字段记录越界。
- 捕获严格采用敌我距离 `<5m`，失守采用敌机到基地距离 `<10m`，捕获优先。
- `dt=0.1s`，最大 500 步。捕获、失守及超时均为 `terminated=True`、`truncated=False`。

四项奖励使用本步完成后的敌我距离。

\[
r_t=-0.01d_{t+1}\Delta t-0.2\Delta t
    +500\mathbb{1}_{capture}-100\mathbb{1}_{defense\ failure}.
\]

1v1 奖励删除机间碰撞项。纯超时触发终止，以上四项中没有额外超时失守罚分。

`algorithms/ppo/agent.py::PPOAgent` 实现以下设置。

| 项目 | 实际设置 |
| --- | --- |
| Actor | 16 → 128 Tanh → 128 Tanh → 3 Gaussian mean |
| Critic | 16 → 128 Tanh → 128 Tanh → 1 value |
| 初始化 | 复用原正交初始化，隐藏层 gain √2，Actor/Critic 输出 gain 0.01/1，bias 0 |
| log_std | 三个独立可学习参数，初始 ln(0.5)，参数及采样范围 [-5, 0] |
| 动作与概率 | pre-tanh 动作保存，Tanh 变换，稳定 log-Jacobian 修正 |
| Entropy | 变换后策略的可重参数化 MC 熵，系数固定 0.01 |
| Actor LR | 按已用环境步数从 3e-4 线性到 3e-5 |
| Critic LR | 按已用环境步数从 1e-3 线性到 1e-4 |
| Adam | betas=(0.9, 0.999)，eps=1e-5，weight_decay=0，AMSGrad=False |
| PPO | gamma=0.99，GAE λ=0.95，clip=0.2，最多 4 epochs，minibatch=256 |
| KL | 每次 minibatch 更新后计算 old-policy 到 current-policy 的解析 KL，严格 >0.015 时停止本 rollout 剩余更新 |
| Value Loss | Lv=0.5 mean((V-return)²)，Critic objective=0.5 Lv，沿用既有 loss 定义及确认的系数 |
| Advantage | 整个 rollout 归一化，epsilon=1e-8 |
| 梯度裁剪 | max norm=0.5 |
| Value Clipping / Normalization | 关闭 |

LR 在初始化时为起始值，每次 PPO 更新使用已完成采样步数对应的 LR，300k 更新使用精确终值。

`VectorRolloutBuffer` 保留时间和环境两个轴。真正终止的 next value 为零，rollout 结束但回合继续时正常 bootstrap。

## 训练、验证、Checkpoint

`utils/formal_ppo_training.py` 实现 8 环境同步向量采样。批量推理后依次执行物理 step，每个环境拥有独立状态及 `SeedSequence.spawn` 派生的局部 Generator。

每个完整 rollout 为 512×8=4096 样本。300k 由 73 个完整 rollout 和 992 样本尾批构成，尾批每环境 124 步，最后 minibatch 为 224 样本。

每 10k 精确验证固定的 100 个线性均匀角度，包含 -30° 和 +30°。验证保留未完成 buffer，CSV 同时记录采样步数、模型最近更新步数和 pending 样本。例如 10k 验证权重一般最近更新于 8192 步。

Best 依次比较捕获率高、防御失败率低、成功平均拦截时间短、平均最小距离小。完全相同时保留原 Best。连续三次 ≥80% 只记录稳定性，继续执行完整 300k。

Checkpoint 保存双网络、双 Adam、Scheduler、log_std、Python/NumPy/Torch 全局 RNG、八环境局部 RNG 和完整状态、当前观测、未完成 buffer、回合累积量、验证和 Best 状态及完整日志。

恢复使用新目标目录，核对 seed、预算、环境和学习配置后继续本运行自己的未完成 rollout。当前正式配置为 CPU，完整逐位恢复验收在 CPU 上完成。

## 正式总流程

`scripts/run_formal_experiment.py` 实现以下顺序。

1. 检查已确认的归一化约定。
2. 用正式配置运行短程 smoke，保留原 300k LR horizon。比较不中断到 12288 步与 10000 步暂停后恢复到 12288 步，核对模型、优化器、调度器、八环境、buffer、RNG 和 CSV 逐位相同。
3. 新建完整解析配置、实际代码快照、源码 SHA256 和共享场景文件。
4. 依次从零运行 seeds 1001–1005，每个 300k。
5. 五次全部完成后，冻结每次验证选出的 Best，在共享的 500 个新均匀随机角度上测试。
6. 保存逐 seed 和六个 10° 角度分区统计。跨训练运行采用样本标准差和 t(4) 均值区间，每模型另记录 Wilson 捕获率区间。

工程上为独立验证和测试场景选用了 root seed 2001、3001，均写入配置和场景记录。验证网格包含端点，测试集在训练开始前独立生成，五模型共享。整体达标判定为五次 Best 平均测试捕获率 ≥80%。

生产设置 smoke 仍待归一化确认。本轮已完成同一实现的显式工程 fixture 短程 smoke。

## 图表、视频和保存路径

`utils/formal_ppo_reporting.py::postprocess_run` 在每次完整训练后自动新建 `postprocessing_001/`，重复生成使用递增目录。

- 曲线涵盖 capture/reward、Actor/Critic loss、双 LR、三维 log_std/std、entropy、KL、clip fraction、回合长度及最小距离。
- 曲线使用原始 CSV，未进行滑动平均。
- 所有真实保存的 checkpoint 在固定 0° 场景运行完整确定性回合。相同 Actor 保存点复用完整轨迹，CSV 记录映射。
- XY 主图由浅到深表示实际采样步数，标出基地、起点、终点及结果。CSV 包含真实计数、最近更新时间、pending 样本、回合长度、事件时间、最小敌我距离和最终敌机到基地距离。
- Best/final 在同一固定 0° 场景生成完整视频，保留初始至终止轨迹、manifest、源文件哈希和全帧解码校验。
- Run 和后处理目录均生成 README 导航，单次报告为 `postprocessing_001/TRAINING_REPORT.md`。
- 总报告为 `FORMAL_EXPERIMENT_REPORT.md`，总对比图位于 `comparison_plots/`。

总目录说明为 `outputs/formal_ppo_v1/README.md`。真正执行后创建 `formal_20261116/seed_1001/` 至 `seed_1005/`，共享场景位于 `scenarios/`，各模型最终测试位于 `seed_*/final_test/`。

## 文件变更清单

新增源码及配置如下。

- `envs/interception_env.py`
- `algorithms/ppo/__init__.py`、`agent.py`、`buffer.py`
- `configs/interception_1v1.yaml`、`configs/ppo_formal_v1.yaml`
- `utils/formal_ppo_config.py`、`formal_ppo_training.py`、`formal_ppo_evaluation.py`、`formal_ppo_reporting.py`
- `scripts/train_formal_ppo.py`、`scripts/run_formal_experiment.py`
- `tests/test_interception_env.py`、`test_single_ppo.py`、`test_formal_ppo_training.py`、`test_formal_ppo_integration.py`、`test_formal_ppo_evaluation.py`、`test_formal_ppo_reporting.py`、`test_formal_experiment.py`
- `experiments/FORMAL_PPO_V1_IMPLEMENTATION.md`

修改 `requirements.txt` 以声明视频依赖，修改 `README.md` 添加入口，在 `experiments/experiment_registry.yaml` 添加工程输出目录覆盖记录。

Git HEAD 为 `c6fe7e7c64e62425c079960b3895615b6f274113`，工作区包含既有未提交内容。正式运行会记录 dirty 状态并复制实际源码及 SHA256。

## 验收结果

| 阶段 | 通过数 | 主要覆盖 |
| --- | --- | --- |
| 环境 | 37 | 字段、动力学、奖励、终止、RNG 和状态恢复 |
| PPO/Buffer | 27 | 密度、熵积分、解析 KL、严格早停、GAE、完整优化状态 |
| 训练器 | 18 | 完整恢复、300k 精确计数、30 次10k验证、达标持续训练 |
| 独立集成 | 3 | 八真实环境及网络恢复、Best 排序、异常时 RNG 恢复 |
| 评估及总流程 | 9 | 共同场景、五 seed 顺序、统计、真实短程 smoke 入口 |
| 后处理 | 14 | 原始曲线、真实 checkpoint、XY 图、完整 MP4 编码解码 |
| 完整 pytest | **643** | 历史测试及全部新增验收，125.56s |

完整日志为 `outputs/formal_ppo_v1/_工程验收/01_初始工程测试/full_pytest.log`。训练器最初专项产物另在 `outputs/_formal_engineering_trainer_20261116/`。300k 控制循环测试使用轻量工程替身，实际模型更新与恢复在短程真实环境及 PPO 测试中核实。

完整测试结束后补齐源码快照的导入依赖清单，正式入口专项再次通过 4 项，5.26s，日志为 `entrypoint_final_tests.log`。以上通过数按独立测试项统计。

注册表检查通过，13 个历史版本、23 个 run、131 个事件、48 个模型的路径和 checkpoint 哈希均核验成功。

本轮一次 Python 子进程曾以 Windows `-1073741502` 初始化错误退出，随后独立启动与同命令重试成功，完整测试正常结束。正式训练尚未启动。

## 运行入口

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& .\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider
& .\.venv\Scripts\python.exe -B -m scripts.run_formal_experiment --check-config
```

配置检查当前明确返回未就绪。补齐确认值后，正式入口先执行生产设置 smoke，通过后运行全部五次。

```powershell
& .\.venv\Scripts\python.exe -B -m scripts.run_formal_experiment
```

当前没有正式模型捕获率或五 seed 性能汇总可供报告。
