# v09 来袭方向课程：实现协议与验证记录

## 状态与目的

当前状态：**流程实现和测试通过；正式 v09-r01 已完成 ±10° 阶段 100k，未达晋级条件，按预算正常停止**。
专项 41 项、全量 340 项测试通过，全量用时 19.48s。实际训练 200 updates、354 完整回合，其中随机探索捕获 3 次。
每 10k 固定 100 角度评估，总计 1000 回合；历史 best 为 30k 的 9% 捕获，100k 为 0%。
±30°/±60°/±90° 未运行。完整结果见 [正式运行报告](V09_DIRECTION_CURRICULUM_RUN_REPORT.md)。
本文记录工程协议与已执行检查，不把短流程测试、配置检查或预期行为当作模型性能结果。

v08 的 92160 步成功策略在原固定初态能够捕获，但冻结后的 90° 逆时针测试失败。
v09 因此研究：保持 Direct Attacker 和原物理任务不变，通过逐步扩大训练方位分布，
能否学会根据不同来袭方向控制拦截机。本轮只增加这一课程流程，不加入新的奖励、
观测、动作、网络、敌机规避或动力学课程。

v09 的 ±10° 阶段从零训练，**不使用 v08 的 92160 步模型初始化**。
v08 及此前训练配置、历史模型和结果保留。

## 文件与职责

| 文件 | 职责 |
| --- | --- |
| `configs/mappo_direction_curriculum.yaml` | v09 训练配置、阶段范围、预算、评估与晋级规则 |
| `utils/direction_curriculum.py` | 初始方位几何变换、独立环境子类、固定评估角度、best 排序、晋级计数、权重/优化器/RNG 继承 |
| `scripts/train_direction_curriculum.py` | 调度阶段、训练更新、确定性评估、遗忘监控、checkpoint 及实际计数 |
| `tests/test_direction_curriculum.py` | 几何、采样、排序、状态继承、隔离和极短课程流程验证 |
| `scripts/experiment_index.py`、`experiments/experiment_registry.yaml` | v09 版本登记及版本专属训练入口；旧版本命令保持兼容 |
| `README.md`、`BASELINE_SPEC.md` | 入口、协议与建模限制同步 |
| 本文 | 实现依据、验证状态与正式实验登记入口 |

训练复用 `MAPPOAgent`、`RolloutBuffer`、原训练回合统计及 `utils.evaluation.run_episode` /
`summarize_episodes`。原 `scripts/train_mappo.py` 的训练窗口 best 逻辑与既有环境接口保持原样。

## 唯一研究变量：初始来袭方位分布

`DirectionCurriculumEnv.reset` 只替换四机参考初态中最后一架敌机的位置。
令原始敌机位置为 `p0`、基地为 `b`、角偏移为 `θ`：

```text
p_initial.xy = b.xy + [[cosθ, -sinθ], [sinθ, cosθ]] · (p0.xy - b.xy)
p_initial.z  = p0.z
θ ~ Uniform(-A, +A)
```

每回合从原始配置重新旋转，不将上一次位置当成下一次的旋转参考。
保持敌机到基地的三维距离、水平半径及绝对高度；三架我方初始位置、所有初速度和
reset 清零的加速度历史均不变。原场景 `b=[0,0,0]`、`p0=[497.49371855331,0,50]`，
因此 `p_initial=[497.49371855331 cosθ, 497.49371855331 sinθ, 50]`，
到基地距离约 500 m。四阶段整个方位区间均在原三轴 ±750 m 数值空间内，不需要裁剪初态。

每个训练回合使用独立 NumPy Generator 连续均匀采样，不从评估角度网格抽样。
基础 seed 为 0，阶段采样 seed 按 `seed + stage_index * stage_sampling_seed_stride` 计算，
默认依次为 0、10000、20000、30000；stage_index 从 0 开始。
敌机仍按原 Direct Attacker 速度跟踪律朝基地飞行，不使用我方位置、规避项或随机动作。

## 阶段、评估与晋级

| 阶段 | 方位范围 | 初始化 | 最大实际训练步数 |
| --- | --- | --- | --- |
| 1 | ±10° | 全新 Actor、Critic、Adam | 100000 |
| 2 | ±30° | 阶段 1 历史 best-eval | 100000 |
| 3 | ±60° | 阶段 2 历史 best-eval | 100000 |
| 4 | ±90° | 阶段 3 历史 best-eval | 100000 |

每阶段在实际训练环境步达到 10000、20000、…、100000 时评估。
评估之前完成该段 rollout 的 PPO 更新，再对更新后的同一组权重执行确定性推理。
每次使用 `np.linspace(-A,A,100,endpoint=True)`，固定 100 个角度并等间距覆盖阶段范围。
同一阶段各评估点复用同一网格，episode seed 为 1000 至 1099。

课程评估数量读取 `curriculum.evaluation_episodes=100`。新配置的通用 `evaluation.episodes`
也记为 100，避免数量歧义；普通固定场景的 `evaluate_mappo` 不会自动旋转初态，不能代替课程评估。
原 v08 配置没有修改。

当前阶段连续两次确定性捕获率 ≥80% 才晋级。未达标时 streak 清零。
达到 100k 仍不满足该条件则停止全课程，不跳过该阶段；若在 100k 恰好出现第二次连续达标，
先完成晋级判定，允许进入下一阶段。最后阶段达标则结束为 completed。
因此 400k 是四阶段预算上限，不是必须耗尽的步数，也不保证运行一定到达 ±90°。

### 历史 best-eval 选择

只使用**当前阶段固定评估集**，按下列顺序作严格字典序比较：

1. 拦截成功率更高。
2. 敌方攻击成功率更低，即防御失败率更低。
3. 成功回合平均捕获时间更短。
4. 平均回合最小敌我距离更小。
5. 碰撞回合率 `collision_episode_rate` 更低。

完全并列保留较早策略。reward 不进入排名；碰撞回合率与碰撞步比例不混用。
无捕获时平均捕获时间保存为 null，比较键内部使用有限占位值；不生成 NaN/Inf。
每个评估点立即保存对应的 `evaluated_XXXXXX.pt`，若胜出则原样复制为 `best_eval.pt`，
保证评估指标与保存权重一致。

### 遗忘监控

从第二阶段起，每个评估点还以相同当前模型测试**前一阶段**的固定 100 角度集，
保存为 `previous` 角色；当前阶段保存为 `current`。previous 仅提供遗忘信号，
不参与 best 排序、streak 或晋级，不会因为前阶段表现好而晋级，也不会因遗忘阻止晋级。
目前不是同时监控所有历史阶段。

## Checkpoint 与优化器继承

`load_best_eval_checkpoint` 加载上一阶段选中的历史 `best_eval.pt`：

- Actor 和 Critic 全部参数及 buffer。
- Actor Adam 与 Critic Adam 的全部状态，包括参数组、step、exp_avg、exp_avg_sq。
- 所选 checkpoint 的 Python、NumPy、Torch 和可用时 CUDA RNG，恢复发生在网络构造之后。

加载时检查 network、ppo、normalization 配置一致及权重/优化器状态有限，随后恢复训练模式。
阶段切换不继承旧环境或未结束回合：重建环境、rollout buffer、回合统计、评估 streak，
并采用新阶段的独立角度采样 seed，不跨阶段拼接 GAE。

课程的 `env_steps`、`update`、`episodes` 统计实际完成的全部训练工作。
如果晋级时历史最佳出现在较早评估点，实际计数不会倒退到那个点；优化器状态仍精确继承
所选历史模型。来源 checkpoint 的步数、父路径和 SHA256 单独保留，用于区分实际消耗与权重继承链。

已有阶段间继承**不等于完整运行的中断恢复**。当前尚未实现 whole-run resume CLI 或自动恢复；
非空输出目录会被拒绝覆盖。中断时保留已写入的模型和 CSV，后续恢复需要单独设计，
不能重新运行同一命令并声称无缝续训。

## 精确 10k 调度与隔离

原 rollout 容量仍为 512；为在精确 10000 步评估，每个完整区间收集 19×512+272 步。
最后一段切短后正常用下一状态 value bootstrap 并做 PPO 更新；真实 terminal 不 bootstrap，
timeout 从 reset 前最终状态 bootstrap，并切断跨回合 GAE。没有伪造终止或修改 buffer 公式。
这种调度的更新次数与整段不断按 512 更新的旧训练不同，应在分析时保留该区别。

`evaluate_policy` 使用独立评估环境、`torch.inference_mode()` 和确定性 Actor。
finally 中恢复全局 RNG 与原网络模式，不重置/推进训练环境或训练角度采样器，不执行 Adam step。
保存 checkpoint 与被评估权重之间没有额外 PPO 更新。

## 与 v08 保持一致的配置

| 项目 | v09 使用值 |
| --- | --- |
| 环境配置 | `configs/baseline_capture500.yaml`，与 v08 相同 |
| `dt` / 最大回合长度 | 0.1 s / 500 步 |
| 捕获 / 攻击半径 | 严格 `<5 m` / 严格 `<10 m`，同帧捕获优先 |
| Actor / Critic | 21→128→128→3 / 24→256→256→1，原 Tanh 与初始化 |
| 初始 / 最小 / 最大 log_std | -0.5 / -5 / 2 |
| Actor / Critic lr，Adam epsilon | 0.0003 / 0.001，0.00001 |
| Gamma / GAE lambda / PPO clip | 0.99 / 0.95 / 0.2 |
| PPO epochs / minibatch | 4 / 64 |
| Entropy / value coefficient / grad clip | 0.01 / 0.5 / 0.5 |
| Advantage normalization / epsilon | true / 1e-8 |
| 观测 / global state 输入缩放 | false / false |
| Reward scale | 1.0 |
| Rollout 容量 / 保存间隔 | 512 / 每 10 updates 周期保存 |
| Seed / device / CPU threads | 0 / cpu / 1 |

奖励仍为原 v08 共享团队奖励（`d_(t+1)` 为更新后的最近敌我距离）：

```text
R_t = -0.01*d_(t+1)*dt - 0.2*dt
      - 5*I(any pursuer pair distance < 2 m)
      + 500*I(captured)
      - 100*I(enemy attack success and not captured)
```

双方水平/爬升/下降速度上限仍为 20/8/6 m/s，水平/垂直加速度 3/2 m/s²，jerk 5 m/s³。
未改变 observation、动作到物理加速度映射、离散捕获判定、hard clip 或碰撞定义。

## 输出与运行命令

默认独立目录：`outputs/v09_来袭方向课程/v09-r01_20261109_seed000/`。

```text
training_config.yaml, environment_config.yaml, metadata.json
evaluation_history.csv
curriculum_state.json, summary.json（阶段/课程结束时写入）
stage_01_pm10/（后续阶段以此类推）
  training_metrics.csv, updates.csv, evaluation_angles.json, summary.json
  checkpoints/initial.pt, update_*.pt, evaluated_*.pt, best_eval.pt, last.pt
  evaluations/at_010000/current/
    evaluation_metrics.csv, angles.json, summary.json, checkpoint.json, first_episode.npz
  evaluations/at_010000/previous/（从第二阶段起）
```

逐回合训练记录含实际角度与初始敌机 XYZ；逐回合评估记录含固定角度、seed、实际初态和任务结果。
metadata 记录配置来源、Git 代码版本/dirty 状态和依赖版本。旧 v08 的 best/final 文件不被覆盖。

```powershell
.\.venv\Scripts\python.exe -B -m scripts.train_direction_curriculum --config configs/mappo_direction_curriculum.yaml --check-config
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
.\.venv\Scripts\python.exe -B -m pytest tests/test_direction_curriculum.py -q
.\.venv\Scripts\python.exe -B -m pytest -q
.\.venv\Scripts\python.exe -B -u -m scripts.train_direction_curriculum --config configs/mappo_direction_curriculum.yaml
```

`--check-config` 只校验/打印调度配置，不创建训练目录，不启动训练。
可用 `--output-dir` 指定新的空目录、`--seed` 显式改变基础训练 seed。

## 验证登记

测试覆盖几何半径/高度和非原点基地、连续均匀采样与 seed 复现、只改变敌机位置、
固定 100 角度覆盖、全部 best 排序与有限性、连续达标与失败清零、双 Adam/网络/RNG 恢复、
历史 best 而非 last 的阶段继承、实际计数不回退、未达标停止、上一阶段监控隔离、
确定性评估不改变训练状态。极短集成测试缩小网络、预算和评估集以检查工程流程，
这些 fixture 不是正式 v09 研究配置或性能证据。

| 检查项 | 实际结果 |
| --- | --- |
| `--check-config` | 已通过，打印 v09 配置计划，未创建训练目录 |
| v09 专项 pytest | 41 passed |
| 全量 pytest | 340 passed in 19.48s（原 299 项 + 新增 41 项） |
| 正式运行开始 | UTC 2026-11-09T05:13:33.487676+00:00；首阶段 fresh，无外部 checkpoint |
| 正式运行状态、实际步数/updates/episodes | 进程 exit 0；stopped_stage_budget；100000 / 200 / 354，探索捕获 3 次 |
| 历史 best 与末次 | 30k：9/100；100k：0/100；best_eval 未被末次覆盖 |
| 后续阶段 | 未开始，未发生正式跨阶段继承；继承/遗忘监控在短集成测试中验证 |
| 数值检查 | 实际 13 个 CSV、33 个 PT 全部有限，无 NaN/Inf；不表示策略性能稳定 |

全量测试原始日志：[pytest.log](../outputs/v09_来袭方向课程/_implementation_checks_20261109/pytest.log)。
用户在实现阶段补充授权：准备好后直接进行正式训练，不需要再次确认。
开训前另核对 4186 个历史实验输出文件和 58 个既有核心源文件保持原 SHA256，
实际训练源码 64 文件另存为 run/source_snapshot，配置、测试和核对依据保存在 run/preflight。

## 已知分析边界

固定评估角度用于选择 best 和晋级，因此也是训练决策集；完成课程后仍需独立留出角度、
额外种子或其他授权扰动来验证泛化，不能将调度评估直接称为无偏泛化结论。
本轮只随机方位，尚未覆盖敌机机动/规避、我方队形、敌机初速度、高度或距离变化。
原 21D Actor 缺少基地相对关系、21D/24D 输入缺少加速度历史等限制仍保留。
只有一次 seed0 运行也不足以证明课程收益或显式协同；应基于实际终局、轨迹和安全统计分析。
