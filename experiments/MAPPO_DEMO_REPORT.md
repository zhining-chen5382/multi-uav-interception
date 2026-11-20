# MAPPO Baseline v1 demo 实现与验证记录

路径说明（2026-11-08）：本文链接及命令示例已更新为版本目录；冻结快照、原始日志、metadata 与权重内嵌旧路径保留，见 [目录迁移表](output_path_migration_20261108.json)。

日期：2026-10-24。此记录是流程验证，尚不是收敛或协同行为的研究结论。

## 1. 完成范围与文件

新增：

- `algorithms/__init__.py`、`algorithms/mappo/__init__.py`：包入口。
- `algorithms/mappo/networks.py`：共享 Actor、集中式 Critic、压缩高斯分布。
- `algorithms/mappo/buffer.py`：按团队时间步存储的 rollout buffer 与 GAE。
- `algorithms/mappo/mappo.py`：PPO 更新、动作接口、checkpoint 保存加载。
- `algorithms/mappo/normalization.py`：默认关闭的固定输入缩放。
- `configs/mappo_baseline.yaml`：训练、网络、评估、绘图参数及环境 YAML 引用。
- `scripts/train_mappo.py`：采样、训练、指标、存档。
- `scripts/evaluate_mappo.py`：独立确定性评估与指标导出。
- `scripts/visualize_mappo.py`：独立确定性轨迹与静态 3D 图。
- `utils/__init__.py`、`utils/experiment.py`、`utils/evaluation.py`：配置、种子、复现元数据、评估工具。
- `tests/test_mappo.py`、`tests/test_mappo_pipeline.py`：算法及 CLI 集成测试。
- 本报告。

修改 `README.md` 与 `requirements.txt`，补充使用说明及 PyTorch、matplotlib 依赖。
既有环境接口、物理 YAML、奖励、捕获定义和 `BASELINE_SPEC.md` 均未修改。

## 2. 网络结构

- 三架追击机共享同一 Actor：`21 → 128(Tanh) → 128(Tanh) → mean(3)`。
- 另有全局可学习的 `log_std(3)`；初始值 `-0.5`，限幅范围 `[-5,2]`。
- 采样 `z ~ Normal(mean,std)`，环境动作 `a=tanh(z)`；评估使用 `tanh(mean)`。
- 集中式团队 Critic：`24 → 256(Tanh) → 256(Tanh) → scalar V`。
- 正交初始化，隐藏层增益 `sqrt(2)`，Actor/Critic 输出层增益分别 `0.01/1`。

## 3. Actor 与 Critic 输入

Actor 每架只读取既有 21D 局部观测；Critic 读取既有 24D 全局状态。
独立评估与绘图只向 Actor 传 observation，无需集中式状态。
没有增加 agent ID、预测、通信、加速度历史或其他观测维度。

## 4. Rollout 与 PPO

单环境顺序采样，每步存三份局部观测与动作 log probability，但仅存一次团队
reward 和 value。保存压缩前动作以稳定重算含 tanh Jacobian 的 log probability。
Entropy 用当前压缩分布的新样本估计，保留重参数化梯度。

GAE：捕获不 bootstrap；超时用 reset 前最终状态 bootstrap 并断开跨回合链；
rollout 中途截断用下一状态价值。按配置标准化 advantage。
默认每次采样 512 步，更新 4 epochs；每批 64 个团队时间步展开为 192 个 actor
样本和 64 个 critic 样本。三个 actor 的 PPO ratio 分别计算，使用共同团队优势。
包含 clipped policy loss、平方 value loss、entropy bonus、梯度范数裁剪。

## 5. 全部新增配置参数

完整键名、默认值和作用已逐项列在 `README.md` 的“全量训练配置参数”表，
实际文件为 `configs/mappo_baseline.yaml`。覆盖范围为：

- 环境配置引用、seed、device、CPU 线程数、输出目录。
- 总训练步数、rollout 步数、日志与 checkpoint 间隔、捕获率窗口。
- Actor/Critic 隐藏层、log standard deviation 初值和上下限、正交初始化增益。
- 两个学习率、Adam epsilon、gamma、GAE lambda、PPO clip、epochs、minibatch size。
- Entropy/value loss 系数、梯度裁剪、advantage 标准化与数值 epsilon。
- Observation/global state 固定缩放开关、显式 reward scale。
- 评估回合数/seed、绘图 seed/dpi/尺寸。

输入缩放默认关闭，reward scale 默认 `1.0`，无运行均值或方差归一化。
训练配置不复制物理参数，checkpoint 单独保存环境快照用于独立评估。

## 6. 测试

仓库根目录：

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
.\.venv\Scripts\python.exe -m pytest -q
```

实际结果：**52 passed in 29.13s**。包括原有环境测试、网络/分布/GAE、单次真实更新、
checkpoint 后确定性动作与价值一致，以及三个 CLI 的端到端调用。
不含耗时长训练；CLI 测试使用 16 步训练与小网络，环境仍为原始配置。

## 7. 短训练

本次已实际执行：

```powershell
.\.venv\Scripts\python.exe -m scripts.train_mappo --config configs/mappo_baseline.yaml --total-steps 4096 --output-dir outputs/v01_旧追逃实验/v01-r02_20261024_seed000
```

8 次更新、8 个完整回合、96 步未完成回合。完整回合平均团队回报 `-141.439959`，
训练窗口捕获率 `0/8`，8 个完整回合均为 500 步。所有记录的 update 指标有限。
最终一次 actor loss `-0.00946088`、value loss `5.76212791`、entropy `1.78363825`。
未完成回合保存在 checkpoint 状态中，不作为第九个完整回合写入 CSV。

另一次先前的 4096 步运行保留在 `outputs/v01_旧追逃实验/v01-r01_20261024_seed000/`，
该运行起初继承全局 NumPy；当前交付演示以完全隔离依赖的 `outputs/v01_旧追逃实验/v01-r02_20261024_seed000/` 为准。

## 8. 正式训练入口

```powershell
.\.venv\Scripts\python.exe -m scripts.train_mappo
# 或显式参数；默认总步数为 100000
.\.venv\Scripts\python.exe -m scripts.train_mappo --config configs/mappo_baseline.yaml --output-dir outputs/formal_seed0
```

未实际运行 100000 步正式实验。重复实验使用新的输出目录。
已有同名训练产物时会拒绝覆盖；`best.pt` 按训练窗口指标选择，不是独立评估最优保证。

## 9. 独立评估

已实际执行：

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_mappo --checkpoint outputs/v01_旧追逃实验/v01-r02_20261024_seed000/checkpoints/final.pt --episodes 3
```

| 指标 | 实测值 |
| --- | --- |
| 评估回合数 | 3 |
| 捕获率 | 0% |
| 平均团队回报 | -333.731004 |
| 平均回合步数 | 500 |
| 平均成功捕获时间 | null，无成功回合 |
| 平均回合最小敌我距离 | 30.0 m |
| 碰撞回合占比 | 100% |
| 碰撞步数 | 78 / 1500 步，每回合 26 步 |
| 碰撞步比例 | 5.2% |
| 平均追击机两两距离 | 63.342170 m |

三个回合使用相同固定场景与确定性策略，轨迹相同，不能当作三个独立测试场景。
结果保存在该 run 的 `evaluation_metrics.csv` 和 `evaluation_summary.json`。

## 10. 3D demo

已实际执行：

```powershell
.\.venv\Scripts\python.exe -m scripts.visualize_mappo --checkpoint outputs/v01_旧追逃实验/v01-r02_20261024_seed000/checkpoints/final.pt
```

生成 `outputs/v01_旧追逃实验/v01-r02_20261024_seed000/trajectory_3d.png` 与 `trajectory.npz`。
已检查绘图效果：四条可辨轨迹、起终点、坐标轴、图例与真实 captured/steps 标题齐全。
NPZ 包含 `(501,4,3)` 位置、`(501,4,6)` 状态、`(500,3,3)` 动作、时间和逐步指标。

## 11. 实际运行环境与产物

- Windows，Python `3.8.8`，CPU。
- PyTorch `2.4.1+cpu`，NumPy `1.24.4`，matplotlib `3.7.5`。
- PyYAML `6.0.3`，pytest `8.3.5`。
- 完全隔离的 `.venv`，`pip check` 通过。
- 初期绘图遇到全局 Anaconda/OpenMP 冲突；通过隔离安装依赖解决，
  没有设置 `KMP_DUPLICATE_LIB_OK`。
- `metadata.json` 保存运行时 Git 版本 `23bf55576cca27944b4ee2ae5c53a70295c7dac3`、
  `git_dirty=true`、seed、有效配置与路径；代码修改尚未提交。
- `checkpoints/final.pt`、`best.pt`、CSV、JSON、NPZ 和 PNG 位于 `outputs/v01_旧追逃实验/v01-r02_20261024_seed000/`。
- 测试中的逐更新 checkpoint 保存/加载也已通过。输出和虚拟环境均不进入 Git。

## 12. 已知限制

短训练未学到有效拦截，也没有证明优于随机策略。训练回报使用随机动作采样，评估
使用确定性动作，两者统计口径不同；不要将其数值差异直接解释为学习是否成功。
可能影响学习的因素包括样本很少、原始输入尺度较大、双方能力相同且逃逸机初始
领先、jerk 历史未包含在既定输入中，以及位置裁剪造成的贴边行为。

当前为前馈网络、单环境顺序采样，没有多 seed/场景泛化评估、动画或 resume 训练入口；
checkpoint 已保存 optimizer、RNG、环境与计数状态，但加载不会自动继续训练。
没有验证 CUDA 或正式长训练。没有通过改变环境、reward、初始化或降低难度
来美化结果。具体物理简化仍以原 README 环境说明为准。

## 13. 研究设定与版本管理

既有 3A1C、21D/24D 输入、规则逃逸机、奖励与动力学保持原定义。
输入归一化与 reward 缩放均显式配置，默认保持原数值尺度。
当前没有待确认的研究设定。未执行 Git commit、merge 或 push。
