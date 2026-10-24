# Baseline v1 边界机制修复与验证

日期：2026-10-24。仅实现用户指定的 workspace 扩大、evader soft boundary steering，
保留 hard clip，不修改 MAPPO、网络、reward、observation、初始化、capture 或动力学限制。

**结论：指定公式已实现，evader hard clipping 明显减少，但尚未消除。**
本次没有继续调整 `boundary_strength`、margin、速度或加速度参数。

## 修改文件与接口

- `configs/baseline_3a1c.yaml`：bounds 改为 `[-300,-300,0]` 至 `[300,300,150]` m，
  增加 `evader.policy.boundary_margin_horizontal=50.0`、
  `boundary_margin_vertical=20.0`、`boundary_strength=2.0`，均为暂定默认值。
- `envs/evader_policy.py`：新增 `boundary_repulsion(position)`，合成后才归一化。
- `envs/pursuit_env.py`：从配置传递边界参数，在原有 hard clip 前计数，增加 info 键。
- `scripts/random_rollout.py`：输出硬裁剪及逃逸机贴边诊断。
- `tests/test_boundary_steering.py`：新增边界力测试。
- `tests/test_pursuit_env.py`：新增配置/裁剪统计测试；原小空间测试显式使用相容的小 margin。
- `tests/test_mappo.py`：仅同步测试中的 workspace 尺度预期，不改 MAPPO 算法或输入定义。
- `pytest.ini`：默认测试收集限定 `tests/`，避免历史实验源码快照产生同名模块冲突。
- `README.md`、`BASELINE_SPEC.md`：同步新边界机制与局限。
- 本报告及 `outputs/boundary_mechanism_validation/` 下的可复现诊断产物。

`reset()` / `step()` 返回结构、21D observation 和 24D state 均不变。
原有两参数 policy 构造方式继续可用。历史 YAML 完全不含三个新参数时保持旧策略，
以复现旧 checkpoint 的环境快照；部分缺省参数会报错，当前默认配置全部显式提供。

## Soft steering 公式

令 `m=[margin_horizontal,margin_horizontal,margin_vertical]`，逐轴计算：

```text
F_boundary = strength * (
    max(bounds_low + m - position, 0)
  - max(position - (bounds_high - m), 0)
) / m
total_repulsion = original_pursuer_repulsion + F_boundary
```

这与给定分段公式等价：中间区域为零；靠下界为正，靠上界为负，进入 soft margin
后线性连续增加。参数校验禁止 margin 非正、非有限或超过对应轴半跨度。
边界和角落结果有限，合力恰好为零时仍输出零，不添加随机破平局。

原 pursuer repulsion 的 `away / max(distance,epsilon)^3` 求和公式保持不变。
两种原始力相加后，才将总方向归一化并按原来的 `[a_horizontal,a_horizontal,a_vertical]`
缩放。之后仍使用同一 jerk、加速度与速度约束。不引入速度感知、预测或额外阻尼。

Pursuer 不计算 boundary steering，也没有新增 boundary reward/penalty。
所有飞机仍执行原有 `np.clip(position,low,high)`，保留速度，作为数值保护。

## 裁剪诊断定义

- `info["boundary_clip_count"]`：回合累计的飞行器-步裁剪事件总数。
- `info["boundary_clip_counts"]`：`[P0,P1,P2,E]` 各机累计计数的副本。
- `info["boundary_clipped"]`：当前 step 各机是否实际提出越界位置的 bool 数组副本。
- `info["boundary_steering_enabled"]`：是否配置了 evader steering。

统计在积分后、hard clip 前进行。单架机同帧多轴越界只算一次；刚好到边界但未越界
不算新裁剪。`reset()` 清零。另在随机 demo 中单独统计 poststep 的 exact boundary
占用时间，避免把“在墙上静止”和“仍提出越界运动”混为一谈。

## 测试结果

实际执行：

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
$env:MPLBACKEND = "Agg"
.\.venv\Scripts\python.exe -m pytest -q
```

最终 **97 passed in 11.92s**，包含原 52 项测试及新增 45 项边界/诊断用例。
覆盖中央零力、六面方向、margin 入口连续性、exact corners 有限性、原始力合成
顺序、强弱追击威胁、旧接口、非法参数、裁剪计数累计/副本隔离/reset、strict boundary
判定、未改变 pursuer 控制，以及完整随机运行。

首次默认 pytest 还收集到 `outputs/formal_run_100k/source_snapshot/tests/` 的历史测试
副本，出现模块重名收集错误。新增 `pytest.ini` 修复收集范围，未删除实验快照。
新增测试的一处 NumPy scalar YAML fixture 也已转换成 Python 原生列表后通过。
这些是测试工程修正，没有改变研究逻辑。

## 随机策略入口运行

实际执行：

```powershell
.\.venv\Scripts\python.exe -m scripts.random_rollout --config configs/baseline_3a1c.yaml --seed 0 --episodes 3
```

| seed | 步数 | 结束原因 | evader hard clips | 全部飞机 hard clips | 最长连续 evader 贴边 |
| --- | ---: | --- | ---: | ---: | ---: |
| 0 | 500 | timeout | 59 | 59 | 47 步 / 4.7 s |
| 1 | 500 | timeout | 70 | 98 | 48 步 / 4.8 s |
| 2 | 500 | timeout | 88 | 327 | 48 步 / 4.8 s |

全部回合完整运行、无 NaN/Inf。未因碰壁而增加新终止条件。

## 配对对照诊断

使用相同 seed 0–9 的随机 pursuer 动作，对三个验证场景各实际运行 10 回合：

- A：旧空间、无 steering，配置来自原正式实验保存的环境快照。
- B：新空间、无 steering，只用于区分扩大空间本身的作用。
- C：新空间及当前指定的 50/20/2 steering 参数。

B 的关闭仅发生在输出目录内的验证配置副本，删除三个新字段，没有改生产 YAML
或试调其他数值。30 回合全部为真实 500 步，无捕获，共 15000 步。
每步检查 shape、有限性、速度、加速度、jerk、位置界限，并独立重构裁剪前位置核对
计数。所有检查通过。

| 验证场景 | Evader hard clipping | 比例 | 最长连续贴边 |
| --- | ---: | ---: | ---: |
| A 旧空间，无 steering | 4150 / 5000 步 | 83% | 415 步 / 41.5 s |
| B 扩大空间，无 steering | 3150 / 5000 步 | 63% | 315 步 / 31.5 s |
| C 扩大空间并 soft steering | 650 / 5000 步 | 13% | 48 步 / 4.8 s |

C 相比 A 的 evader 裁剪事件减少 **84.34%**，相比 B 减少 **79.37%**。
C 各 seed 的 evader 裁剪数为 `59,70,88,63,66,71,48,90,47,48`。
C 距任意墙不超过 1 m 的状态共 751 步（15.02%），最长连续 56 步（5.6 s）；
这里的 1 m 只是离线诊断阈值，没有进入控制或 reward。

B/C 同一 seed 的每架 pursuer 裁剪数完全一致，总 pursuer 事件均为 1275；
seed 0 的 pursuer 状态与动作也逐元素相同，支持未向 pursuer 添加边界控制的检查。
随机动作独立于 observation，这一对照不意味着将来闭环 MAPPO 在两场景中会选择
相同动作；本轮未重训或修改 MAPPO。

![三个场景的真实 evader 位置对照](../outputs/boundary_mechanism_validation/evader_position_comparison_seed0.png)

## 仍然存在的限制

当前 hard clipping 仍占 13%，不能称为已消除边界 artifact，也还不是仅极偶发的
数值保护。Soft steering 能缩短贴边平台、使 evader 转回内部，但不能保证不触墙。

seed 0 在第 161 步进入 x soft margin 时，`x=250.963 m`、`vx=19.908 m/s`；
第 189–235 步仍在 x_max 被连续裁剪 47 步，首次碰壁时 `vx=14.038 m/s`，
最后一个裁剪步的 `vx=0.238 m/s`，之后才回到内部。
第 489–500 步又在 y_min 发生 12 步裁剪，回合结束时仍在边界。

一个可验证的限制是：水平最大速度 20 m/s、最大加速度 3 m/s²，即使忽略 jerk、
始终全力向内制动，也需要 `v²/(2a)=66.67 m` 的理想制动距离，已超过给定的
50 m 水平 margin。实际合力方向、jerk 和转向还可能使制动更晚。
这解释了为何指定位置型转向不能保证没有越界，而不是需要通过隐藏调参纠正的
程序错误。增大 boundary strength 也不会直接突破现有物理加速度上限。

本轮按要求固定所有参数，不继续调 strength、margin，不加速度感知或预测。
若后续要求接近零硬裁剪，需要另行决定如何验证缓冲距离或速度相关制动；
当前结论应保留为“显著缓解，但仍有贴边”，不能宣称问题完全消失。

## 产物与复现

目录：`outputs/boundary_mechanism_validation/`。

- `diagnose_boundary.py`：三场景配对随机运行脚本。
- `episodes.csv`、`summary.json`：全部 30 回合的原始指标与汇总。
- `manifest.json`：seed、配置来源、诊断定义、Git commit 和依赖版本。
- A/B/C 的 YAML 配置副本、seed 0 原始轨迹 NPZ。
- `seed0_behavior_observations.json`：碰壁步数与速度证据。
- `evader_position_comparison_seed0.png`：已视觉检查的对照图，无平滑或伪造数据。

Git commit 只读记录为 `23bf55576cca27944b4ee2ae5c53a70295c7dac3`；工作区仍含此前
未提交内容，本轮没有 commit、merge 或 push。历史 100k 实验、模型和视频不被覆盖，
它们属于原边界机制与原 workspace，必须与当前配置区分。
