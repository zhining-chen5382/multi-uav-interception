# Multi-UAV Cooperative Interception

## 项目简介

本项目研究 multi-UAV cooperative interception（多无人机协同拦截）问题。

当前阶段的目标是先建立一个简单、稳定、可复现的 MAPPO（多智能体PPO）baseline（基线），用于验证三架追击机协同拦截一架逃逸机的基本可行性，并为后续研究改进提供参考。

当前基础场景为：

`3A1C`

即：

- 3 架 pursuers（追击机）
- 1 架 evader（逃逸机）
- 3D（三维）连续空间
- 暂无 obstacles（障碍物）
- pursuers 使用 MAPPO
- evader 使用 rule-based policy（规则型策略）

---

## 当前阶段

当前已搭建 Baseline v1 的 NumPy 3D 环境骨架，可运行随机策略；尚未接入 MAPPO。

主要任务包括：

- 搭建 3D environment（三维环境）
- 实现简化 flight dynamics（飞行动力学）
- 定义 observation（观测）
- 定义 action（动作）
- 定义 reward（奖励）
- 实现 random-policy rollout（随机策略运行）
- 接入 MAPPO
- 完成 evaluation（评估）
- 完成 3D trajectory visualization（三维轨迹可视化）

---

## Baseline v1 设计

当前 baseline 使用：

- 3D point-mass dynamics（三维质点动力学）
- 3D acceleration control（三维加速度控制）
- decentralized observation（分散式观测）
- centralized critic（集中式评论家）
- parameter sharing（参数共享）
- shared team reward（共享团队奖励）

详细技术定义见：

`BASELINE_SPEC.md`

---

## 项目结构

计划中的项目结构：

```text
multi-uav-interception/
│
├── README.md
├── BASELINE_SPEC.md
├── AGENTS.md
├── .gitignore
│
├── configs/
│
├── envs/
│
├── algorithms/
│   └── mappo/
│
├── scripts/
│
├── utils/
│
├── tests/
│
├── experiments/
│
└── outputs/
```

---

## 环境骨架的运行与测试

在仓库根目录执行（Python 3.8 或以上）：

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python -m scripts.random_rollout
```

若本机额外安装的 pytest 插件导致启动长时间无输出，可在当前 PowerShell 会话禁用
第三方插件自动加载后再运行测试（本项目测试不依赖额外插件）：

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -m pytest -q
```

随机策略 demo 默认读取 `configs/baseline_3a1c.yaml` 中的 `rollout.seed` 和
`rollout.episodes`，每个回合输出一行 JSON，包括种子、步数、捕获结果、累计团队奖励、
最近敌我距离和发生追击机碰撞的步数。可显式覆盖：

```powershell
python -m scripts.random_rollout --config configs/baseline_3a1c.yaml --seed 42 --episodes 5
```

这是环境运行检查，暂不保存正式实验记录、模型或轨迹文件。

基础接口：

```python
from envs import PursuitEnv

env = PursuitEnv("configs/baseline_3a1c.yaml")
observations, info = env.reset(seed=42)
terminated = truncated = False
while not (terminated or truncated):
    actions = env.sample_actions()
    observations, rewards, terminated, truncated, info = env.step(actions)
```

- `actions`：形状 `(3, 3)`，范围 `[-1, 1]`，有限越界值会裁剪；NaN/Inf 会报错。
- `observations`：形状 `(3, 21)`，依次为自身速度、相对逃逸机位置、相对逃逸机速度、
  两名队友相对位置、两名队友相对速度。队友始终按固定编号升序排列，暂未归一化。
- `rewards`：形状 `(3,)`，三个值相同，即规格中的共享团队奖励。
  每步只要存在不安全的追击机对，就扣一次 `collision_penalty`。
- `env.get_state()` / `info["global_state"]`：形状 `(24,)`，顺序为追击机 0、1、2、
  逃逸机，每架均为 `[x, y, z, vx, vy, vz]`。
- `terminated`：任意追击机与逃逸机距离严格小于捕获半径。
  `truncated`：达到配置步数上限；同时捕获时优先记为捕获。
  结束后需重新 `reset()` 才能继续 `step()`。
- `info` 还提供 `min_distance`、`captured`、`step_count`、`collision_pairs`，
  `step()` 另外提供 `reward_components`（`capture/progress/time/collision`）。

所有环境尺寸、初始位置与速度、双方物理约束、捕获/安全距离及奖励系数均来自 YAML。
本阶段使用固定初始队形，种子控制随机动作；每次 reset 将加速度历史清零。
配置中新增的边界、初始状态、逃逸机参数和安全距离均为暂定默认值。

当前的简化处理与限制：

- 边界仅裁剪位置并保留速度，不模拟墙面碰撞或反弹，也不额外给予边界奖励或终止。
  这可能使飞行器沿边界停留，后续行为分析应注意这一点。
- 动作经缩放、水平模长/垂直加速度约束、三维模长 jerk 限制后更新速度，
  再限速并使用新速度更新位置。jerk 约束针对加速度指令历史；
  触发限速时，实际 `dv/dt` 可能与该指令不同。
- 逃逸机采用所有追击机的逆距离平方斥力合成方向，不含边界转向。
  合力恰好为零时输出零指令，不加入随机破平局机制。
- 捕获和碰撞按每个仿真步更新后的位置判定，未做步内连续接触检测。
- 接口为普通 NumPy 类，未提供 Gymnasium/PettingZoo 适配或空间对象。
  MAPPO、正式评估导出、3D 可视化和观测归一化尚未实现。
  本次未加入障碍物、显式包围奖励、观测噪声或轨迹预测。

---

## 开发原则

本项目当前阶段优先保证：

- simple（简单）
- understandable（可理解）
- configurable（可配置）
- reproducible（可复现）

在 baseline 稳定前，不提前加入不必要的复杂功能。

---

## 当前研究关注

Baseline 完成后，计划重点分析：

- MAPPO 是否能够自发形成 cooperative behavior（协同行为）
- 是否出现 flanking（夹击）或 encirclement（包围）
- baseline 在哪些 evader strategy（逃逸策略）下容易失败
- observation / reward 是否存在明显不足
- 哪些 failure modes（失败模式）可以作为后续研究切入点

---

## 潜在扩展方向

后续可能研究：

- Explicit Encirclement Prior（显式包围先验）
- Unknown / Adaptive Evader Strategy（未知 / 自适应逃逸策略）
- Adversarial Evader Learning（对抗逃逸者学习）
- 3D Obstacle Environment（三维障碍环境）
- Curriculum Learning（课程学习）
- Trajectory Prediction（轨迹预测）
- Observation Noise（观测噪声）
- More Realistic Flight Dynamics（更真实飞行动力学）
- Sim-to-Real Transfer（仿真到实机迁移）

---

## 状态

当前状态：

**Baseline v1 — Initial Development**
