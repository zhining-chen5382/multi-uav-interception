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

当前项目处于 Baseline v1 搭建阶段。

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