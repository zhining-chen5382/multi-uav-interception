# Codex 项目协作规则

## 1. 项目目标

本项目用于研究多无人机协同拦截问题。

当前阶段的主要目标是：

- 搭建 3A1C 的 3D（三维）MAPPO（多智能体PPO）baseline（基线）。
- 跑通环境、训练、评估和可视化流程。
- 记录 baseline 的失败模式，为后续研究切入点提供依据。

在修改 baseline 相关代码前，应先阅读：

`BASELINE_SPEC.md`

其中定义了当前 baseline 的 observation（观测）、action（动作）、reward（奖励）、dynamics（动力学）和评估标准。

---

## 2. 开发基本原则

开发时应遵守以下原则：

- 优先保持代码简单、清晰、可解释。
- 不主动增加当前任务没有要求的复杂功能。
- 不因为“看起来更高级”而提前加入额外模块。
- 不在没有说明的情况下改变 baseline 的研究设定。
- 新功能应尽量模块化，方便后续替换或做 ablation study（消融实验）。

Baseline v1 的目标是建立稳定参考系统，而不是一次性完成最终研究方案。

---

## 3. 参数管理

所有实验参数、物理参数和训练超参数必须优先放入配置文件。

禁止将关键参数直接 hard-code（硬编码）在核心逻辑中。

包括但不限于：

- 环境尺寸
- `dt`
- 最大速度
- 最大加速度
- 最大 jerk（加加速度）
- capture radius（捕获半径）
- reward coefficients（奖励系数）
- episode length（回合长度）
- MAPPO hyperparameters（超参数）

默认配置文件放在：

`configs/`

如需调整参数，应优先修改或新增 YAML 配置，而不是修改核心 Python 代码。

---

## 4. Baseline 设定保护

除非任务明确要求，否则不要主动修改以下 baseline 核心设定：

- 3 pursuers + 1 evader
- 3D（三维）环境
- Baseline v1 无障碍
- Pursuer 使用 MAPPO
- Evader 使用 rule-based policy（规则型策略）
- Pursuer action 为 3D acceleration（三维加速度）
- Actor 使用 decentralized observation（分散式观测）
- Critic 使用 centralized global state（集中式全局状态）
- 第一版 reward 不包含 explicit encirclement reward（显式包围奖励）

如认为某项设定存在明显问题，应先说明问题及理由，不要直接替换。

---

## 5. 文件和模块职责

项目应保持清晰的模块划分。

建议结构：

```text
envs/
    pursuit_env.py
    dynamics.py
    evader_policy.py

algorithms/
    mappo/

configs/

scripts/
    train.py
    evaluate.py
    visualize.py

utils/

tests/

experiments/
```

环境逻辑、算法逻辑、训练脚本、评估和可视化不应混写在同一个大型文件中。

---

## 6. 修改代码时的要求

每次任务应尽量只解决一个明确问题。

例如：

- 搭建环境骨架
- 实现动力学
- 实现 observation
- 实现 reward
- 接入 MAPPO
- 添加评估指标
- 添加 3D 可视化

不要在完成一个任务时顺便大规模重构无关模块。

如果必须修改多个模块，应说明：

- 修改了哪些文件
- 每个文件为什么需要修改
- 是否改变了已有接口

---

## 7. 测试要求

修改以下模块时必须优先考虑补充或更新测试：

- dynamics（动力学）
- observation
- reward
- termination condition（终止条件）
- action scaling（动作映射）
- physical constraints（物理约束）

至少应验证：

- tensor / array shape 正确
- 数值范围合理
- 无 NaN / Inf
- physical constraints 正确生效
- episode 可以正常 reset / step
- capture condition 正确触发

在接入完整 MAPPO 前，应先保证 environment 可以通过 random policy rollout（随机策略运行）稳定执行。

---

## 8. 不要自动修改研究逻辑

Codex 不应在未明确要求的情况下：

- 自动重新设计 reward
- 自动新增 encirclement reward
- 自动加入 obstacles
- 自动加入 observation noise
- 自动加入 trajectory prediction
- 自动把 evader 改成 RL agent
- 自动升级到完整 quadrotor dynamics
- 自动改变任务难度

这些都属于研究决策，而不是普通代码优化。

---

## 9. 实验可复现性

所有正式实验应记录：

- config 文件
- random seed（随机种子）
- 代码版本
- 主要指标
- 输出目录

实验结果不应依赖手工修改代码中的参数。

如有随机初始化，应支持固定 seed。

---

## 10. Git 和生成文件

不要将以下内容默认提交到 Git：

- 大型 model checkpoint（模型检查点）
- training outputs（训练输出）
- `wandb/`
- cache（缓存）
- Python `__pycache__`
- 虚拟环境
- 本地 IDE 配置
- 临时文件
- 私有或外部原始资料

不要将不属于项目源代码的原始参考文件自动复制进仓库。

---

## 11. 文档同步

如果代码行为发生了会影响 baseline 定义的变化，应同步检查：

- `BASELINE_SPEC.md`
- `README.md`
- 配置文件

但不要在未经确认的情况下直接改变研究设定。

---

## 12. 输出要求

完成一个 Codex 任务后，应简要说明：

1. 完成了什么。
2. 修改了哪些文件。
3. 如何运行或测试。
4. 当前是否存在已知限制。
5. 是否有任何地方需要用户确认。

---

## 13. 当前阶段优先级

当前优先级依次为：

1. Environment 正确
2. Dynamics 正确
3. Observation / action 定义清楚
4. Reward 和 termination 正确
5. Random-policy demo 可以运行
6. MAPPO 可以训练
7. Evaluation 和 visualization 完整
8. 再考虑研究性扩展

不要跳过基础环境验证直接追求复杂算法结果。

---

## 14. 模型评估的视频交付

完成模型训练或确定性评估后，默认生成并在回复中直接展示 best 与 final 各一回合完整仿真视频，无需用户再次提出。纯单元测试不触发新的训练或模型性能评估。

- 两个模型优先使用相同初始场景，注明来袭角度、checkpoint 步数及回合结果。
- 若阶段末文件名为 `last.pt`，应说明 final 视频使用该文件，不改名或覆盖原权重。
- 选择成功示例时明确标注选择方法，并保留整体评估捕获率，不能把单例成功解释为整体成功率。
- 保留完整初始至终止轨迹、视频来源 manifest 和解码检查结果。

## 15. 训练后轨迹演化图

每次训练结束后，默认生成并交付 XY 俯视轨迹演化图和逐 checkpoint 指标 CSV。每个 checkpoint 在同一个明确标注的固定 deterministic 场景下运行完整回合，图中从浅到深表示实际训练步数。缺失的初始或中间 checkpoint 不重造，用真实可用步数并记录抽样映射。只读推理和制图不能改动训练算法、环境定义、奖励或原权重。
