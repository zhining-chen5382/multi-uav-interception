# 公开图像来源

这些图像来自v16的k=0.1、Seed1001实验，均直接复制现有PNG，未裁剪、重采样、平滑或重新推理。它们用于说明历史训练行为，不替代完整实验数据或跨Seed结论。

| 文件 | 范围 |
| --- | --- |
| [trajectory-evolution-v16.png](trajectory-evolution-v16.png) | 0–500k，124个真实策略时点；同一固定0°确定性场景的XY轨迹；非角域成功率测量 |
| [training-capture-reward-v16.png](training-capture-reward-v16.png) | 0–500k训练记录和±30°固定100角度验证；标明300k续训分界 |
| [actor-critic-loss-v16.png](actor-critic-loss-v16.png) | PPO update 1–123的实际Actor/Critic Loss |

[provenance.json](provenance.json)记录原始本地相对路径、字节数和SHA256。三个文件总计801,935字节；没有包含模型、视频或逐步轨迹数据。

对应任务为1v1 PPO，不是三架我方机的协同成绩。该Seed的310k Best在原±30°500场景为46.6%，500k Final为0%；另一个限定负角域测试的100%不能作为整个任务的总体成功率。原500场景曾参与奖励离线标定，仅用于历史受控诊断。

本地有完整输出时可按来源路径定位原图；仅克隆公开代码时仍可读取这三张精选图和对应源代码/研究报告。
