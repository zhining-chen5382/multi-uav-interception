# Outputs 版本目录整理与100k final视频

日期：2026-11-08。按用户要求实际迁移12个历史目录；不是新增训练或研究版本。
随后按用户要求为七个根目录追加可读说明，版本ID与run ID均不变。当前实际名称见下表。

## 最新完整一回合视频

[v06-r01归一化100k final视频](../outputs/v06_输入归一化/v06-r01_20261108_seed000/continuation_100k/videos/final_full_episode.mp4)
直接绘制该模型既有确定性评估的完整轨迹，没有重新训练、重跑仿真或插值。
包括初态和全部282个step，共283帧，1280×720、10fps；仿真28.2秒，视频28.3秒。
视频标注三架pursuer、攻击机、基地、起止点、当前距离、全程最近距离、reward、碰撞和任务结局。
本回合最近42.891m，未捕获，最终敌方攻击成功。

已逐帧解码，帧数、帧率和分辨率正确；首/中/尾帧与渲染原帧比对通过。
Checkpoint SHA256为 `bfc292ba37d5705977599f732f104961dd9a61f76c27f8de7d12554181b5e03d`，
确认是100000步final。见 [video QA](../outputs/v06_输入归一化/v06-r01_20261108_seed000/continuation_100k/videos/video_qa.json)。

## 目录对应关系

日期按Asia/Shanghai；run编号沿用既有注册表。验证容器包含多个seed，不标成seed000训练。

| 原outputs子目录 | 新位置（相对outputs） | 内容 |
| --- | --- | --- |
| mappo_smoke_4096 | v01_旧追逃实验/v01-r01_20261024_seed000 | 旧追逃smoke |
| mappo_demo | v01_旧追逃实验/v01-r02_20261024_seed000 | 旧追逃隔离环境demo |
| formal_run_100k | v01_旧追逃实验/v01-r03_20261024_seed000 | 旧追逃100k |
| boundary_mechanism_validation | v02_边界机制验证/validation_20261024_boundary | 旧追逃边界验证 |
| point_defense_validation | v03_GoalDirected点防御/validation_20261024_point_defense | GoalDirected点防御验证和smoke |
| direct_attacker_validation | v04_DirectAttacker_旧奖励/validation_20261024_direct_attacker | Direct Attacker验证和smoke |
| direct_attacker_100k | v04_DirectAttacker_旧奖励/v04-r02_20261025_seed000 | Direct Attacker旧reward 100k |
| active_progress_reward_20k | v05_ActiveProgress奖励/v05-r01_20261025_seed000 | Active progress 20k |
| active_reward_100k | v05_ActiveProgress奖励/v05-r02_20261025_seed000 | 实际100k→200k，同一run |
| project_management_20261108 | _project_项目核对与管理资料/20261108_management | 历史整理和离线诊断 |
| project_resume_20261107 | _project_项目核对与管理资料/20261107_resume | 项目恢复核对 |
| visualization_check | _project_项目核对与管理资料/undated_visualization_check | 日期未确认的绘图检查 |

`outputs/v06_输入归一化/v06-r01_20261108_seed000/`内部层次不变：根目录是冻结20k阶段，
`continuation_100k/`是同一run续至总100k的产物。项目资料不增加研究版本号。
完整登记见 [EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md)；机器可读旧新对应见
[output_path_migration_20261108.json](output_path_migration_20261108.json)。
第二次根目录说明追加见 [output_folder_labels_20261108.json](output_folder_labels_20261108.json)。

## 数据完整性与路径兼容

迁移前记录2071个既有输出文件的SHA256，迁移后逐项核对新位置。
全部183个`.pt`文件、CSV、NPZ、日志、YAML快照、metadata、manifest、源码快照和历史Python脚本内容保持不变。
只允许Markdown导航更新；原文备份保存在本次管理目录的`markdown_before/`。
目录移动在同一工作区内进行，没有删除或合并实验数据。

当前README、BASELINE_SPEC、实验报告、注册表、生成索引和reward sanity默认输出路径已更新。
模型内嵌的旧路径仍记录实际训练时的位置，不为改文件夹名称而重存权重或改写历史配置。
旧路径可通过 `utils.output_paths.resolve_recorded_path` 和迁移表解析。

归一化历史分析脚本保留原字节；迁移后使用路径适配器加载，避免直接运行其旧硬编码路径：

```powershell
.\.venv\Scripts\python.exe -B -m scripts.replay_normalization_analysis --stage 20k --check-only
.\.venv\Scripts\python.exe -B -m scripts.replay_normalization_analysis --stage 100k --check-only
```

两条命令均已运行通过，不修改结果。将`--check-only`换为`--run`会重新生成对应阶段分析JSON和图表，
不训练或改权重；本轮没有执行重新生成。其他早期归档脚本同样保留原样，重放前应依据迁移表核对路径。
现行评估CLI直接支持迁移后的checkpoint；实际加载搬迁后的v05-r02近距模型复查一回合，
仍得到8.629391m、reward -303.109868、0捕获，和原记录一致。

## 测试与证据

- 全量pytest：**250 passed in19.80s**，新增4项路径边界、迁移解析和归档源码适配测试。
- 所有20个受保护环境/算法/配置核心文件保持原SHA256，没有改变reward、obs、动力学、网络或学习参数。
- 模型入口及SHA256均通过注册表校验；文档链接核验通过。
- 没有Git add、commit、merge或push。

证据保存在 [本次管理目录](../outputs/_project_项目核对与管理资料/20261108_output_organization/)：
[迁移前清单](../outputs/_project_项目核对与管理资料/20261108_output_organization/migration_plan.json)、
[完整迁移核验](../outputs/_project_项目核对与管理资料/20261108_output_organization/migration_verification.json)、
[pytest日志](../outputs/_project_项目核对与管理资料/20261108_output_organization/pytest.log)、
[搬迁模型复查](../outputs/_project_项目核对与管理资料/20261108_output_organization/moved_model_check/evaluation_summary.json)。
