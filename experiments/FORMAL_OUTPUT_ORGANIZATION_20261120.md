# 单机PPO实验目录整理验收

2026-11-20将`outputs/formal_ppo_v1`原23个一级目录归入v14–v17、离线标定与工程验收。没有新训练、回合性能评估或研究参数修改。

**[打开实验总览](../outputs/formal_ppo_v1/00_实验总览.html)** · [全项目版本/运行/模型索引](EXPERIMENT_INDEX.md) · [分类目录源](formal_ppo_output_catalog.yaml) · [旧路径映射](formal_ppo_output_path_migration_20261120.json)

## 最终结构

| 目录 | 内容 |
| --- | --- |
| v14_单机PPO_原始奖励 | 5个Seed，从零300k及续训500k；坐标与学习行为分析 |
| v15_历史最小距离奖励 | record-minimum near，λ20m、β100；3个Seed各300k |
| v16_线段偏离成本_k0p1 | 3个Seed各300k；Seed1001续训500k；310k Best的方向测试及模型/速度档案 |
| v17_主动接近归因_k0p01 | β100与β200各3个Seed、300k；两组与历史对照 |
| _离线奖励标定 | 普通差分、历史最小距离、β、线段成本、主动归因及配比重放 |
| _工程验收 | 初始工程、归一化、300k与500k配置Smoke、恢复验收、此次整理证据 |

各版本`00_图表总览/index.html`可按轨迹、Reward、Loss和优化诊断筛选。每个run的`00_结果速览/index.html`保留分阶段结果表、精选图、配置/模型和完整视频入口。原始训练阶段整体保留，未拆散共享场景、源码快照与各Seed数据。

版本编号代表研究方案，独立Seed对应run，续训不重复计run。v17是最近新增方案，最近一次续训发生在v16。完整索引保留旧记录，共17版本、40个独立run、137条事件、82个模型；本轮补登记17个formal run与其34个最新阶段Best/Final模型。

## 直观图与指标

- 43个导航页面，204张精选PNG副本，均保留原图来源与SHA。
- 实际23个阶段×Seed记录，对应17个独立run。
- v16密集演化使用已核验的密集版本；多个Seed保留并排图。
- 各阶段的Best与Final根据实际模型来源区分；不依靠`final_test`目录名猜测模型身份。
- v14的5个Final没有完整500场景测试，明确标为“未评估”；0°视频不当作整体捕获率。
- 310k Best在原±30°500场景的46.6%与选定[−29°,−5°]新500场景的100%分开呈现。
- 全局主页经1440×1100浏览器截图检查，使用两列卡片，Best/Final测试率无需横向滚动；完整表留在版本/单run页面。

[导航来源manifest](../outputs/formal_ppo_v1/00_导航来源manifest.json) · [来源与指标CSV](../outputs/formal_ppo_v1/00_图表来源与指标.csv)

## 原始数据保护与路径兼容

迁移前记录155,554文件SHA，共26,054,677,424字节。23项目录均完成迁移，原一级目录无残留；迁移日记与全文件清单保存在工程验收中。

迁移后155,455个原始文件保持原字节；其余99个Markdown仅修复链接目标，改前原件另存并逐一核对原SHA。Checkpoint、NPZ、CSV、JSON、日志、源码快照没有重写。没有删除历史后处理、失败尝试或验收fixture。

17个活跃YAML仅变更存储路径，逐项解析对比确认物理、奖励、网络、PPO及评估参数不变。Checkpoint/JSON/CSV/日志中的旧路径保留历史原文，`utils/artifact_paths.py`按严格前缀映射解析。恢复配置只规范化输出/对照存储路径，其余设置继续严格比较；旧manifest的预期SHA通过同字节迁移文件、改前原件或源码快照核对，未跳过完整性检查。

原生Python目录rename曾被Windows拒绝一个目录；使用已核对工作区绝对目标的PowerShell `Move-Item`完成。文件占用检查与路径长度检查未显示中文命名问题。最终23项迁移日记及全文件SHA均通过。

## 验收结果与中文路径

| 检查 | 结果 |
| --- | --- |
| 全量pytest | 1213 passed，168.24秒 |
| 最后路径/导航/索引专项 | 52 passed，4.83秒；随后索引模型登记专项另见索引验收记录 |
| 原始文件最终核验 | 155554项通过；99份旧报告原件SHA额外通过 |
| 本地导航与根文档链接 | 1678项通过；全局索引链接另行核验 |
| 复制图像 | 204张源/副本SHA一致 |
| 活跃配置数值 | 17份YAML逐项一致，仅存储路径变化 |
| 中文路径Checkpoint | 46个实际Best/Final文件均恢复Actor、Critic、Adam及Scheduler，数值有限 |
| 中文路径视频输入 | FFmpeg读取两段现有Best/Final视频首帧通过；原完整解码QA保留 |
| Registry | 路径、配置指纹、模型SHA检查通过 |

默认Windows沙箱临时路径过长时出现过WinError206；使用仓库内短`--basetemp`后测试通过，未删减或弱化断言。中文命名已在当前Python/PyTorch/YAML/FFmpeg与本地HTML链路实际验证；命令仍应按惯例引用完整路径。

主要验收证据位于[目录整理记录](../outputs/formal_ppo_v1/_工程验收/目录整理_20261120/)：`original_file_inventory.jsonl`、`move_journal.jsonl`、`all_original_bytes_verified.json`、`link_changes.json`、`active_path_changes.json`、`chinese_path_runtime_check.json`、`final_navigation_audit.json`及pytest日志。

## 修改范围与后续使用

- 新增分类catalog、迁移映射、目录整理工具、导航生成器与对应测试。
- 增加历史路径解析工具，并在正式恢复、离线重放和轨迹总览读取点使用；不改变数值算法。
- 更新活跃配置/入口的存储路径、根README、版本README和全局registry/索引。
- 新导航与精选PNG放在输出目录；未复制模型或全量轨迹。

重新汇集现有图表可运行：

```powershell
.\.venv\Scripts\python.exe -B -m scripts.build_formal_output_dashboard --write
.\.venv\Scripts\python.exe -B -m scripts.experiment_index --check
.\.venv\Scripts\python.exe -B -m scripts.experiment_index --write-index
```

新增研究方案或独立训练前先登记catalog/registry与新输出位置。迁移工具的`inventory`、`move`阶段不是日常刷新命令；已有迁移记录应保留。

本轮未执行Git add、commit、merge或push。
