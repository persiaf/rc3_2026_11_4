# RC3：可达约束下的二维静态任务路径规划

本仓库归档截至 **2026-11-05** 的源码、冻结模型、实验结果、审查材料和论文展示文件。版本标识为 **2026.11.05-nav-value-proposal-reviewed**。默认规划方法是 **Fixed3／Detour**；旧 MLP 保留可选入口，Structured 与 G2～G2.3 保留研究及诊断记录。

当前验证范围是二维静态批次任务与代理可达约束。真实机器人标定与 IK 暂缓。G2.3 的“局部窗口选择＋STOP”作为第三部分 DRL 核心的路线已结束追加训练，不能将这些记录解释为学习方法已经优于强规则。独立构造式研究已完成C0／C1和一次512更新C2小试：速度部分达标，质量及继续门槛未通过；随后完成冻结路线 C2.1：72 次共同后处理使 RL 与 Fixed3 的平均逐场景相对差缩至 +0.8170%，相同后处理下相对先验的学习增量区间仍跨零；没有追加训练或启动C3。归因见 [C2.1报告](rc3_constructive/REPORT_C21.md)，原小试详见 [研究目录](rc3_constructive/README.md) 与 [完整报告](rc3_constructive/REPORT_C012.md)。

C2.2 已完成一次匹配暖启动的raw/post目标对照：两组各128更新、共8192策略采样；新48机制场景上B相对A改善1.5491%，但相对起点平均改善−0.9084%、区间跨零，按事前规则判为无定论并停止。没有进入C3或查询60保留测试。见 [C2.2报告](rc3_constructive/REPORT_C22.md)。

当前同结构继续策略梯度训练支线已收尾，原第128次权重保留为研究参照，A/B仅作证据归档，不追加训练。未来候选间导航信息方向仅记录立项门槛，尚未启动搜索、模型或训练。见 [收尾报告](rc3_constructive/REPORT_CLOSEOUT.md) 与 [未来信息价值门槛](rc3_constructive/NAVIGATION_INFORMATION_GATE.md)。

新增覆盖—导航深度价值规划方案已完成文献核实、理论边界与实验预算修订，仅为待执行提案；旧支线保持收尾，R0～R3均未启动。见 [修订Word](RC3_研究模型与实验计划_20261105_评审修订版.docx) 与 [评审记录](research_proposals/20261105_nav_value_review/REVIEW.md)。用户原稿保持原字节。

## 获取完整文件

模型、训练数据与历史 ZIP 使用 **Git LFS**。先安装 Git LFS，然后克隆并获取大文件：

```powershell
git lfs install
git clone https://github.com/persiaf/rc3_2026_11_4.git
cd rc3_2026_11_4
git lfs pull
```

需要有该私有仓库的访问权限。仅下载 GitHub 自动生成的源码 ZIP 不作为本项目完整恢复方式；完整恢复应检出指定提交并执行 `git lfs pull`，再运行文件清单校验。

## 当前使用入口

在仓库根目录创建 Python 环境，安装运行依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r rc3_v5/requirements-runtime.txt
.\.venv\Scripts\python.exe -B -m rc3_v5.plan --method fixed3 --scale 10x10 --mode blocked --seed 771903 --output outputs/fixed3.json
.\.venv\Scripts\python.exe -B -m rc3_v5.plan --method detour --budget 8 --scale 10x10 --mode blocked --seed 771903 --output outputs/detour.json
```

配置预检入口为 `python -m rc3_v5_1.robot_input`。未填写有效标定的真实空模板返回 `blocked_inputs` 是预期状态。训练环境依赖与各阶段命令保留在对应目录中，不能用运行依赖替代全部历史训练依赖。

## 版本与证据入口

| 目录 | 用途与状态 |
| --- | --- |
| `rc3_baseline/`、`rc3_fast/`、`rc3_improve/`、`rc3_learning/` | 初期无学习、选择器与学习原型的历史记录 |
| `rc3_v3/`、`rc3_v4/` | 可靠规划演进及冻结主实验；v4 继续提供主算法比较证据 |
| `rc3_v5/` | 当前规划入口、共同约束、缓存及输入准备维护证据 |
| `rc3_v5_1/` | 当前机器人配置预检入口 |
| `rc3_rl_research/`、`rc3_rl_g2/`、`rc3_rl_g21/`、`rc3_rl_g22/`、`rc3_rl_g23/` | G0/G1 至 G2.3 的研究协议、训练、对照与诊断记录 |
| `rc3_g23_closeout/` | 已执行的收尾决定、只读复算和论文可用表述 |
| `rc3_constructive/` | 独立构造环境、插入基线、一种子REINFORCE小试、重放验证、冻结路线共同后处理归因及学习优势尚未确立的证据 |
| `paper/`、`paper_figures_v1/` | 论证组织、固定案例和展示材料 |
| `*_import/`、根目录历史 ZIP／审查文档 | 依赖来源、原始交付与审查证据，原样保留 |
| `project_management/`、`tools/` | 本轮备份、版本清单、清理收据与恢复工具 |

G2.3 测试包含 6 个独立场景、180 份输出；多步模型共 36 次测试，“严格降本且完整时间不超过 0.8 秒”为 0/36。每个最终多步模型为 512 次优化更新，3,840 是全部种子和对照的合计。有限次要学习增量与主目标失败同时保留，详见 [收尾报告](rc3_g23_closeout/REPORT_CLOSEOUT.md)。

## 备份与清理

按用户后续要求，本机只保留 `project_backups/2026-11-04_g23-closeout/full_before_cleanup.zip` 这一份完整清理前快照及必要说明／收据。它采用 ZIP 内 SHA256 内容去重存储，包含当时的原始文件和本机 Python 环境。与 GitHub 已保存内容重复的标准项目 ZIP 及无用操作日志已删除，具体记录见 `project_management/backup_retention_20261104.json`。

后续项目变更持续提交并同步到本仓库，默认不再生成额外整包备份或重复解压目录。维护约定保存在 [AGENTS.md](AGENTS.md)。本机快照不进入 Git；原始交付／审查 ZIP 继续作为研究证据保留。

清理仅针对可重建计算缓存、迁移测试解压副本和解释器字节码缓存。源码、已保存路径、检查点、训练日志、协议、审查材料、原始 ZIP、机器人资产及本机 Python 环境保留。训练缓存重建方法见 [清理与恢复说明](project_management/ARCHIVE_AND_RESTORE.md)。

前轮归档与清理没有新增训练；本轮构造式C2训练和性能记录单独保存在 `rc3_constructive/`，没有覆盖旧实验。文件通过 SHA256 清单核验。后续研发使用独立输出目录，分支及同步按AGENTS约定执行；Git不存储账号凭据、本机虚拟环境或额外整包备份。
