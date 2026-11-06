# RC3：可达约束下的二维静态任务路径规划

本仓库归档截至 **2026-11-06** 的源码、冻结模型、实验结果、审查材料和论文展示文件。版本标识为 **2026.11.06-global-local-trainer**。默认规划方法是 **Fixed3／Detour**；旧 MLP 保留可选入口，Structured 与 G2～G2.3 保留研究及诊断记录。

当前验证范围是二维静态批次任务与代理可达约束。真实机器人标定与 IK 暂缓。G2.3 的“局部窗口选择＋STOP”作为第三部分 DRL 核心的路线已结束追加训练，不能将这些记录解释为学习方法已经优于强规则。独立构造式研究已完成C0／C1和一次512更新C2小试：速度部分达标，质量及继续门槛未通过；随后完成冻结路线 C2.1：72 次共同后处理使 RL 与 Fixed3 的平均逐场景相对差缩至 +0.8170%，相同后处理下相对先验的学习增量区间仍跨零；没有追加训练或启动C3。归因见 [C2.1报告](rc3_constructive/REPORT_C21.md)，原小试详见 [研究目录](rc3_constructive/README.md) 与 [完整报告](rc3_constructive/REPORT_C012.md)。

C2.2 已完成一次匹配暖启动的raw/post目标对照：两组各128更新、共8192策略采样；新48机制场景上B相对A改善1.5491%，但相对起点平均改善−0.9084%、区间跨零，按事前规则判为无定论并停止。没有进入C3或查询60保留测试。见 [C2.2报告](rc3_constructive/REPORT_C22.md)。

当前同结构继续策略梯度训练支线已收尾，原第128次权重保留为研究参照，A/B仅作证据归档，不追加训练。严格候选间导航信息必要性方向仅记录立项门槛，原G1—G4尚未执行；后续独立R0/R1价值研究不构成该严格门槛的通过。见 [收尾报告](rc3_constructive/REPORT_CLOSEOUT.md) 与 [未来信息价值门槛](rc3_constructive/NAVIGATION_INFORMATION_GATE.md)。

覆盖—导航深度价值规划方案已完成评审，随后单独登记并完成R0：32例有界教师相对g_B＋H平均改善2.2507%，32例均正收益、最大单例贡献8.9448%，全部有效，实际2144/2304次顶层评价。R0仅通过教师机会门槛，未证明学习收益或改变既有默认方法；旧支线保持收尾；R1随后按独立任务完成，R2/R3未启动。见 [修订Word](RC3_研究模型与实验计划_20261105_评审修订版.docx) 与 [评审记录](research_proposals/20261105_nav_value_review/REVIEW.md)。用户原稿保持原字节。R0结果见 [完整报告](rc3_nav_value/REPORT_R0.md) 与 [逐场景表](rc3_nav_value/r0/per_scene.csv)，研究资源和独立审查在 [独立研究目录](rc3_nav_value/README.md)。

R1 已完成单种子有界筛选并收尾当前配置：实际9,286/16,384次顶层评价、6,144步优化；首轮两表示参考均未晋升，终点A=C、B=D。D对B改善0%，对暖起点改善区间跨零，质量与完整时间均未建立强规则部署优势。没有形成有效参考更新的API证据，也没有反证API一般无效。R2/R3未启动，60保留测试未读，默认Fixed3／Detour不变。见 [R1完整报告](rc3_nav_value/REPORT_R1.md) 与 [质量—时间表](rc3_nav_value/r1/quality_time.csv)。

R1研究证据冻结于提交 `df8cf28`。后续用户复核与候选研究边界登记在 [R1收尾补充](rc3_nav_value/r1_closeout/REVIEW_AND_DECISION.md)：优先验证教师动作识别与完整菜单预测可靠性，候选方向尚未启动。本次收尾维护不新增训练、规划或性能实验。

研究内容三的目标固定为：在大规模复杂10×10父场景中进行深度强化学习训练，冻结策略后快速规划不同规模局部任务。监督和规则仅作辅助与对照，不能替代DRL核心。新独立支线已完成 [S0全局—局部环境一致性](rc3_global_local/REPORT_S0.md)：3父图18窗口、63条未训练轨迹及精确重放，1次无更新梯度探针；S1/S2/S3正式训练和性能验证未启动。完整目标与后续协议见 [固定研究目标](rc3_global_local/RESEARCH_OBJECTIVE.md) 与 [研究目录](rc3_global_local/README.md)。S0提交`057830d`已随`codex/global-local-preflight`研究分支历史上传GitHub；旧研究结论不因软件检查通过而改变。

S1/S2正式执行前评审已完成：**正式训练REVISE，本轮工程准备PASS**。实际111物理布局满足预定独立划分；36端点基线与18全候选闭环参照通过；三个已存S0父图的9完整策略轨迹和3次反向通过，优化器更新为0。新增不可变输入/缓存与端点适配层，原S0/v1/v2原字节未改。v3仅为新提案，未激活；完整正式runner、batch/RNG/恢复与权重谱系验收仍须补齐。S3目标及原60测试未评价。见 [本轮全部交付](rc3_global_local_impl_review/README.md)、[实施判断](rc3_global_local_impl_review/IMPLEMENTATION_REVIEW.md) 与 [执行方案](rc3_global_local_impl_review/NEXT_STEP_DECISION.md)。用户明确要求GitHub交付后，已推送至`codex/global-local-preflight`，未合并main；[仓库内交付与远端核对记录](project_management/github_delivery_global_local_preflight/README.md)取代此前仅本地交付状态。

S1/S2正式训练器已实现，登记工程验收与独立复核通过：batch4梯度等价、命名随机流、S1/S2连续与恢复状态逐位相同、独立权重分叉及无授权拒绝。累计21次工程Adam、12次真实完整H，正式科学训练更新0。教师失败后隐式重试预算漏洞已修并零更新回归；两次绑定和旧源字节保留。**工程GO，正式执行仍REVISE**：v2保持活动冻结设计，v3及执行澄清待正式批准，不自动激活，不消费S3/原60。见 [训练器六项交付](rc3_global_local_trainer/README.md)、[工程验收](rc3_global_local_trainer/TRAINER_VERIFICATION.md) 与 [下一步决定](rc3_global_local_trainer/NEXT_STEP_DECISION.md)。成果在`codex/global-local-preflight`交付，不合并main。

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

后续项目变更持续版本化，按对应任务授权推送并核对远端，默认不再生成额外整包备份或重复解压目录。维护约定保存在 [AGENTS.md](AGENTS.md)。本机快照不进入 Git；原始交付／审查 ZIP 继续作为研究证据保留。

清理仅针对可重建计算缓存、迁移测试解压副本和解释器字节码缓存。源码、已保存路径、检查点、训练日志、协议、审查材料、原始 ZIP、机器人资产及本机 Python 环境保留。训练缓存重建方法见 [清理与恢复说明](project_management/ARCHIVE_AND_RESTORE.md)。

前轮归档与清理没有新增训练；本轮构造式C2训练和性能记录单独保存在 `rc3_constructive/`，没有覆盖旧实验。文件通过 SHA256 清单核验。后续研发使用独立输出目录，分支及同步按AGENTS约定执行；Git不存储账号凭据、本机虚拟环境或额外整包备份。
