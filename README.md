# RC3：大规模父场景训练与局部任务完整规划

硕士论文研究内容三：在大规模复杂多任务父场景中以**深度强化学习**训练规划策略，冻结后快速完成不同规模局部子场景的完整任务与轨迹规划，满足导航、覆盖、逐臂操作可达和安全约束。监督、教师搜索和规则用于辅助与对照。

**当前为二维静态代理可达性实验，尚未完成真实双臂IK、连续关节轨迹认证或实机验证。** 默认可靠规划基线仍为 **Fixed3／Detour**；训练执行完成不代表DRL性能目标已经达到。

本首页整理于2026-11-10，科研来源快照为`0c268426f`。本轮位于`codex/repository-handoff-20261110`，**仅本地提交待人工检查，不推送、不合并main**。main仍是较早稳定收尾版本；[分支与远端审计](docs/REPOSITORY_AUDIT.md)列出真实差异。

## 技术流程

完整父场景 → 目标任务/局部窗口 → 候选停靠位姿与逐臂服务表 → 绕障导航、覆盖与访问顺序规划 → 底盘完整路径及任务用臂指派 → 独立完整验证。

典型目标是在10×10父场景训练，同一冻结策略处理4×3、6×6、8×8等局部任务；保留完整父地图和合法候选，不把窗口裁剪当作新环境泛化。目标函数使用`J=L+0.75K`；这是离散代理模型的长度加停靠罚项。

## 已实现与仍需研究

| 已有实现/证据 | 当前判断与缺口 |
|---|---|
| Fixed3、Detour、代理覆盖与路径、方案导出及独立验证 | 当前最可靠的运行基线；不认证真实机器人 |
| 全局—局部MDP、合法mask、端点、冻结H、监督及DRL训练链 | S1/S2真实执行已完成，不能凭loss或梯度宣布规划改善 |
| S3冻结评价及GL-v4/v5机制筛选 | 完整评价格未建立所要求的DRL质量/速度优势，负结果保留 |
| 后续Integrated RPPO模块及部分不可变运行收据 | 运行尚未闭合于本快照；没有最终成功/晋级结论 |
| 机器人输入预检、外部F/A契约、教师交接接口说明 | 真实标定、双臂IK、关节/实机验证仍未完成 |

详见[真实进度](docs/RESEARCH_PROGRESS.md)与[实验史](docs/EXPERIMENT_HISTORY.md)。原60保留测试继续封存；本轮没有新训练或大规模性能实验。

## 最小运行

在本整理分支的仓库根目录使用PowerShell。以下命令已在全新Python 3.13.5虚拟环境实际通过；不需要Torch、GPU或模型权重。

```powershell
python -X utf8 -m venv .venv-handoff
.\.venv-handoff\Scripts\python.exe -X utf8 -m pip install -r rc3_v5/requirements-runtime.txt
.\.venv-handoff\Scripts\python.exe -B -X utf8 docs/tools/reproduce_baselines.py --output outputs/teacher_handoff
.\.venv-handoff\Scripts\python.exe -m pip check
```

第三条命令只生成一个新的4×3 regular工程场景，分别运行Fixed3与Detour16，并重新调用现有独立验证器。输出目录必须不存在，重复复现请另选新目录。本轮两方案均合法；该例没有Detour降本收益。完整命令、环境、耗时及保存方案见[验收报告](docs/HANDOFF_VERIFICATION.md)。

获取版本与Git LFS定向恢复见[QUICK_START](docs/QUICK_START.md)。仅运行上述规则示例不需下载全部历史模型或测试数据。

## 文档入口

| 文档 | 内容 |
|---|---|
| [研究总览](docs/RESEARCH_OVERVIEW.md) / [真实进度](docs/RESEARCH_PROGRESS.md) | 固定目标、已完成/未通过/未运行 |
| [算法设计](docs/ALGORITHM_DESIGN.md) / [代码结构](docs/CODE_ARCHITECTURE.md) | 真实调用链、约束、schema与阅读顺序 |
| [实验史](docs/EXPERIMENT_HISTORY.md) / [原首页全文](README_HISTORY.md) | 数值来源、固定版本、原科研证据完整保留 |
| [快速开始](docs/QUICK_START.md) / [验收](docs/HANDOFF_VERIFICATION.md) | 环境、LFS、两份完整代理方案及限制 |
| [交接包整合](docs/HANDOFF_PACKAGE.md) / [参考资料](docs/REFERENCES.md) | 新接口示例与历史代码的来源区别 |
| [老师研究问题](docs/TEACHER_QUESTIONS.md) / [协作机制](CONTRIBUTING.md) | 优先阅读文件、三项可开始任务、Issue/PR流程 |
| [仓库审计](docs/REPOSITORY_AUDIT.md) / [本轮变更清单](docs/CHANGESET.md) | 分支、未同步成果及整理范围 |

## 希望老师优先帮助

1. 审查状态表示与信用分配，解释为何RL更新未稳定改善监督起点的完整路线。
2. 审查RPPO的冻结基座、残差、Critic、更新约束和证据链，形成有匹配对照的下一步方案。
3. 明确真实数据、MATLAB映射与机器人标定/IK接入缺口，制定可执行验收清单。

建议`main`承载经审查的稳定版本，`teacher/research-review`记录老师建议，`codex/*`承载独立实现；Issue记录研究问题，PR记录修改与验证。现有分支和冻结实验全部保留，合并main由用户决定。
