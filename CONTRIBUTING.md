# 科研协作流程

先读[研究目标](docs/RESEARCH_OVERVIEW.md)、[真实进度](docs/RESEARCH_PROGRESS.md)与[AGENTS约定](AGENTS.md)。科研目标与证据冻结边界优先于便利性改动。

## 分支与审查

| 位置 | 建议用途 |
|---|---|
| `main` | 经人工审查后接纳的稳定、可复现研究版本；当前仍是较早收尾版本，不能假定包含最新研究 |
| `teacher/research-review` | 老师的研究建议、静态审查及独立实验提案；本轮仅建议，未创建 |
| `codex/*` | 独立开发、文档整理和事前登记的实验分支 |
| GitHub Issues | 单个研究问题、证据缺口、方法假设、输入与复现问题 |
| Pull Requests | 说明具体修改、来源、实际检查、资源和结论边界；用户确认后才能合并main |

开始工作先核对status、branch、log、remote、fetch和相对main/历史分支的关系；保留用户修改。没有跟踪分支不代表提交未上传，必须检查是否已被其他远端研究分支包含。推送与合并分别服从用户本次授权，本轮整理明确不推送。

## 文档和实验必须分账

文档/维护分支可以增加说明、来源与有界工程复现收据，不改规划器、训练代码、权重、原日志或冻结结果。需要算法实验时另建协议，事前绑定源码、H、环境、种子、数据谱系、预算、失败处置和门槛。IMPLEMENTATION_REVIEW技术GO不自动授予训练/保留测试权限；协议批准与阶段执行授权分别保存。

每轮保留IMPLEMENTATION_REVIEW、EXPERIMENT_REPORT、RESULT_TO_CLAIM、RESOURCE_ACCOUNTING、NEXT_STEP_DECISION与Git记录。本次维护文件位于[docs/repository_handoff](docs/repository_handoff/NEXT_STEP_DECISION.md)，不是旧科研阶段的新状态。

## 数据、结果与失败

不读取原60保留场景，不补旧S3失败槽，不静默换seed、删例、重采样或增加预算。负结果、未运行和不完整结果分别记录。所有历史数值链接到原报告、原账/明细和精确提交；训练loss、H收益和单次计时不冒充DRL性能提升。

保留现有Git LFS规则与冻结字节；按精确清单恢复所需对象，不改写历史、移动冻结标签或强推。不要提交Office锁文件、虚拟环境、活动账本及无关用户文件。用新空输出目录进行复现；不自动生成整包备份。
