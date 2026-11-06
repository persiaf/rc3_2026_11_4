# v3 执行协议定稿与分阶段科学授权准备

固定研究目标：大规模复杂父场景DRL训练，冻结策略快速完成不同规模局部任务的覆盖、停靠位姿选择与完整路径规划。分支：`codex/global-local-preflight`，基础`aa0247c8c38243504028bfad24795e0db7120264`。

当前候选：[V3_EXECUTION_CANDIDATE.json](V3_EXECUTION_CANDIDATE.json)，**PENDING_USER_APPROVAL_NOT_ACTIVE_NOT_AUTHORIZED**。原活动设计仍为v2，旧指针、v1/v2/v3提案及研究证据不修改。

协议定稿成果：

- [完整差异JSON](PROTOCOL_LEAF_DIFF.json)、[差异CSV](PROTOCOL_LEAF_DIFF.csv)，按数组元素展开，并区分历史变更。
- [原字节registry](REGISTRY.json)、[256教师清单](S1_TEACHERS_256.csv)、[108例S3声明](S3_DECLARATIONS_108.csv)，仅声明元数据，无目标任务生成。
- [H精确绑定](H_BINDING.json)与[定稿核验](PROTOCOL_FINALIZATION_RECEIPT.json)。

实施分阶段提交：协议定稿；授权适配代码；有限工程回归；最终状态与GitHub发布记录。本目录后续执行入口要求独立用户协议批准和逐阶段授权；当前没有真实教师生成或科学训练权限。

S1-A为共享256条教师生成与冻结，最多256native+256H、1800秒、0optimizer；S1-B为三固定种子各256次监督初始化、每seed1800秒、0新教师查询。全部S1-B完成后才可按新的S2授权继续，原60和S3目标继续封存。
