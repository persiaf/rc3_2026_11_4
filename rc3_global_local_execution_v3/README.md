# v3 执行协议定稿与分阶段科学授权准备

<!-- S2_CURRENT_BEGIN -->
**当前：S2 `COMPLETE_ALL3_FIXED_ENDPOINTS`，实际3072次更新，三seed权重和资源按真实收据归档。** 源码0f0ffd2、原批准9a45与冻结协议身份保持。S3模型谱系就绪 `GO`；执行就绪 `REVISE_PENDING_FROZEN_PROTOCOL_EVALUATOR_AND_SEPARATE_AUTHORIZATION`，S3无授权/未启动，目标及原60封存。生产入口目前只实现S1-A/S1-B/S2；后续须补齐冻结协议评估器的冷/暖计时、共同H、父场景聚合与108声明一致性，目标语义访问仍需独立阶段授权。本轮开发仅原18DEV任务的 108 次重复评价，不是S3的108目标。见[本轮报告](../rc3_s2_results_20261107/README.md)、[当前准确状态](ACTIVE_EXECUTION_STATUS.json)及[已核验发布收据](../rc3_s2_results_20261107/PUBLICATION_RECEIPT.json)。下文保留历史。
<!-- S2_CURRENT_END -->

<!-- S1B_CURRENT_BEGIN -->
**当前：v3及0f0ffd2兼容源码已经明确批准，S1-A/S1-B实际完成。** S1-B三seed各256更新，S2仍未授权/未启动，S3与原60继续封存。见[当前活动执行记录](ACTIVE_EXECUTION_STATUS.json)、[本轮真实收据](authorizations/20261106_S1_B/)及[三种子训练实际结果](../rc3_s1b_results_20261106/README.md)。下文为候选定稿时的历史状态，候选原文件中的PENDING不覆盖其后独立批准收据。
<!-- S1B_CURRENT_END -->


固定研究目标：大规模复杂父场景DRL训练，冻结策略快速完成不同规模局部任务的覆盖、停靠位姿选择与完整路径规划。分支：`codex/global-local-preflight`，基础`aa0247c8c38243504028bfad24795e0db7120264`。

当前候选：[V3_EXECUTION_CANDIDATE.json](V3_EXECUTION_CANDIDATE.json)，**PENDING_USER_APPROVAL_NOT_ACTIVE_NOT_AUTHORIZED**。原活动设计仍为v2，旧指针、v1/v2/v3提案及研究证据不修改。

协议定稿成果：

- [完整差异JSON](PROTOCOL_LEAF_DIFF.json)、[差异CSV](PROTOCOL_LEAF_DIFF.csv)，按数组元素展开，并区分历史变更。
- [原字节registry](REGISTRY.json)、[256教师清单](S1_TEACHERS_256.csv)、[108例S3声明](S3_DECLARATIONS_108.csv)，仅声明元数据，无目标任务生成。
- [H精确绑定](H_BINDING.json)与[定稿核验](PROTOCOL_FINALIZATION_RECEIPT.json)。

协议、授权适配代码和有限回归已完成。**工程准入GO；正式科学执行REVISE，等待用户批准协议与对应阶段。** 当前没有真实教师生成或科学训练权限，活动v2指针未改变。

当前准确协议SHA256：`0c5bc7d7a9557edf2af76f51a9f418550f0b4ce1ec6a8b47ddea95c6256e78bb`。33份源码、registry、H和环境见[EXECUTION_SOURCE_BINDING.json](EXECUTION_SOURCE_BINDING.json)，其SHA256为`435b1c8be95d6ec4f14f4aa87913a6eb3e937e3684eaadfabbad160b57ffe968`。

- [实施评审](IMPLEMENTATION_REVIEW.md)、[授权与恢复链](AUTHORIZATION_CHAIN.md)、[S1-A/S1-B准备与入口](S1_PREPARATION.md)。
- [工程回归报告](AUTHORIZATION_VERIFICATION.md)、[原始结果](engineering_runs/admission_02/RESULT.json)、[独立审查](INDEPENDENT_REVIEW.md)。
- [资源账](RESOURCE_ACCOUNTING.json)、[实际执行报告](EXPERIMENT_REPORT.md)、[下一步决定](NEXT_STEP_DECISION.md)、[Git版本核对](GIT_VERSION_AUDIT.json)。
- [待批准模板](authorization_templates/)均为PENDING状态，不构成许可；[模板生产入口拒绝检查](PENDING_TEMPLATE_CHECK.json)确认没有产生生产ExecutionPermit。

S1-A为共享256条教师生成与冻结，最多256native+256H、1800秒、0optimizer；S1-B为三固定种子各256次监督初始化、每seed1800秒、0新教师查询。全部S1-B完成后才可按新的S2授权继续，原60和S3目标继续封存。
