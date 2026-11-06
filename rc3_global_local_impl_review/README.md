# S1/S2 正式执行前评审与工程准备

基准：S0 `057830d584b6c71c79ef5118087c681c98135fe2`。本轮分支：`codex/global-local-preflight`。日期：2026-11-06。

**正式 S1/S2 判断：REVISE；本轮有界工程准备 PASS。** 固定目标仍是大规模复杂父场景 DRL 训练，冻结策略后完成不同规模局部任务的完整快速规划。本轮没有优化器更新，没有 S3 目标任务查询，没有读取原60测试的语义数据，没有合并main。后续按用户明确要求已交付到GitHub研究分支，见[仓库内发布收据](../project_management/github_delivery_global_local_preflight/README.md)。

交付入口：

- [实施评审](IMPLEMENTATION_REVIEW.md)：问题分级与正式训练前缺口。
- [实际执行报告](EXPERIMENT_REPORT.md)：纯几何独立性、开放端点及无更新探针。
- [证据与结论](RESULT_TO_CLAIM.md)：支持范围和未证明事项。
- [资源账](RESOURCE_ACCOUNTING.json)：真实调用、合成单测和只读验证分别计账。
- [下一步决定](NEXT_STEP_DECISION.md)：形成正式执行器后再申请对应阶段执行。
- [版本与远端记录](GIT_VERSION_AUDIT.json)：原评审阶段本地状态的历史记录；当前GitHub上传、远端SHA及完整性核对见[发布收据](../project_management/github_delivery_global_local_preflight/DELIVERY_RECEIPT.json)。原外部核验结果已另存[仓库内历史核验收据](../project_management/github_delivery_global_local_preflight/REVIEW_INTEGRITY_RECEIPT.json)，交付不再依赖本地路径。

原 S0/v1/v2/FUTURE_DESIGN_STATUS 保持原字节。新增 [v3提案](PROPOSED_PROTOCOL_V3.json)、[种子提案](PROPOSED_SEEDS_V3.json)与[变更说明](PROTOCOL_CHANGELOG.md)均未激活。不得把这些准备文件当作启动训练的授权。

`GEOMETRY_GATE.json`、`ENGINEERING_RESULT.json`、各 journal 与计划保存已执行的有界检查。`check_layouts.py`、`check_engineering.py` 拒绝覆盖已有结果；不要为浏览报告再次执行科学调用。独立审查只读取这些保存结果。原生基线内部搜索和神经轨迹的完整 H 均单列，不能归为神经学习收益。
