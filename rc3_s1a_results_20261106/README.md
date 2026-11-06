# S1-A 正式科学执行结果

实验ID：`rc3_v3_global_local_20261106_01`；研究分支：`codex/global-local-preflight`。本轮已收到两项明确用户决定，建立独立真实批准/阶段授权并直接执行，没有新增审批准备轮次。

**实际运行完成：256/256教师，native256+H256=512次，累计190.153795秒，缓存0、失败0、optimizer0。512份返回计划经独立代理约束复核通过。** 复核发现导航元数据的保存/重载身份缺口，因此**S1-B为REVISE，未启动且仍需独立授权**。

- [实际实验报告](EXPERIMENT_REPORT.md)、[实施与验收判断](IMPLEMENTATION_REVIEW.md)、[结论边界](RESULT_TO_CLAIM.md)。
- [独立逐例审计](INDEPENDENT_REVIEW.md)、[机器审计收据](INDEPENDENT_AUDIT.json)、[资源质量复核](RESOURCE_REVIEW.md)。
- [资源账](RESOURCE_ACCOUNTING.json)、[导航重载问题](NAVIGATION_REPLAY_FINDING.md)、[下一步决定](NEXT_STEP_DECISION.md)、[Git记录](GIT_VERSION_AUDIT.json)。
- [原始完整运行目录](../rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/)、[256池索引](../rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/teachers256/POOL_INDEX.json)、[阶段完成与资源](../rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/stage_admission/S1-A/COMPLETE.json)。
- [真实协议批准](../rc3_global_local_execution_v3/authorizations/20261106_S1_A/PROTOCOL_APPROVAL.json)、[仅S1-A执行授权](../rc3_global_local_execution_v3/authorizations/20261106_S1_A/S1_A_AUTHORIZATION.json)、[活动执行状态](../rc3_global_local_execution_v3/ACTIVE_EXECUTION_STATUS.json)。

原候选协议、435b…源码绑定、registry、H及旧v2历史指针均保留原字节；新的活动状态引用用户已批准的准确版本。S1-A原始产物未因审计而改写，没有重跑教师或生成正式模型。固定目标仍为大规模DRL训练—冻结策略局部快速完整规划，S1教师仅作辅助初始化。S1-B/S2/S3及原60仍保持未授权/封存边界。
