# M2a SL238受控恢复执行

**COMPLETE_RECOVERED；关闭后独立只读审计PASS。** 新增2192次教师请求、274次更新，累计4102次请求、512次optimizer尝试，累计墙钟4773.906秒（79.57分钟）。无新增失败，原历史TECHNICAL_INCOMPLETE及6次消费保留。

- [实际报告](EXPERIMENT_REPORT.md) · [模型、恢复收据及原始数据](MODEL_AND_DATA_INDEX.md)
- [关闭后审计](CLOSED_RUN_AUDIT.json) · [累计资源账](RESOURCE_ACCOUNTING.json)
- [M2b来源资格映射](M2B_INITIALIZATION_MAPPING.json) · [M2b准入评审](M2B_ADMISSION_REVIEW.md)
- [停止决定](NEXT_STEP_DECISION.md) · [结果到结论](RESULT_TO_CLAIM.md) · [Git交付](GIT_DELIVERY.md)

M2b模型来源资格PASS；当前准入REVISE，需要明确支持恢复收据的适配版本和新的阶段授权。本轮未运行M2b、DRL、继续SL、新父测试或MATLAB。

以下保留本轮启动时的授权登记。

本目录记录用户基于d7b76b5e4329c1ca1aaa41f51c8d90a4237a4306批准的一次M2a恢复。准确协议及修复源码保持原字节；新批准和阶段权限单独登记。

- [用户授权范围](USER_AUTHORIZATION.md)
- [准入审查](IMPLEMENTATION_REVIEW.md) · [准确核对收据](ADMISSION_CHECK.json)
- [协议批准](authorizations/PROTOCOL_APPROVAL.json) · [M2a恢复阶段授权](authorizations/M2A_RECOVERY_AUTHORIZATION.json)
- [阶段状态](CURRENT_STAGE_STATUS.json)
- [原恢复协议](../rc3_relation_recovery_20261114/RECOVERY_PROTOCOL_DRAFT.json)

仅执行239—512批，累计请求4102、optimizer512、墙钟14400秒。历史1910请求及238更新不清零。M2b未获授权，执行后停止并报告真实结果。
