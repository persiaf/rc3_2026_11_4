# 研究内容三M2：TECHNICAL_INCOMPLETE

M2a完成238/512次科学SL更新，在资源账原子替换时发生Windows拒绝访问异常。1910请求已扣，最后同一请求先complete后failed；原账保留。没有SL512完成收据，M2b及全部评价端点未运行。本轮已停止。

- [实际报告与错误证据](EXPERIMENT_REPORT.md) · [原始数据和部分模型](DATA_AND_MODEL_INDEX.md)
- [实际资源账汇总](RESOURCE_ACCOUNTING.json) · [部分状态审计](M2A_PARTIAL_AUDIT.json)
- [图表与CSV](figures/README.md) · [结果到主张审查](RESULT_TO_CLAIM.md)
- [停止决定](NEXT_STEP_DECISION.md) · [Git交付](GIT_DELIVERY.md)

以下为本轮开始时登记的授权与范围。

# 研究内容三 M2正式科学执行

准确协议和源码由用户基于`9c878928e7f619bce25ddd00792757431ca705a1`批准。稳定科学ID：`relation_navcover_m2_v1_20261113`。本目录是新授权、状态和报告入口；原M1及冻结协议保留原字节。

- [用户授权范围](USER_AUTHORIZATION.md)
- [实施审查GO](IMPLEMENTATION_REVIEW.md) · [准入字节核对](ADMISSION_CHECK.json)
- [独立协议批准](authorizations/PROTOCOL_APPROVAL.json) · [M2a阶段授权](authorizations/M2A_AUTHORIZATION.json)
- [当前执行状态](CURRENT_STAGE_STATUS.json)
- [准确冻结协议](../rc3_relation_drl_20261113/M2_PROTOCOL.json)

M2a首先执行SL512。只有完整成功并验证真实COMPLETE后，才按本轮条件授权创建独立M2b收据并执行固定配对日程。任何原协议停止条件触发即停止；不增加种子、消融、训练池、预算或后续阶段。
