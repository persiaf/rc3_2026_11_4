# S3 正式执行：真实部分结果及 IO 中断

**执行 REVISE；科学判定 NOT_EVALUATED_INCOMPLETE。** 已使用批准的 b348… 评估源码实际运行，首种子预扣 163 次、完成 162 次，第 163 次持久化失败后停止，后两种子未启动。

| 交付 | 文件 |
|---|---|
| 完整实际报告 | [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) |
| 实施及风险 | [IMPLEMENTATION_REVIEW.md](IMPLEMENTATION_REVIEW.md) |
| 资源账 | [RESOURCE_ACCOUNTING.json](RESOURCE_ACCOUNTING.json) |
| 已保存结果审计 | [SAVED_RUN_AUDIT.json](SAVED_RUN_AUDIT.json) |
| 原始部分 J/L/K 与计时 | [PARTIAL_REQUESTS.csv](PARTIAL_REQUESTS.csv) |
| 外部失败与资源冻结 | [EXTERNAL_FAILURE_RECEIPT.json](EXTERNAL_FAILURE_RECEIPT.json) |
| 独立审查 | [INDEPENDENT_REVIEW.md](INDEPENDENT_REVIEW.md) |
| 科学结论边界 | [RESULT_TO_CLAIM.md](RESULT_TO_CLAIM.md) |
| 下一步 | [NEXT_STEP_DECISION.md](NEXT_STEP_DECISION.md) |
| 批准与授权 | [源码批准](authorizations/EVALUATION_SOURCE_APPROVAL.json)、[S3授权](authorizations/S3_AUTHORIZATION.json) |
| 只用合成夹具的故障复现 | [IO_FAULT_REPRO_RESULT.json](IO_FAULT_REPRO_RESULT.json) |

[完整原始运行](../rc3_s3_evaluator/runs/rc3_v3_global_local_20261106_01)保留所有输入、计划、源哈希、journal、原 HEAD 和 pending；[索引](ARTIFACT_INDEX.json)保存180个文件哈希。没有再次运行失败请求，没有修改批准评估器或消费原60测试。

准确阶段墙钟因闭合失败不可恢复；132.83秒仅为完成请求时间总和。592秒为外部保守冻结额，含启动前等待与故障后观察，不能写成实测阶段耗时。科学质量与部署优势暂不判断。

GitHub研究分支为 `codex/global-local-preflight`；版本及发布核对记录见 [GIT_VERSION_AUDIT.json](GIT_VERSION_AUDIT.json) 和 [PUBLICATION_RECEIPT.json](PUBLICATION_RECEIPT.json)。
