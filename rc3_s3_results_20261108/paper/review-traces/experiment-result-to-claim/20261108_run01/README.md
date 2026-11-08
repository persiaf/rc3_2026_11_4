# 独立审查追踪读取说明

正式审查的权威计数来自 `READ_ONLY_AUDIT_V2.json`，与本轮 `RESOURCE_ACCOUNTING.json`、`SAVED_RUN_AUDIT.json` 交叉一致。

`READ_ONLY_AUDIT.json` 是保留的初稿。初稿将请求之间的导航预热计数混入前一请求的最后计数，不能用于资源汇总。V2 已通过明确的 request_start/request_complete 活跃区间修正；正式报告、独立审查和结论均使用 V2。

初稿保留原字节以记录审查修正过程。该修正只涉及只读审计草稿，没有修改正式运行日志、HEAD、pending、资源消费或实际路线。
