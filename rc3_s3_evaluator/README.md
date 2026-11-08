# 冻结策略 S3 评估器

**技术 GO；正式 S3 未执行、目标继续封存。**

| 交付 | 文件 |
|---|---|
| 实际实现与验收 | [S3_EVALUATOR_REPORT.md](S3_EVALUATOR_REPORT.md) |
| 机器可读验收 | [S3_ENGINEERING_VERIFICATION.json](S3_ENGINEERING_VERIFICATION.json) |
| 七方法定义 | [S3_METHOD_PIPELINES.json](S3_METHOD_PIPELINES.json) |
| 最终评估源码绑定 | [S3_SOURCE_BINDING.json](S3_SOURCE_BINDING.json) |
| 一次性正式入口与预算 | [S3_EXECUTION_PLAN.json](S3_EXECUTION_PLAN.json) |
| 工程资源账 | [RESOURCE_ACCOUNTING.json](RESOURCE_ACCOUNTING.json) |
| 独立审查 | [INDEPENDENT_REVIEW.md](INDEPENDENT_REVIEW.md) |
| 下一阶段决定 | [NEXT_STEP_DECISION.md](NEXT_STEP_DECISION.md) |

126 个完整工程结果保存在 [工程运行](runs/S3_ENGINEERING_20261107_01)，逐请求 J/L/K、H 增益和时间见 [CSV](ENGINEERING_REQUEST_SUMMARY.csv)。这六例旧开发输入只用于软件验收，不能作为 S3 性能证据。

正式接口：

```powershell
.\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_s3_evaluator.runner formal --authorization <真实S3阶段授权JSON> --source-approval <真实评估源码批准JSON>
```

当前没有可执行的 S3 授权，命令在授权验证前不生成目标。正式输出目录由实验 ID 固定，拒绝用新目录、换授权、失败重试或新种子重置预算。完成阶段不能重复运行。

原工程代码版本在提交 `6be82edc`，几何来源修复后的最终版本为代码绑定 V2；旧登记及 126 结果未重写。`runner engineering` 使用已完成的原登记，不能作为新推理或重复验收的入口；修复后的源与其原绑定不匹配时会拒绝重放，这是冻结证据保护。
