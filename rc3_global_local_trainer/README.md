# S1/S2 正式训练器与执行准入

固定目标：大规模复杂父场景DRL训练，冻结策略后快速完成不同规模局部任务的完整规划。基准提交：`af8048149f6263e358f635385591950ab9a8f7c3`；分支：`codex/global-local-preflight`。

**训练器已实现，登记的工程验收PASS；正式S1/S2仍为REVISE，等待正式协议批准及执行授权。** v2仍为活动冻结设计，v3未激活。工程验收实际21次Adam调用、12次真实H，均不计为正式S2训练。

本轮六项交付：

- [训练器实现报告](TRAINER_IMPLEMENTATION_REPORT.md)
- [训练器验收](TRAINER_VERIFICATION.md)
- [协议决定](PROTOCOL_DECISION.md)与[逐项比较](PROTOCOL_COMPARISON.json)
- [资源账](RESOURCE_ACCOUNTING.json)
- [Git与远端核对](GIT_VERSION_AUDIT.json)
- [下一步判断](NEXT_STEP_DECISION.md)

代码入口：[核心](engine.py)、[随机流](rng.py)、[Checkpoint](checkpoint.py)、[数据与混合采样](data.py)、[阶段编排](runner.py)、[正式准入](approval.py)。[工程注册](ENGINEERING_VERIFICATION_PROTOCOL.json)、[容差](VERIFICATION_TOLERANCES.json)、两份执行绑定、原始journal及[验收汇总](VERIFICATION_RESULT.json)保留；[独立审查](INDEPENDENT_REVIEW.md)只读保存证据，不重复训练。

下面命令只查看接口，不产生训练：

```powershell
.\rc3_learning\.venv\Scripts\python.exe -B -m rc3_global_local_trainer.runner --help
```

正式接口提供`teachers/S1/S2/all`阶段、显式seed、已有run和更新边界resume。默认v2因执行定义不完整而拒绝。显式v3也必须提供独立的用户协议批准、执行授权及当前源码绑定的工程PASS；没有`--yes`旁路。本仓库没有已批准/已授权的正式执行收据。不要为查看交付重复运行engineering verification或将工程权重作为正式模型。

所有成果与LFS工程检查点交付在本GitHub研究分支，不合并main；旧研究、v1/v2、v3提案与S0保持原字节。原60与S3目标任务继续封存。
