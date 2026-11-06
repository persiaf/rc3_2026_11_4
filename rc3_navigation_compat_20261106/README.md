# S1-A 兼容修复与S1-B执行衔接

**技术GO：256/256真实生产重放通过，原1482文件字节未变；S1-B尚未执行。**

实验来源：`rc3_v3_global_local_20261106_01`；基准：`0b715b1eedfd3107a1e5188c25c60fe7fde113f1`；分支：`codex/global-local-preflight`。

- [实际代码修复与根因](NAVIGATION_COMPATIBILITY_FIX.md)
- [256例生产重放验收](TEACHER_REPLAY_VERIFICATION.md)、[逐例机器结果](TEACHER_REPLAY_VERIFICATION.json)
- [独立审查](INDEPENDENT_REVIEW.md)、[新执行源码绑定](SOURCE_BINDING_UPDATED.json)
- [阶段来源衔接](STAGE_SOURCE_BRIDGE.md)、[S1-B精确执行计划](S1_B_EXECUTION_PLAN.json)
- [资源账](RESOURCE_ACCOUNTING.json)、[下一步判断](NEXT_STEP_DECISION.md)、[结论边界](RESULT_TO_CLAIM.md)
- [原教师精确兼容清单](LEGACY_TEACHER_REPLAY_MANIFEST.json)、[原始S1-A数据](../rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/)

所有旧数据、旧批准收据、旧协议及旧源码绑定保持原字节。新旧源码在Git历史和独立绑定中分别留存。本轮是零科学调用的工程修复；不自动启动S1-B/S2/S3。
