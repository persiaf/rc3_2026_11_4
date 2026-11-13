# R07动作反事实完整Rollout机会验证

**完成：OPPORTUNITY_INSUFFICIENT。**48个互异训练父布局、144个真实前缀状态、288次主比较+16次固定复测，共304次完整请求；Actor/Critic更新0。

[完整报告](EXPERIMENT_REPORT.md) · [逐状态比较](STATE_PAIRS.csv) · [逐分支raw/H成本](RAW_H_ROUTE_METRICS.csv) · [父布局聚合](PARENT_OPPORTUNITIES.csv) · [尺度/进度](RESULTS.json) · [教师差距](TEACHER_GAP_SUMMARY.csv) · [完整双路线归档索引](rollouts/REQUEST_POINTERS.jsonl) · [实际重复核验](REPRODUCIBILITY.json) · [研究决定](NEXT_STEP_DECISION.md) · [资源](RESOURCE_ACCOUNTING.json)

[协议](PROTOCOL.json)、[任务](TASKS.json)、[源码绑定](SOURCE_BINDING.json)、[首次计算前Git登记](PREREGISTRATION_GIT_RECEIPT.json)已冻结。实际完成命令：

```powershell
& .\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_r07_20261113.run
```

该身份已封闭，命令再次执行会拒绝，不能作为续跑授权。潜在有利候选选择不是已训练策略收益。D5/D6旧负结果、96完整教师和256局部教师及模型原字节保留。
