# 原Actor五张量候选：工程PASS，科学数据准入阻塞

[完整主计划](MASTER_PLAN.md) · [机器协议](PROGRAM_PROTOCOL.json) · [实际报告](EXPERIMENT_REPORT.md) · [独立审查与主张](RESULT_TO_CLAIM.md) · [下一步决定](NEXT_STEP_DECISION.md)

准确SL256；原Actor65088标量中只更新key.weight、dynamic.weight、nav.weight、query.weight、query.bias共25152，其余39936冻结。原输入/评分可微，独立Trainer使用完整任务回报，没有A1残差幅度、B-cond、优势RMS、更新guard或Critic。

P0九项工程检查PASS。含前置失败及一次记账修复，累计15次完整尝试（14完成/1失败）、Actor5、Critic0；无新教师/场景/导航/native/保留集。准确256教师池完整10×10标签为0，无法支持同清单50%完整父场景的PG/SL，因此进入P6技术数据未完成；科学训练/评价均NOT_RUN。

## 实际验证过的命令

仓库根目录PowerShell：

```powershell
& .\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_decoder_program_20261111.run --status
& .\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_decoder_program_20261111.run --continue-authorized-plan decoder-program-20261111-v1
& .\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_decoder_program_20261111.run --resume decoder-program-20261111-v1
```

持续与恢复入口准确返回数据阻塞，新增forward/optimizer/规划均0。已实际执行的工程命令为`-m rc3_decoder_program_20261111.run --stage engineering`；该attempt已完成，重复调用会拒绝重置资源，不应再次运行。R02记账修复已执行1次普通任务，不重跑P0。

实现API：[Actor与冻结导出/加载](actor.py)、[独立Trainer更新/保存/恢复](trainer.py)、[统一原Actor完整请求与教师回放](runtime.py)。P0实际调用这些API，模型索引见[MODEL_INDEX.json](MODEL_INDEX.json)。**当前没有交付可启动全部科学阶段的训练/评价CLI；不能以这些API或阻塞状态入口冒充科学驱动已验收。**

## 结果和身份

- [逐请求J/L/K、完整时间、停靠集合及顺序变化](ENGINEERING_REQUEST_SUMMARY.csv)、[更正计数侧表](CORRECTED_REQUEST_RESULTS.csv)。全部为工程观测，不用于路线收益结论。
- [工程9项证据](P0/CHECKS.json)、[R02记账检查](P0_ACCOUNTING_REPAIR/CHECKS.json)、[275项只读复核](INDEPENDENT_NUMERICAL_REVIEW.json)、[最终资源账](RESOURCE_ACCOUNTING.json)。原13条记录的逐请求搜索细账UNKNOWN，全局新消费仍有真实累计。
- [参数白名单](PARAMETER_ALLOWLIST.json)、[数据NOGO](DATA_ADMISSION.json)、[身份字段说明](IDENTITY_NOTES.md)、[旧文件保持](HISTORY_PRESERVATION_CHECK.json)。
- [工程Actor](P0/models/D5-PG.engineering.actor.pt)、[工程checkpoint](P0/models/D5-PG.engineering.training.pt)只作工程归档，禁止正式初始化；科学模型NOT_RUN。
- [完整规范原字节副本](NORMATIVE_TASKBOOK.txt)、[用户批准收据](APPROVAL_RECEIPT.json)、[P0源码绑定](SOURCE_BINDING.json)、[活动R02绑定](SOURCE_BINDING_R02.json)、[只读审查绑定](SAVED_OUTPUT_AUDIT_BINDING.json)。

独立分支`codex/rc3-decoder-program-20261111`，不合并main。保留旧53行表、固定32封存、旧权重与用户文件。后续需要改变教师边界才请求具体变更批准，不自动重开研究支线。
