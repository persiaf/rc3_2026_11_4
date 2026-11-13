# R08-A：联合决策模型与下一阶段方案

**已完成保存证据分析与最小接口；没有新训练或规划。B—E未授权、未执行。**

先读[导师短材料](TEACHER_DECISION_BRIEF.md)，再看[路线实际分析](ROUTE_GAP_ANALYSIS.md)、[唯一主设计](HIERARCHICAL_DRL_DESIGN.md)和[分阶段主计划](R08_MASTER_PLAN.md)。

|交付|用途|
|---|---|
|[CURRENT_SYSTEM_AUDIT](CURRENT_SYSTEM_AUDIT.md)|真实代码、调用链、状态与控制权|
|[HISTORICAL_TRAINING_SUMMARY](HISTORICAL_TRAINING_SUMMARY.md)|训练/负结果/封存及准确SL来源|
|[ROUTE_GAP_ANALYSIS](ROUTE_GAP_ANALYSIS.md) / [CSV](ROUTE_GAP_RESULTS.csv)|352训练配对、D5/D6主终点、R07的3550结构比较|
|[JOINT_OPTIMIZATION_MODEL](JOINT_OPTIMIZATION_MODEL.md)|覆盖、同臂分配、配置计数、端点排序和完整约束|
|[HIERARCHICAL_DRL_DESIGN](HIERARCHICAL_DRL_DESIGN.md)|一个完整方案配置联合编辑主候选|
|[CODE_REFACTOR_PLAN](CODE_REFACTOR_PLAN.md)|已实现与后续待实现的文件/接口/验收|
|[R08_MASTER_PLAN](R08_MASTER_PLAN.md) / [预算算术](stage_budget.json)|B—E拟议条件、预算、恢复和失败出口|
|[TEACHER_DECISION_BRIEF](TEACHER_DECISION_BRIEF.md) / [NEXT_STEP](NEXT_STEP_DECISION.md)|导师决策与下一项真实任务|
|[EXPERIMENT_REPORT](EXPERIMENT_REPORT.md) / [RESULT_TO_CLAIM](RESULT_TO_CLAIM.md)|本轮完成范围与科学主张边界|

辅助证据：[父布局表](PARENT_STRUCTURAL_RESULTS.csv)、[分规模汇总](ROUTE_GAP_SUMMARY.csv)、[来源](ANALYSIS_INPUTS.json)、[完整结构归档](ROUTE_STRUCTURE_EVIDENCE.json.gz)、[合成单测](INTERFACE_TEST_RESULTS.json)、[资源](RESOURCE_ACCOUNTING.json)。归档含每配置服务任务/臂、独占覆盖、集合差、共有配置顺序及原始来源引用，全部来自保存记录。

实际可运行命令（现有Python环境，**无新科学请求**）：

```powershell
& .\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_r08_joint_decision_design_20261113.analyze_saved
& .\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_r08_joint_decision_design_20261113.extend_r07
& .\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_r08_joint_decision_design_20261113.write_reports
& .\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_r08_joint_decision_design_20261113.test_interfaces
```

重分析会重建本目录派生CSV/JSON，不写历史证据。发布时再无损打包结构JSON（package_evidence，固定归档已存在时拒绝覆盖）。正式train/resume CLI仍是后续开发目标，不假装当前可以训练。初始和元数据修复源码SHA分别登记；SOURCE_BINDING及manifest绑定最终版本。

继承原60/旧S3/固定32/D5/D6等边界。只读历史S3汇总不等于解封其task/route。固定研究目标仍是DRL训练—冻结局部完整规划，二维代理不作IK/实机认证。
