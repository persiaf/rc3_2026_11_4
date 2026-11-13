# 第五章导航—覆盖关系DRL：M0/M1交付

**工程结论：PASS。** 固定8案例完成126次完整处理、14次工程更新，0失败；M2科学训练未执行。唯一新增研究假设为显式候选导航D和动态剩余任务覆盖关系是否改善完整构造决策与DRL收益。

|材料|内容|
|---|---|
|[工程报告](EXPERIMENT_REPORT.md)|每题4轨迹成本/重复率、梯度/缓存、最大规模资源、教师/空任务、冻结恢复|
|[资源账](RESOURCE_ACCOUNTING.json)|实际消费、原上限、工程/科学分账；[原始工程账](engineering/RESOURCE_ACCOUNTING.json)|
|[结果与主张](RESULT_TO_CLAIM.md)|工程PASS成立；科学规划改善尚未验证|
|[正式M2协议说明](M2_PROTOCOL.md) · [机器协议](M2_PROTOCOL.json)|独立起点、固定日程、六对照、0/512/2048端点、主门槛、预算及恢复|
|[科学源码绑定](SCIENTIFIC_SOURCE_BINDING.json) · [冻结收据](M2_FREEZE_RECEIPT.json)|准确实现/H/环境/数据/模型身份；FROZEN_NOT_AUTHORIZED|
|[结构说明](ARCHITECTURE.md) · [Actor源码](actor.py)|64维/2块/4头关系注意力，无额外动作或搜索|
|[科学运行器](m2.py) · [评价器](evaluate.py)|已实现，正式日程未执行|
|[数据说明](DATA_ADMISSION_NOTES.md) · [训练日程](M2_DATA_SCHEDULE.json) · [评价清单](M2_EVALUATION_CASES.json)|352既有训练配对；LOCAL66与FULL12分别报告|
|[初始化登记](M2_RANDOM_INIT.json)|独立新结构随机权重；0前向/0更新，非工程模型；SL512尚未产生|
|[工程登记](ENGINEERING_REGISTRATION.json) · [B4G4补充登记](M1_BATCH4_REGISTRATION.json)|执行前用例、源码、容差及共同上限|
|[工程最终收据](engineering/M1_FINAL_RESULT.json)|126/14真实结果及最新可恢复工程checkpoint|
|[实施审查](IMPLEMENTATION_REVIEW.md) · [下一步](NEXT_STEP_DECISION.md)|停止于M1；协议批准和阶段授权分开|
|[MATLAB边界](MATLAB_BOUNDARY.md)|E、D、成本、端点与候选差异；27案例仅为既有机制证据|
|[计划修订说明](PLAN_REVISION.md) · [当前计划](../rc3_ch5_learning_plan_20261113/EXPERIMENT_PLAN.md)|保留v1；修订唯一机制、评价及32小时配对上限|
|[Git交付](GIT_DELIVERY.md)|独立分支、提交与实际远端记录|

工程模型仅用于工程恢复，不得用于科学初始化。M2协议冻结不自动启动训练；当前没有科学阶段授权收据。原60、旧S3、固定32和关闭支线保持封存，用户论文及历史结果保持原状，未合并main。
