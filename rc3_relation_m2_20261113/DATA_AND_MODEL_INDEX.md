# 实际数据与模型索引

- [原始教师回放记录](../rc3_relation_drl_20261113/science/relation_navcover_m2_v1_20261113/init/requests)：1910文件，包含动作序列、完整教师路线成本J、来源hash及保存的独立验证结果。这些是teacher-forcing记录，不是新策略raw/H规划评价。
- [逐次更新](../rc3_relation_drl_20261113/science/relation_navcover_m2_v1_20261113/init/updates)：238文件，含损失、梯度范数、逐参数梯度、更新前后模型hash和实际任务详情。
- [完整事件账](../rc3_relation_drl_20261113/science/relation_navcover_m2_v1_20261113/init/EVENTS.jsonl)与[原资源账](../rc3_relation_drl_20261113/science/relation_navcover_m2_v1_20261113/init/RESOURCE_ACCOUNTING.json)：原字节保留，包括失败重叠事件。
- [SL238部分checkpoint](../rc3_relation_drl_20261113/science/relation_navcover_m2_v1_20261113/init/current_slot0.pt)、[其准确sidecar](../rc3_relation_drl_20261113/science/relation_navcover_m2_v1_20261113/init/current_slot0.json)和[部分状态审计](M2A_PARTIAL_AUDIT.json)：含模型、Adam、RNG、游标、请求预算。不是SL512，未准入M2b。
- [冻结随机起点](../rc3_relation_drl_20261113/M2_RANDOM_INIT.json)、[固定日程](../rc3_relation_drl_20261113/M2_DATA_SCHEDULE.json)、[源码/环境/数据绑定](../rc3_relation_drl_20261113/SCIENTIFIC_SOURCE_BINDING.json)：准确来源；没有载入工程权重。
- [逐请求教师CSV](analysis/TEACHER_REPLAYS.csv)、[训练CSV](analysis/TRAINING_UPDATES.csv)、[完整请求时间](analysis/COMPLETE_REQUEST_TIMES.csv)、[资源曲线数据](analysis/RESOURCE_CURVE.csv)：全部从上述原记录派生，无新的回放、验证、模型或H调用。
- [评价覆盖缺口](analysis/METHOD_TABLE.csv)：六方法×LOCAL66/FULL12×0/512/2048的预期表格，全部未运行。成本空值不等于0。
- [图表PDF、PNG与逐图脚本](figures/README.md)：仅F1/F4有实际观测，F2/F3未生成。

任务输入继续引用已批准的352项训练元数据和原始来源hash，不重复生成布局、导航或教师，不复制/读取封存测试语义。原始来源文件已在同一Git仓库，运行记录保留其精确路径和SHA。
