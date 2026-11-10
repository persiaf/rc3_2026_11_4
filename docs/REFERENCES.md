# 参考资料与来源锁定

本页是接手阅读索引，不是完整论文参考文献表。只列已获得来源的资料；未取得原始源码或未复现的内容明确列出。

## 外部论文

Mirco Theile, Harald Bayerlein, Richard Nai, David Gesbert, Marco Caccamo. *UAV Path Planning using Global and Local Map Information with Deep Reinforcement Learning*. ICAR 2021. [作者提交的arXiv v4](https://arxiv.org/abs/2010.06917v4)，[会议DOI](https://doi.org/10.1109/ICAR53236.2021.9659413)。2026-11-10核对arXiv页面的作者、题名、ICAR说明与关联DOI。该工作研究DDQN及压缩全局/裁剪局部地图；不构成本项目双臂操作或当前策略结构的验证证据。

作者代码锁定为[theilem/uavSim的75316e8b提交](https://github.com/theilem/uavSim/tree/75316e8bf598820d1d3ef740de2321bed58f1076)，`icar`分支，交接包记录BSD-3-Clause。具体提交与许可来源为随包SOURCE_LOCKS及源码来源记录；本轮未重训该TensorFlow工程，也未重新核验整个上游仓库。参考映射详见[交接说明](HANDOFF_PACKAGE.md)。

## 本项目主来源

| 来源 | 用途与边界 |
|---|---|
| [固定研究目标](../rc3_global_local/RESEARCH_OBJECTIVE.md) | 大规模DRL训练与冻结局部推理的任务定义 |
| [活动v3执行状态](../rc3_global_local_execution_v3/ACTIVE_EXECUTION_STATUS.json) | 原S1/S2/S3批准、来源和阶段处置，不能当作后续RPPO总状态 |
| [实验史及固定提交](EXPERIMENT_HISTORY.md) | 数值→报告→明细/账本→代码版本追溯 |
| [Fixed3/Detour实现](../rc3_v4/runtime.py)及[验证器](../rc3_baseline/validator.py) | 默认规则和本轮代理复现的真实代码 |
| [交接ZIP](../交接/RC3_Teacher_Handoff_v1.zip) | 新接口示例、来源锁定和原包测试报告；不替代正式模型结果 |
| [综合DRL原方案](../RC3_Integrated_DRL_Plan_and_Codex_Task_v1.md) | 历史设计来源；与后续实际执行协议区分 |

MobileManipulationPlanning旧MATLAB工程目前只取得交接包保存的README摘录，未取得可核验的完整`.m`源码、固定提交和toolbox环境；新导出模板不能当作原代码。开题、中期及论文材料仅按交接需求说明提供研究脉络，本轮未重新逐篇审查或据其新增数值主张。

仓库可见性以本轮GitHub API审计为准：当前为public；交接包`private=true`是来源时点的历史描述。本轮没有更改仓库权限。[仓库审计](REPOSITORY_AUDIT.md)
