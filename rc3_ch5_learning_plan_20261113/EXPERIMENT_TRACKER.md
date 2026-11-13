# 第五章关系DRL跟踪表 v2

2026-11-13；[v1原记录](EXPERIMENT_TRACKER_20261113_v1.md)保留。本版以用户实施修订及新目录实际收据为准。

|条目|目标|状态|证据/下一动作|
|---|---|---|---|
|假设修订|只检验D和动态剩余覆盖表示|完成|[计划v2](EXPERIMENT_PLAN_20261113_v2.md)|
|R001/M0|唯一关系Actor、可变规模、mask/缓存|PASS|[结构](../rc3_relation_drl_20261113/ARCHITECTURE.md)，10人工单测|
|R002/M1|有界真实工程、教师/梯度/恢复/生产批|PASS|[报告](../rc3_relation_drl_20261113/EXPERIMENT_REPORT.md)，126请求/14更新/0失败|
|R003/M2a|独立新结构SL512起点|未执行|[冻结协议](../rc3_relation_drl_20261113/M2_PROTOCOL.json)，待独立协议批准及阶段授权|
|R004/M2b|单seed DRL2048对继续SL2048|未执行|固定0/512/2048端点，需M2a真实完成收据及独立阶段授权|
|开发评价|LOCAL66与FULL12分别报告、六方法|清单冻结，未执行|[评价网格](../rc3_relation_drl_20261113/M2_EVALUATION_GRID.json)|
|额外种子/时间匹配SL/关系消融|主门槛通过后研究|未制定/未执行|不得从工程PASS直接启动|
|MATLAB接口|E/D/J/端点/候选差异|完成文档|[差异表](../rc3_relation_drl_20261113/MATLAB_BOUNDARY.md)，未混用数据或权重|

本轮科学更新0。工程PASS只支持实现和训练接口可用，尚无DRL规划收益结论。
