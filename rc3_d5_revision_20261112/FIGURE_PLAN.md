# 本轮图表范围

依据已批准MASTER_PLAN的P6交付，采用paper-figure技能的数据图流程。当前完整实验根目录为本目录，原历史图表不改。

| 图 | 数据 | 作用 | 状态 |
|---|---|---|---|
| fixed_endpoint_quality | 本轮固定0/1024/2048/4096的66请求、11父等权J | 同时展示PG、SL-U的raw/H开发轨迹，0点共用准确SL基线 | 生成PDF和300dpi PNG |
| parent_paired_HJ | PARENT_PAIRED_RESULTS.csv主4096终点 | 显示对FrozenSL、SL-U的11父配对收益及原门槛；不构造任务级伪独立区间 | 生成PDF和PNG |
| saved_route_examples | 已保存raw/H完整导航折线 | 按H-J差的正负/相等类别取字典序第一个案例；所有66仍在统计中；无持平例即NOT_AVAILABLE | 生成PDF、PNG及选择收据 |
| 方法示意 | METHOD_DIAGRAM.md | 可编辑Mermaid；解释冻结表示、五张量、完整回报、参考与独立验证 | 手工代码示意，非数据图 |
| 三seed、Full-PG归因、部署冷热、新父确认/最终 | 条件阶段未触发 | 无数据，不补零或借历史均值作图 | NOT_RUN |

图仅支持单seed、已暴露开发任务的描述。开发时间表保留实际完整观测，但冷热与系统负载不同，不作为受控部署加速结论。
