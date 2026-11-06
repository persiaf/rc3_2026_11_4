# 导航保存/重载身份缺口

**发现属实，影响S1-B直接复用；不改变已保存路线的成本与可行性。** 本轮不改批准源码，不重跑任何教师，不覆盖输入。

冻结`baseline.navigation()`返回`rejected_edge_count`；`bind_navigation`将完整nav字段加入内容身份。批准`data.py`保存NPZ及receipt时未保存该字段，磁盘重载也未恢复。

因此有两种实际来源schema：首96个新生成导航输入包含该字段；后160个从保存nav读取的输入不含该字段。两者D、coordinates、service_nodes、predecessors、node/edge计数和有效路径均有原始保存证据，但首96个原parent identity不能由简化schema直接重建。批准loader检查原`parent_identity`时会拒绝，不能自动给S1-B GO。

本次独立审计用保存的corridor/connector候选边总数减已保存有效edge_count，恢复缺失整数，未做碰撞过滤、Dijkstra、导航或教师重计算。仅对简化schema不匹配者尝试该元数据恢复，96例严格匹配原身份；160例保持原简化schema直接匹配。该恢复是独立核验方法，不是已安装的生产修复。

最小后续修复建议：

1. 对未来nav明确保存并重载所有参与identity的字段，记录schema版本。
2. 对本次256旧产物，使用不可覆盖的兼容性/恢复清单绑定原nav和task receipt hash、96/160的原schema及期望parent/view identity；不能强行重写所有旧身份或改动D/F/A/路线。
3. 只读256旧产物做重放回归，证明无需新scene/nav/native/H即可得到同一视图、合法标签与成本；拒绝错元数据或错来源。
4. 工程修复需新源码版本/绑定和审查。原435b绑定与本次S1-A证据冻结保留，S1-B另行授权；不因修复重新消费512次评价。

本问题不是网络、奖励或策略梯度算法失败，也不能由路线验证PASS掩盖。固定DRL目标和正式实验预算不变。
