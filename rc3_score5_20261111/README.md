# Score5：冻结表示，仅训练原评分五张量

独立候选；工程验收已登记、待执行，正式科学训练和独立保留集评价未授权。[历史说明](HISTORICAL_CLARIFICATION.md)明确S2/GL-v4/GL-v5已做全Actor训练，本候选不以“主Actor可训练”泛称新机制。

允许：key.weight、dynamic.weight、nav.weight、query.weight、query.bias。原表示与prefix GRU冻结；不使用固定32权重或detached206残差入口。

[工程审查](IMPLEMENTATION_REVIEW.md) · [准确协议](PROTOCOL.json) · [源码](SOURCE_BINDING.json) · [日程](SCHEDULE.json)
