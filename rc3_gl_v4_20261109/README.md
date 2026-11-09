# GL-v4部署拆分与自然起点DRL对照

实际完成396部署请求、三组384更新、3072训练终局请求和216新任务终点评价。部署严格等价PASS；自然起点单种子筛选：**REVISE**。旧S3及两缺失原样保留。

- [部署等价、大小、启动和完整时间](DEPLOYMENT_EQUIVALENCE.md) · [396逐请求CSV](DEPLOYMENT_BENCHMARK.csv)
- [起点目标完整对照及raw/H成本](START_OBJECTIVE_ABLATION.md) · [384更新＋216路线CSV](TRAINING_AND_ROUTE_RESULTS.csv)
- [完整路线216行](results/ENDPOINT_EVALUATION.csv) · [训练曲线](results/TRAINING_CURVES.csv) · [匹配比较](results/MECHANISM_COMPARISONS.json)
- [资源账](RESOURCE_ACCOUNTING.json) · [独立数值审查](INDEPENDENT_REVIEW.md) · [下一步](NEXT_STEP_DECISION.md)
- [原始角色字段遗漏的追加澄清](METHOD_IDENTITY_CLARIFICATION.json) · [216份精确模型身份映射](METHOD_IDENTITY_AUDIT.json)
- [预登记协议](PROTOCOL.json) · [512共同日程及36新任务声明](INPUT_REGISTRY.json) · [准确源码绑定](SOURCE_BINDING.json) · [实施前评审](IMPLEMENTATION_REVIEW.md)
- [部署策略包及来源](actors/ACTOR_INDEX.json) · [固定128终点策略包](runs/SCIENCE/DEV216/ACTOR_INDEX.json) · [实际完整checkpoint/Adam/RNG/梯度与计划](runs/)
- [Git版本与远端核验](GIT_VERSION_AUDIT.json)

新任务使用3个已有DEV物理布局上的6个新task draw，非未见S3。单种子不能证明最终泛化。精度/网络/奖励/H未变；工程速度、后处理与学习收益分开。所有PT/NPZ按现有Git LFS管理，检出提交后执行`git lfs pull`。不生成额外整包备份、不合并main。
