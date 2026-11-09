# GL-v5固定贪心参考基线自然起点DRL

仅G128真实训练128更新，1536采样＋512冻结参考＋432固定终点评价，共2480请求。机制筛选 **REVISE**。固定SL256/fresh Adam和原B512问题/自然随机流，唯一主要变量为冻结SL贪心+H成本基线。

- [完整条件、raw/H后质量及机制判断](EXPERIMENT_REPORT.md) · [逐更新/逐路线CSV](TRAINING_AND_ROUTE_RESULTS.csv)
- [训练曲线](TRAINING_CURVES.csv) · [采样/参考成本与优势](SAMPLE_REFERENCE_COMPARISON.csv) · [各实例梯度](GRADIENTS_BY_INSTANCE.csv)
- [432完整路线摘要](results/ENDPOINT_EVALUATION.csv) · [原始完整计划、梯度、权重/Adam/RNG](runs/SCIENCE/)
- [模型张量谱系与固定终点](WEIGHT_LINEAGE.json) · [准确源码绑定](SOURCE_BINDING.json)
- [2048训练请求的实际参数/参考actor来源](TRAINING_REQUEST_IDENTITIES.json)
- [事前协议](PROTOCOL.json) · [512日程与72新任务清单](INPUT_REGISTRY.json) · [优化前实际输入封存](runs/SCIENCE/NEW_TASK_BANK_FROZEN.json)
- [资源账](RESOURCE_ACCOUNTING.json) · [独立审查](INDEPENDENT_REVIEW.md) · [结果到主张](RESULT_TO_CLAIM.md) · [下一步](NEXT_STEP_DECISION.md)
- [实施前评审](IMPLEMENTATION_REVIEW.md) · [数学/RNG/角色工程检查](ENGINEERING_VERIFICATION.json) · [Git版本核对](GIT_VERSION_AUDIT.json)

12个已训练物理布局mixed/blocked各6，每布局1新task draw，三尺寸两端点。不是未见环境测试。旧证据、108原角色字段与追加映射、旧负结论和两缺失保持，未重训A/B/C、未重跑D0/D1/D2、未访问新S3或原60。SL/规则作辅助对照，最终研究方法保持DRL。PT/NPZ沿用Git LFS；工程提速/H收益不记作学习增量。
