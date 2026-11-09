# GL-v4实际实验报告

部署396/396完成，等价PASS；三组各128更新，共384；RL请求3072、固定终点评价216，共3288，新增失败0。部署阶段305.698028/1800秒，训练与评价共享阶段1767.577342/7200秒。

模型包仅改变部署生命周期，保留参数/实际mask/logits/路线/J/L/K一致；完整开销是否降低及启动费用见[DEPLOYMENT_EQUIVALENCE](DEPLOYMENT_EQUIVALENCE.md)。自然起点仅改变强制首动作的采样/计分，保留原权重与detached LOO；真实新36例的B对A/SL/C及强规则完整结果、raw/H归因和单种子筛选REVISE见[START_OBJECTIVE_ABLATION](START_OBJECTIVE_ABLATION.md)。

新任务6draw/3旧DEV几何在优化前封存；不宣称新布局泛化。原S1/S2/S3权重、scope_r1、两旧缺失及负结论原字节保持，未追加旧训练或消费新S3/原60。新协议、实例日程、源码绑定在首实际调用前已保存并提交，运行源码SHA为`d8cf86b93a7f189c401ba43c2c7f6e656186d03e45b4e5f3087d5e7ca5460df7`。

实际模型与路线来源通过独立复核。发现并追加披露108份A/B/C结果的通用默认角色字段标注遗漏；实际actor/tensor/source和CSV分组全部正确，原始结果字节与冻结源码不改写，见[身份澄清](METHOD_IDENTITY_CLARIFICATION.json)。

[全部逐请求/逐更新CSV](TRAINING_AND_ROUTE_RESULTS.csv)、[部署CSV](DEPLOYMENT_BENCHMARK.csv)、[资源账](RESOURCE_ACCOUNTING.json)、[独立审查](INDEPENDENT_REVIEW.md)、[下一步](NEXT_STEP_DECISION.md)。
