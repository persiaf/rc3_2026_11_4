# 开源参考与适配范围

本轮查阅一手论文、作者仓库及官方环境教程，未宣称原设定原样复现。

- [CSP-Attention 论文](https://arxiv.org/html/2102.05875v1)及[作者代码](https://github.com/kevin031060/CSP_Attention)：覆盖路线、动态覆盖表示和 REINFORCE 已有先例。[训练代码](https://github.com/kevin031060/CSP_Attention/blob/master/train.py)采用成本减基线的策略梯度。作者城市同时为需求与服务点；本项目任务与服务候选分离、使用导航图距离、指定depot、加停靠费。CSP适配训练对照留待C3，本轮没有运行或冒称复现该对照。
- [AM 贪心 rollout 基线代码](https://github.com/wouterkool/attention-learn-to-route/blob/master/reinforce_baselines.py)：冻结策略在同一实例上贪心求成本；用配对检验决定是否替换基线。本项目每128更新仅在开发场景比较 raw J，不按选择集更新 rollout 基线。
- [RL4CO 自定义环境教程](https://rl4co.ai4co.org/examples/3-creating-new-env-model/)：参考 reset／step、合法掩码、reward及解验证的接口划分。本轮采用精简自定义PyTorch实现，没有引入RL4CO训练器或宣称其自定义环境集成已完成。

新代码自行实现，不复制大段外部实现。没有把“注意力＋动态覆盖＋跨规模”作为新颖性结论。本轮完整轨迹log概率求和，不对服务步数归一化。作者 attention 模块的网络抓取未取得可读源码，故不把其细节当作本轮已独立核验的事实。

新增的可学习规则先验显式登记并保留初始权重与纯先验对照；小试不能单靠质量较好就认定关系编码或RL机制已有独立贡献。正式C3仍需CSP适配、BC／一步及聚合特征对照。
