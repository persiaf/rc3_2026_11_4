# 保守PPO后：Critic尺度与Actor排序离线诊断

基准：`83b0df9ceb6b43c5e04de883db1376558393af4d`。本目录是独立的离线机制研究，保留冻结基座、小幅残差和训练期Critic框架。R03原协议、128停止记录、ON/OFF及继续监督结果、成本和时间均保持原字节；256终点不恢复。

首次数值计算前一次登记[精确协议和源码身份](PREREGISTRATION.json)、[128条轨迹](TRAJECTORY_SELECTION.json)、[完整父布局划分](CALIBRATION_SPLIT.json)及[256步共同批次](BATCH_SCHEDULE.json)。原保存记录缺少完整特征和候选logits，采用精确保存输入和原动作前缀重建，核对原选中动作logit/概率与全局任务服务集合。不会生成新任务、选择新动作、调用导航、教师、native或H。

代码入口为`rc3_critic_diagnostics_20261111.run`，必须使用原`rc3_learning/.venv/Scripts/python.exe`（Python3.10.20、Torch2.12.1+cu126、NumPy2.2.6）。入口只允许一次执行；闭合或失败日志存在即拒绝重启。首次登记使用`python -m rc3_critic_diagnostics_20261111.register`，同样只允许一次。查看和复核现有结果无需重新执行拟合。

首次使用默认Python时发现没有torch，1.3731034秒、0前向/优化器的失败已保留。精确环境恢复使用`.\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_critic_diagnostics_20261111.recover_environment`，同一日志接续、累计已耗墙钟，原登记和算法源码不改；恢复收据和原失败完整保留，不是新预算或拟合重播。

交付入口：

- [Actor排序响应](ACTOR_RANK_RESPONSE.csv)及[全合法候选证据](ACTOR_CANDIDATES.csv.gz)
- [Critic匹配校准](CRITIC_MATCHED_CALIBRATION.csv)与[分窗口/阶段统计](CRITIC_STRATIFIED_CALIBRATION.csv)
- [机制发现与五项回答](MECHANISM_FINDINGS.md)
- [资源账](RESOURCE_ACCOUNTING.json)、[独立数值复核](NUMERICAL_RECHECK.json)
- [最小后续修订](NEXT_MINIMAL_REVISION.md)

Critic的C0/C1/C2均为268维，使用相同随机净张量初始化、拟合折、批次、Adam及256步。C0为原raw分量参数化；C1直接预测按初始任务数归一分量；C2加入组合value损失。初始化的物理输出含义不同，必须同时看0/256终点，不能将差异全部归因于优化速度。

离线holdout只表示父布局不进入本轮Critic拟合；其轨迹仍来自历史R03训练父布局。state-pooled、父布局等权与R03原停止指标分别保留。当前仍为二维代理实验；MSE下降、动作翻转及guard接受均不构成路线收益或真实IK证据。
