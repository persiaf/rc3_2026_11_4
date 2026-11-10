# 128早停后的决定

本配置已按批准R03终闭合：64、128两终点ON/OFF均未改善冻结SL，Critic value MSE分别为固定warm常数的2.4122、2.3107倍。两项规则同时触发，**停止本配置，不追加训练，不执行256，不放宽guard、不换场景求正**。准确协议、批准、失败工程、完整路线和未用额度保留。见[正式报告](FORMAL_EXPERIMENT_REPORT.md)及[独立审查](INDEPENDENT_FORMAL_REVIEW.md)。

三种方法各128次Actor候选全部接受，正式拒绝0；ON/OFF最大相对基座TV约3.49e-5/3.31e-5。不能将无收益归因为guard拒绝；也不能由更新很小认定任意增大学习率会有效。训练G/V/A、Critic量纲和排序响应需要联合查看，因果关系尚未验证。

老师可立即基于保存证据开展三项具体任务：

1. **核查Critic的尺度和条件信息。** 对照[逐父校准](runs/conservative_paired_11622026_v1/analysis/CRITIC_CALIBRATION_PARENTS.csv)、[预测分位表](runs/conservative_paired_11622026_v1/analysis/CRITIC_CALIBRATION_BINS.csv)及[PPO优势](runs/conservative_paired_11622026_v1/analysis/PER_STATE_ADVANTAGES.csv)，分窗口/任务数检查L、K、H信用与归一化value偏差。先说明为何未胜固定常数，不把损失下降写成长期预测有效。
2. **核查修正是否改变有效排序。** 对照[逐机会guard](runs/conservative_paired_11622026_v1/analysis/GUARD_UPDATES.csv)、[逐请求raw/H路线](runs/conservative_paired_11622026_v1/analysis/ALL_RAW_H_ROUTES.csv)及冻结源码，检查206维新摘要、logit差异和候选排序间距的关系。区分模型接受更新但贪心选择未变与信息永远无用；不直接放宽既定阈值。
3. **形成下一版可检验修订。** 根据前两项诊断提出最小Actor/Critic表示或尺度修订，明确同SL起点、ON/OFF因果对照、完整成本/时间目标、源码/H/环境绑定与停止条件，再提交新的实施审查、协议及阶段授权。核心仍为大父布局DRL训练、冻结后不同规模局部完整规划；不替换成动态重规划或纯监督，也不直接增加本配置轮数。

新训练、多seed、额外独立布局或保留测试均未启动。原60、旧S3和未批准final继续封存；历史模型、旧实验和用户Word未覆盖。当前结果为单seed、8个已训练父布局及3个独立但已暴露开发父布局的二维代理证据，未建立泛化、真实IK或实机能力。默认Fixed3/Detour保留。
