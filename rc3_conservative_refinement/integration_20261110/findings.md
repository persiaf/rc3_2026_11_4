# R03研究发现

- 实验：conservative_paired_11622026_v1，准确协议a20eca45…、绑定ea605e87…、seed11622026。
- 意图：后服务mean/max距离信息在冻结SL起点的保守DRL中改善完整成本并保留时间价值。
- 判定：no；high仅限当前冻结配置；provisional（新支线paper claim audit不可用）。证据见INDEPENDENT_FORMAL_REVIEW.md/.json与review-traces。
- 实际：ON/OFF/继续监督各128次更新；64/128六方法各66视图；ON/OFF全部raw/H路线和成本与SL相同。Critic value误差连续高于固定常数，完整时间无优势，128按两规则停止。256未启动。
- 不支持：长期预测有效、距离摘要改善路线、部署晋升、多seed/未暴露父布局泛化、真实IK或实机能力。
- 尚不能确定：Critic尺度、信用分配、微小排序移动与输入作用各自的因果贡献。分量误差或损失下降不能自动解释主结果。
- 约束：不追加本配置轮数，不放宽guard，不换场景，不把补跑预算重置；不解封原60/S3/final。保持大父布局DRL训练、冻结后局部完整规划目标。
- 路由：核查保存校准/优势/排序证据，提出新协议并独立实施审查；不执行新训练或保留评价。三项具体任务见NEXT_STEP_DECISION.md。
