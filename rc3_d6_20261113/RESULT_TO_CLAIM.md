# D6结果到主张

独立架构/审稿角色判定：**claim_supported=no，confidence=high**，范围仅是本预定seed、已暴露SEL8+DEV3开发筛选。`pending_external_review=false`。没有paper/PAPER_CLAIM_AUDIT.json，论文写作裁决仍为provisional；不将实验审查包装为论文级审计。

主终点256的B父等权H-J收益：FrozenSL -1.230479%，匹配继续SL -1.583450%，A 0.798505%。两个主比较和各自SEL/DEV均未达到原门槛。相对退化A的小幅改善不足以通过；B对A的DEV分组还退化1.250991%。128也没有通过，未用其选择模型。

可支持：32768旧轨迹的D/raw/H成本分解成立；基线只依赖动作前历史，q只除初始任务数一次；A/B/SL同起点、同任务序列完成256更新，当前完整代理规划可执行；本B配方没有获得要求的完整路线收益。

不能支持：方差降低、裁剪/Adam更新无偏、机制普遍优越、独立父布局泛化、部署公平加速、实机机械臂IK或关节轨迹安全认证。baseline零均值推导针对裁剪前score估计器；实际A/B256步均触发裁剪，不能只凭梯度范数或系数分布解释唯一根因。

数值复核30692项PASS，独立角色另从792条原始评价记录重算。新A前256步的权重与损失严格复现旧D5对应前缀；实际使用的是新采集轨迹、新Adam及新的H请求，没有用旧D5轨迹更新新Actor。数学成立、梯度有限、动作/停靠变化均没有替代路线指标。

独立审查完成时主复核/模型索引仍在生成，原审查快照保留这一时间事实；随后主复核已PASS，见[INDEPENDENT_NUMERICAL_AUDIT.json](INDEPENDENT_NUMERICAL_AUDIT.json)。两套审查主收益一致，原始审查不回写。共享skill附录缺失按内联契约留痕，没有伪造外部审计。

路由：关闭本估计器配方，不追加训练或其他算法。原D5 NO_DEVELOPMENT_SIGNAL、P2—P5 NOT_RUN及固定32/A1/Critic等封存边界保持。不能据此否定所有因果基线或DRL。

[独立审查prompt/raw/parsed/routing](../paper/review-traces/experiment-result-to-claim/20261113_d6_run01/TRACE_MANIFEST.json)
