# D6：因果剩余成本估计器

本候选只改策略梯度的逐动作成本系数。Actor、五张量白名单、准确SL256、Adam1e-4、裁剪1、完整gamma=1、batch8、mask、端点、J=L+0.75K与固定H均保持。

令动作前历史为F_t，包含已执行动作、剩余覆盖任务与前缀路径。第t次停靠的原始增量为D[current,node]+0.75；最后一次另加D[node,exit]，所有raw增量之和为J_raw。只在最后增量加入J_H−J_raw，最终和为J_H。动作前prefix不含本次停靠、未来动作、出口闭合或H校正，等于已走导航长度+0.75×已停靠次数。以原始D及保存导航leg逐项相互核对，float64绝对容差预定1e-7。

A严格沿用旧D5损失：`mean_episode(detach(J_H−J_ref_H) * sum(logpi_t))`，保留原float32乘法/归约顺序。

B使用`w_t=detach(J_H−prefix_before_t−q_t*J_ref_H)`、`q_t=remaining_before_t/N_initial`，损失`mean_episode(sum(w_t*logpi_t))`。全部J和prefix保持原总成本单位；除q的一次比例计算，不再除初始任务数，也不按动作数平均PG。SL仍沿用mean_episode(mean_action CE)。

prefix与q只依赖动作前F_t，J_ref_H是同完整问题的冻结SL贪心+H常数，故b_t=prefix_before_t+q_tJ_ref_H在给定F_t后不依赖本次动作。对固定合法mask，E[b_t grad log pi(a_t|F_t) | F_t]=b_t grad sum_a pi(a|F_t)=0。由此B仍是完整终端目标E[J_H]的score-function估计器，假设成本/H/输入没有显式可训练参数。H即使重排或删除此前停靠，也包含在完整J_H中；tail允许为负，不能裁为零。第一个动作prefix=0、q=1，A/B系数必须相同。

这只是候选估计器的因果与单位说明，不证明方差降低、优化更快或路线改善。旧D5轨迹只核算保存成本和损失贡献，不参与新Actor更新。原梯度范数是八条混合批次的范数，不能从中辨识每尺寸参数梯度的贡献。

主终点256；128固定报告、不选中间模型。B对FrozenSL及匹配SL的父等权H-J收益均须≥0.5%，SEL与已暴露DEV各自不退化，全部代理约束有效。另报告B对A，不用A退化替代两个主比较。
