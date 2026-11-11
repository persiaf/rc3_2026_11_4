# Guard列语义

此说明解释事前协议已有的两类参考，不改变guard。

- `candidate_max_step_KL`与`candidate_max_probability_change`以及归档中`guard_reference=previous_applied_to_candidate`比较更新前已接受策略与当前候选；它们用于原增量guard接受/拒绝。
- CSV的`max_step_KL`、`max_probability_change`以及终点`per_state.guard_reference=base_to_model`是SL基座到当前已应用模型的累计描述，使用现有guard算术产生；其`accepted`字段不能作为本次更新是否被拒绝的判断。
- `base-TV`始终比较SL与候选/当前模型，上限始终0.1。
- 构造的共享见证必须一次由SL到见证通过全部原guard。其`base_to_model`实际PASS。

排序128次候选全部通过实际增量guard。终点累计KL0.003125059不意味着曾有单步KL超过0.001；全程最大真实单步KL仅0.00000390155。没有mask、幅度或实际应用保护违规，没有拒绝或回滚。
