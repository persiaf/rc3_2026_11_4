# G2.2 Research Findings

2026-11-04，独立委派 reviewer 完成保存数据复算。实验来源：72 份固定权重完整规划，12 个旧 extra 百桌开发场景，6 方法，无训练、无 PPO、无额外分支。

- `claim_supported = partial`；`confidence = high` 仅针对有限记录描述和门禁判断。
- 未查询动作有可核验的正负实例，不能当作统一负样本；即时奖励、实际冻结后缀与相对同状态 STOP 的记账必须区分，负后缀不等于不值得继续。
- 继续 CE 相对根 STOP 平均回报 −0.014086，1 胜 / 7 平 / 4 负；CE+收益 +0.045764，5 胜 / 0 平 / 7 负。CE+收益相对继续 CE 的有限变化属于 BC 模型比较，不是 PPO 增量。
- 原 Detour 的 J 对两个学习模型都在 12/12 场景更低；固定早停 Detour 的平均 J 和平均共享回报优于 CE+收益，且 12/12 次单次计时更快。现有记录不支持稳定实用优越性。
- 72 份返还路线的成本、折线长度、分配、奖励账本与汇总一致；最终返还最好路线，不混同末端当前路线。代理验证不包含 IK 或真实标定。
- 路由：`supplement-design-only / narrow-claim`。本轮评估已结束；不自动追加训练、PPO、场景、分支或消融。后续设计需要用户授权后再执行。
- `paper_claim_audit = unavailable`；论文写作结论标记为 **provisional — no paper claim audit available**。

完整审查见 `EXPERIMENT_RESULT_TO_CLAIM.md`；复算与角色追踪见 `paper/review-traces/experiment-result-to-claim/2026-11-04_run01/`。
