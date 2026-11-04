# RESULT-TO-CLAIM EVALUATION — 独立 paper architect/reviewer

本记录保留实际委派任务的规范化文本。角色与执行者分离；阅读完整 NEXT_RUN_SPEC、user_request、EXPERIMENT_PLAN、protocol、verification_protocol 和 core/learning/trial。不得执行规划、修复、训练或模型前向，不得调用 trial 或 verify 脚本。所有输出只写 G23。等待 analysis.json、label_verification.json 完成后，只读保存记录并独立复算。

判定两层问题：规则操作的可表达性及同资源前缀复现；在相同接口和完整 .8s 软限下，多步价值策略是否对规则、匹配一步、冻结初始化、等更新静态控制及原 Detour16 达到平均相对 J 改善至少 .3%，且配对 CI 下界>0，返还方案通过代理终验且无超时。

必须核对：实际 API 闭环与一次静态回归的边界；规则接受语义；一步标签包含次态 STOP；MC continuation 冻结及参数摘要；matched_static_rule_value 与 multi 的同初值、同 D1 首转移、同优化预算及不同后缀评价策略；6 新场景×2重复×3 seeds 不能当36独立场景；完整计时及超时；训练每 seed 仅4场景；双臂分边32维双 graph encoder 与独立数值头；额外标签复核计原8192账；旧文件冻结。

返回 claim_supported yes|partial|no、what_results_support、what_results_dont_support、missing_evidence、suggested_claim_revision、next_experiments_needed、confidence。主目标失败必须执行用户止损，不自动追加局部窗口+STOP同方向轮数/损失或扩规模；不泛化成所有同类算法无效。未完成的证据不得当成功。

保存 EXPERIMENT_RESULT_TO_CLAIM.md、findings.md，以及本目录中的 prompt、raw reviewer response、parsed verdict、routing 和 independent recalculation。论文 claim audit 不存在时按 skill 标记成文 provisional。

主要证据来源：inputs/rc3_g22_review_and_g23_spec/NEXT_RUN_SPEC.md、inputs/user_request.txt、EXPERIMENT_PLAN.md、outputs/protocol.json、verification_protocol.json、core.py、learning.py、trial.py、repair_counted.py、继承环境/编码器代码、alignment.json、132条训练pickle、模型快照（仅张量读取）、verified_labels.json.gz、label_verification.json、test_results.json、plans/、plan_validation.json、analysis.json、resource.jsonl、resources.json、freeze_before.json、runtime_metadata.json。
