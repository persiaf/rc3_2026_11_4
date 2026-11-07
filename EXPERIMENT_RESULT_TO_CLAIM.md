# 实验结果到主张记录

## 2026-11-07：v3 S2 三种子固定端点

- 实验：`rc3_v3_global_local_20261106_01`，三个固定种子各 RL512＋continued-SL512，共3072次正式更新。
- 既定研究主张：大规模复杂父场景DRL训练后，冻结策略快速完成不同规模局部任务的完整约束规划，并取得可确认的质量与时间收益。
- 独立 reviewer 判定：`claim_supported=partial`，`confidence=high`；狭义的按协议完成S2、固定权重谱系、保存数值及资源闭环得到支持。
- 未建立：RL相对frozen-SL、continued-SL、untrained及强规则的完整路径质量、时间优势和独立泛化。训练曲线不是配对评价，H收益不能归于网络。
- 下一步：保持研究目标及全部固定端点；先补全冻结协议的S3评估器，再依据独立明确授权实施S3。不增加S2更新、不挑选中间模型，不读取S3或原60保留测试。
- S3模型谱系：`GO`；S3执行：`REVISE_PENDING_FROZEN_PROTOCOL_EVALUATOR_AND_SEPARATE_AUTHORIZATION`。
- 写作判定：`provisional — no paper claim audit available`；本仓库未提供 `paper/PAPER_CLAIM_AUDIT.json`，不以内部审查冒充论文数字全审计。

依据：[本轮结果到主张](rc3_s2_results_20261107/RESULT_TO_CLAIM.md)、[独立审查](rc3_s2_results_20261107/INDEPENDENT_REVIEW.md)、[完整review trace](paper/review-traces/experiment-result-to-claim/20261107_run01/)。本新记录不替代或改写历史负结果。
