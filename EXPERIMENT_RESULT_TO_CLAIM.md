# 实验结果到主张记录

## 2026-11-08：S3持久化修复及真实恢复停止

- 独立审查 `claim_supported=partial`，confidence=high：限定持久化工程命题yes，完整S3科学性能命题no（证据不齐，不能解释为模型性能已被否定）。写作仍provisional，未提供paper claim audit。
- 17项零科学工程验收通过；实际新增1896预扣/1895成功/1新输入前置检查失败，累计2059/2057/2，13817未执行。原180文件、旧HEAD/pending及12固定权重来源保持，旧失败未重跑。6次真实瞬态锚重试无重复追加，新故障的Journal和父阶段收费收尾闭合。
- 停止点S3_042在全父10×10闭合端点连通服务检查，早于局部提取及模型/native/H。保存F/A中233/233任务存在同臂候选；不能断言F&A自身缺覆盖或4×3窗口无解。没有持久化本次D/连通掩码，具体端点过滤任务及更深根因未独立确定。
- 质量294/2268、cold881/6804、hot882/6804，6质量/12部署格全部不完整，不作总体原门槛判定。无训练、原60未读；固定DRL目标和全部终点不变。
- 路由：保留停止与累计账，先处置新的真实输入失败；不自动跳过/换seed/免费重试，不重新批准未变的v3设计。见[实际报告](rc3_s3_recovery_20261108/EXPERIMENT_REPORT.md)、[独立审查](rc3_s3_recovery_20261108/INDEPENDENT_FINAL_FAILURE_REVIEW.md)及[结果到主张](rc3_s3_recovery_20261108/RESULT_TO_CLAIM.md)。

## 2026-11-08：正式 S3 因日志 IO 中断

- 准确 b348… 评估源码及完整 S3 获用户明确批准，真实阶段运行产生163预扣、162有效结果、1持久化错误；后两种子未启动。没有重跑失败、追加训练或消费原60。
- 完整科学命题的独立判定：claim_supported=no，因为预登记网格未完成；正确状态是NOT_EVALUATED_INCOMPLETE，不能写成质量门槛失败或强化学习无效。
- 已支持的窄结论：已保存162计划符合代理约束，原模型/来源不变，失败机制定位为Windows原子HEAD替换及异常持久化级联。
- 路由：单独修复IO与阶段闭合，保留163预扣/失败/原日志并明确资源结转后再考虑原S3恢复；不重复批准v3、不增加模型性能门槛。详见 [实际报告](rc3_s3_results_20261108/EXPERIMENT_REPORT.md)、[独立审查](rc3_s3_results_20261108/INDEPENDENT_REVIEW.md) 和 [完整追踪](rc3_s3_results_20261108/paper/review-traces/experiment-result-to-claim/20261108_run01)。

## 2026-11-07：S3 评估器工程验收

- 狭义命题：冻结模型 S3 评估器具备按已批准 v3 正式执行的技术条件。
- 实际证据：126/126 旧开发请求、112.330247 秒、72 次共同 H、18 对 RL 原始路线一致、12 权重原始哈希、真实闭合恢复和合成父聚类统计；几何来源问题已修复，最终静态回归11/11通过。
- 独立判定：claim_supported=yes，confidence=high，仅支持评估器技术 GO；正式 S3 尚未执行，科学性能命题未评价。
- 路由：沿已批准 v3，等待对应 S3 阶段授权后使用固定终点；不追加训练或增加模型性能前置门槛。详见 [结果与主张](rc3_s3_evaluator/RESULT_TO_CLAIM.md) 和 [独立追踪](rc3_s3_evaluator/paper/review-traces/experiment-result-to-claim/20261107_run01)。

## 2026-11-07：v3 S2 三种子固定端点

- 实验：`rc3_v3_global_local_20261106_01`，三个固定种子各 RL512＋continued-SL512，共3072次正式更新。
- 既定研究主张：大规模复杂父场景DRL训练后，冻结策略快速完成不同规模局部任务的完整约束规划，并取得可确认的质量与时间收益。
- 独立 reviewer 判定：`claim_supported=partial`，`confidence=high`；狭义的按协议完成S2、固定权重谱系、保存数值及资源闭环得到支持。
- 未建立：RL相对frozen-SL、continued-SL、untrained及强规则的完整路径质量、时间优势和独立泛化。训练曲线不是配对评价，H收益不能归于网络。
- 下一步：保持研究目标及全部固定端点；先补全冻结协议的S3评估器，再依据独立明确授权实施S3。不增加S2更新、不挑选中间模型，不读取S3或原60保留测试。
- S3模型谱系：`GO`；S3执行：`REVISE_PENDING_FROZEN_PROTOCOL_EVALUATOR_AND_SEPARATE_AUTHORIZATION`。
- 写作判定：`provisional — no paper claim audit available`；本仓库未提供 `paper/PAPER_CLAIM_AUDIT.json`，不以内部审查冒充论文数字全审计。

依据：[本轮结果到主张](rc3_s2_results_20261107/RESULT_TO_CLAIM.md)、[独立审查](rc3_s2_results_20261107/INDEPENDENT_REVIEW.md)、[完整review trace](paper/review-traces/experiment-result-to-claim/20261107_run01/)。本新记录不替代或改写历史负结果。
