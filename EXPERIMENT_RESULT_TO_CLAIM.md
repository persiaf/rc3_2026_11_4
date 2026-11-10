# 实验结果到主张记录

## 2026-11-09：GL-v5固定贪心参考自然起点DRL

- intended_claim：在72个新任务窗口上同时改善共同SL、既有B和C的完整成本。claim_supported=no，confidence=high；事前AND筛选REVISE。
- 实际G128更新128次、1536采样终局＋512参考＋432评价，共2480请求；未重训旧对照，参考模型冻结、独立缓存/RNG，所有优势含负值保留。
- One registered seed11622026 G128 run completed128 updates and2480 requests within989.3444799s. Direct recomputation of432 saved results on72 new views over12 trained physical layouts reproduces parent-averaged H-final gains versus SL/B/C of -2.8641351%/-2.4678596%/-3.2426334%, with95% bootstrap intervals strictly negative. All432 result artifacts are valid and finite and match CSV SHA256. H lowers G's own raw mean J by25.8719397199, but the trained G policy loses to every primary control before and after H. The reference adds512 requests,33.3333% relative to1536 sample training requests.
- 缺口：The separate exhaustive numerical/provenance audit is still running and INDEPENDENT_AUDIT.json is absent at review issuance; do not call it PASS. No paper/PAPER_CLAIM_AUDIT.json or PAPER_PLAN.md is available. Independent optimization seeds, unseen physical layouts, a matched gradient-variance comparison, causal clipping/scale diagnostics and real physical calibration/IK are absent; these omissions limit broader claims and do not make the failed screen a positive result.；路由：REVISE: preserve and report the negative result, complete the already-running read-only audit, retain the fixed DRL research goal, and do not automatically extend training or advance this recipe to independent-multiple-seed confirmation, ablations, or confirmed writing.。单优化种子、12已训练布局上的新任务；非最终泛化或部署证明，原研究DRL目标不变。
- [完整证据](rc3_gl_v5_20261109/EXPERIMENT_REPORT.md) · [独立角色审查及轨迹](rc3_gl_v5_20261109/INDEPENDENT_REVIEW.md)。旧证据和角色原文件保持。

## 2026-11-09：GL-v4部署拆分与自然起点对照

- 实际396/396部署请求、三组384优化器更新、3072终局请求和216新任务方案；独立数值/来源审计PASS。
- 部署3生命周期的参数、mask/logits、路线、J/L/K一致。工程完整开销按role/cache报告，不算RL学习收益，H单列。
- B只改变两强制首动作，保留原1/.25/.25与detached LOO；同起点、同512问题日程、fresh Adam，必须对A/SL/C全部检查。单种子candidate=False，具体方向和区间见[完整结果](rc3_gl_v4_20261109/START_OBJECTIVE_ABLATION.md)。
- 新36任务来自3旧DEV布局6新draw，不是新布局泛化；不把旧S3变回未见测试、不填补其缺失、不自动扩训，固定DRL研究目标保持。

## 2026-11-09：S3 scope_r1三种子冻结正式结果

- 实际新13817/13817成功、失败0，累计15874有效/15876预扣，两个旧缺失未补值，全部未执行请求已处理。108局部输入、816657条服务分配、固定模型与2515旧证据复核通过。
- 独立reviewer `claim_supported=no`，`confidence=high`，具体审计命题与性能命题区分见[独立审查](rc3_s3_scope_results_20261109/INDEPENDENT_REVIEW.md)。完整质量5格与完整部署9格均未通过；另1质量／3部署格缺失，全局AND为null。
- 当前RL+H未建立相对监督或强规则的质量/完整速度优势，无完整规模通过原部署门槛；H改善属于后处理，不能记为RL学习收益。不外推DRL一般无效，不更改固定研究目标。
- 路由：归档实际负结果、窄化论文主张；不自动补齐失败、追加训练、挑中间模型或读取原60。论文数字审计/provisional边界按独立审查记录，不能冒充完整论文审计。

依据：[实际结果](rc3_s3_scope_results_20261109/EXPERIMENT_REPORT.md)、[结果到主张](rc3_s3_scope_results_20261109/RESULT_TO_CLAIM.md)、[原始数据](rc3_s3_scope_results_20261109/results/REQUEST_METRICS.csv)。

## 2026-11-09：S3_042诊断与目标范围修订

- 独立reviewer判定partial/high：具体诊断/工程命题yes，完整S3质量/部署仍no/incomplete，不解释为模型性能失败。
- 收费1次冻结navigation，原保存F/A与诊断D下，目标23任务可服务，唯一全父阻断task221/table94位于窗口外。旧D哈希未知，明确是诊断重算；不修改场景/A/目标/安全间隙。
- 统一scope_r1保留全父诊断、实际目标硬检查、完整上下文和边界候选；确实改变输入接纳契约，不是完全无科学影响。原训练/网络/奖励/H/权重不变。
- 一次8测试/6067断言通过，18旧局部输入语义相同，2515冻结文件保持；实际S3请求0。两失败槽保留，原账与剩余预算不清零，无完整原网格或总体AND性能结论。
- 路由：按准确新绑定/一次恢复处置继续原顺序未执行请求，不重批未改v3、不补值/换seed/重训。依据：[诊断与报告](rc3_s3_connectivity_20261109/README.md)、[独立审查](rc3_s3_connectivity_20261109/INDEPENDENT_REVIEW.md)。

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


## Integrated-RPPO-v1-small256（2026-11-10）

固定DRL目标保持；真实完成首seed256轮，P2未通过，claim_supported=no／confidence=high，写作provisional。所选RL128对SL H后+0.223271%、对匹配监督256为−0.868168%；raw对SL−0.094853%。只属8个已训练物理布局的开发选择，未触发其余seed／独立晋级／最终216／计时／P5。不以工程、loss、H收益替代DRL增量。

[实际证据](rc3_integrated_rppo/EXPERIMENT_REPORT.md) · [独立审查](rc3_integrated_rppo/INDEPENDENT_REVIEW.md) · [研究发现与收尾路由](rc3_integrated_rppo/findings.md)。原S0—S3、GL-v4/v5和旧失败保留。
