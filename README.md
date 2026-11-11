# RC3：可达约束下的二维静态任务路径规划

## 2026-11-11：M1监督修复真实验收，唯一恢复仍未拟合

[本轮实际报告](rc3_r04_m1_recovery_20261111/M1_CALIBRATION_REPORT.md)：系统电源日志确认原长停顿主要为主机睡眠，旧2859.2566759秒不扣减。独立进程监督7项真实验收通过；唯一新M1尝试11.586754秒因来源分类器误认Torch虚拟命名空间失败，轨迹恢复及C0/C1更新均0，**G3未评价，不进入G2/M3**。分类修正另通过9项标准库测试，未再次启动科学尝试。[超时测试](rc3_r04_m1_recovery_20261111/TIMEOUT_FIX_AND_TEST.md)、[资源账](rc3_r04_m1_recovery_20261111/RESOURCE_ACCOUNTING.json)。历史原文保留。

## 2026-11-11：R04-M1执行失败，未开始拟合

[本次M1实际报告](rc3_r04_m1_20261111/M1_CALIBRATION_REPORT.md)：已锁定33fit/16holdout父、116条轨迹及C0/C1生产接口源码；真实入口在源码预检发现墙钟超限，累计2859.26秒，Critic/Actor更新及轨迹恢复均0。**执行FAIL、G3 INSUFFICIENT；无新校准模型或性能结果，不进入G2/M3。** 时间硬上限未被严格守住，失败和全部消费保留，不自动补跑。[资源账](rc3_r04_m1_20261111/RESOURCE_ACCOUNTING.json)、[独立失败复核](rc3_r04_m1_20261111/INDEPENDENT_FAILURE_REVIEW.md)。下文历史原文保留。

## 2026-11-11：R04长期价值与相对策略方案及工程审查

[本轮R04完整设计和交付](rc3_r04_review_20261111/README.md)已完成C1、训练折条件基线、合法集合pre-tanh居中Actor的数学与接口原型。37项人工单元、8个保存SL-logit状态接口和41,624,938项保存CSV检查PASS，累计150.839秒；所有Actor/Critic optimizer和新规划请求为0。独立终审为设计/工程GO，生产schema迁移及校准门槛仍未通过，**未获联合PPO许可、未启动科学训练**。居中不保证post-tanh均值为0或突破原gap界；条件基线尚非必需。

[前轮Critic尺度与Actor响应诊断](rc3_critic_diagnostics_20261111/README.md)保留C1/C2的数值改善与H代价、29.79%理论锁定和99.53%共同能量。R03在128停止、256未开始、ON/OFF路线未变；原60和旧S3、旧模型与结果保持原边界。以下历史全文保留。

## 2026-11-10：206维真实PPO与ON/OFF有界训练结果

[本轮正式训练器集成](rc3_conservative_refinement/integration_20261110/README.md)已实现独立206维PPO、准确SL256永久冻结、L/K/H Critic、逐步guard与Actor/Adam严格回滚、闭合账本恢复、轻量Actor部署及完整请求计时。90项测试与20个真实工程请求通过，实际5次Actor/4次Critic调用仅用于工程验收；此前两次失败原账保留。Fixed3/Detour16真实示例有效。

**正式ON/OFF已按预定规则在128终点停止：64/128各66视图的raw/H路线和成本均未改善FrozenSL，Critic未优于常数；256未执行。** [具体协议R03](rc3_conservative_refinement/integration_20261110/PROTOCOL_DRAFT_R03.json)冻结每组最多256次机会、同一起点/日程、64/128/256终点、5100请求及5小时累计上限；用户随后明确批准参数，[协议批准与独立阶段授权](rc3_conservative_refinement/integration_20261110/authorizations)分别保存。不能由工程PASS或训练诊断推断规划收益。[实际正式报告](rc3_conservative_refinement/integration_20261110/FORMAL_EXPERIMENT_REPORT.md)、[完整资源账](rc3_conservative_refinement/integration_20261110/FORMAL_RESOURCE_ACCOUNTING.json)、[判读与边界](rc3_conservative_refinement/integration_20261110/RESULT_TO_CLAIM.md)与[下一步](rc3_conservative_refinement/integration_20261110/NEXT_STEP_DECISION.md)。仍是二维双臂代理约束，非真实IK/实机认证；旧研究与原60封存边界保持。

## 2026-11-10：老师反馈的独立工程候选

[保守修正与停靠后通行信息](rc3_conservative_refinement/README.md)保留原SL/H/mask/完整回报，加入“假设选择候选并扣除本次服务后，其余任务最近服务距离的均值/最大值”。同结构206维对照只关闭新增距离摘要；原导航信息仍保留。独立候选还提供零输出修正、训练专用L/K/H评估员及逐状态更新检查。

**上一轮状态：58项合成零优化器工程测试通过，当时未接runner、未开展新训练或性能对照。** 后续真实训练器进展见上节，尚不能声称降本或保持速度。已收尾small256负结果保持，原60封存，真实IK/实机未完成。见[上轮工程报告](rc3_conservative_refinement/EXPERIMENT_REPORT.md)、[实验计划](rc3_conservative_refinement/paper/EXPERIMENT_PLAN.md)和[上轮下一步](rc3_conservative_refinement/NEXT_STEP_DECISION.md)。以下保留原研究历史与当时状态。

本仓库归档截至 **2026-11-09** 的源码、冻结模型、实验结果、审查材料和论文展示文件。版本标识为 **2026.11.09-gl-v5-fixed-greedy-reference**。默认规划方法是 **Fixed3／Detour**；旧 MLP 保留可选入口，Structured 与 G2～G2.3 保留研究及诊断记录。

<!-- GL_V5_CURRENT_BEGIN -->
**GL-v5固定贪心参考小试完成：G128、128更新、1536采样＋512参考＋432终点评价；机制筛选REVISE。** 12个训练物理布局上的72个新任务窗口，mixed/blocked各6布局；固定参考为SL256同输入D1贪心＋H，三自然轨迹原权重、完整似然和原B512问题日程不变。见[本轮实际结果](rc3_gl_v5_20261109/README.md)与[结果到主张](rc3_gl_v5_20261109/RESULT_TO_CLAIM.md)。没有重训A/B/C或重新跑部署网格；旧108角色原文件、身份映射、负结论及两S3缺失原字节保持。以下为此前各阶段历史记录。
<!-- GL_V5_CURRENT_END -->

<!-- GL_V4_CURRENT_BEGIN -->
**GL-v4独立实验已完成：396部署配对请求、三组384更新、3072训练终局及216新任务完整评价。** 轻量部署与原完整checkpoint参数、mask/logits、路线和J/L/K严格一致，具体完整开销与启动费用逐条件报告；自然起点筛选为REVISE，B未可靠改善共同SL及继续SL。B保留原1/.25/.25，仅改变两辅助首动作的采样与计分。新36例来自3个旧DEV几何上的6个新任务draw，单种子机制筛选不能宣称最终泛化/部署成功。见[本轮全部结果](rc3_gl_v4_20261109/README.md)、[部署对照](rc3_gl_v4_20261109/DEPLOYMENT_EQUIVALENCE.md)、[自然起点结果](rc3_gl_v4_20261109/START_OBJECTIVE_ABLATION.md)和[资源/审查](rc3_gl_v4_20261109/INDEPENDENT_REVIEW.md)。原S3负结论、暴露状态及两个缺失原字节保留；以下是此前各阶段历史记录。
<!-- GL_V4_CURRENT_END -->

<!-- S3_SCOPE_RESULTS_CURRENT_BEGIN -->
**S3 scope_r1已实际完成剩余13817/13817请求，新增失败0；当前配置未建立DRL质量或速度优势。** 累计15876预扣、15874有效计划与2个原历史缺失；三seed封账，108输入和816657条同臂F&A服务分配核验通过，12固定模型及2515旧文件未改。5完整质量格／9完整部署格均未通过，另1质量／3部署格INCOMPLETE，全局AND为null。完整质量格RL+H相对frozenSL成本高1.10%–1.96%、相对Detour高3.72%–7.52%；完整部署格中位耗时为Detour的1.76–2.72倍。H收益单列，不归于学习；不追加训练、补缺失或读原60。详见[本轮全部交付](rc3_s3_scope_results_20261109/README.md)、[实际报告](rc3_s3_scope_results_20261109/EXPERIMENT_REPORT.md)、[逐请求CSV](rc3_s3_scope_results_20261109/results/REQUEST_METRICS.csv)、[独立审查](rc3_s3_scope_results_20261109/INDEPENDENT_REVIEW.md)和[下一步](rc3_s3_scope_results_20261109/NEXT_STEP_DECISION.md)。以下记录保留各历史阶段当时状态，当前以上述结果和活动状态为准。
<!-- S3_SCOPE_RESULTS_CURRENT_END -->

<!-- S3_CONNECTIVITY_CURRENT_BEGIN -->
**S3_042目标窗口可行，统一目标范围修订和恢复入口技术GO；本轮未启动剩余S3。** 事前登记后实际1次冻结导航诊断，完整2.833479秒；全父唯一阻断task221/table94在窗口外，752/753候选落入0.4米膨胀障碍区，本次23目标均可服务。保留全父诊断、完整地图/上下文/F&A与全部合法边界候选，实际目标仍硬检查；这改变S3输入接纳契约，后续标记v3+scope_r1。独立一次8项/6067断言通过，18旧保存输入语义相同、2515旧文件原字节保持。原2059预扣/2057成功/2失败与2896.9277159秒继承，剩余3233/5292/5292请求，两失败不补跑，原60未读、训练0。见[具体诊断与全部交付](rc3_s3_connectivity_20261109/README.md)、[范围修订](rc3_s3_connectivity_20261109/INPUT_SCOPE_DECISION.md)、[恢复处置](rc3_s3_connectivity_20261109/RECOVERY_DECISION.md)和[资源账](rc3_s3_connectivity_20261109/RESOURCE_ACCOUNTING.json)。下面是原停止和恢复的历史记录。
<!-- S3_CONNECTIVITY_CURRENT_END -->

<!-- S3_RECOVERY_CURRENT_BEGIN -->
**S3持久化修复工程GO，已实际恢复1895个未执行请求；随后因新的S3_042输入不可行停止。** 累计2059预扣、2057有效结果、2失败，13817未执行。原180文件、旧HEAD/pending与12固定模型来源保持；原第163槽没有补跑。6次真实瞬态锚重试成功，新失败的Journal及父阶段时钟正常闭合。首种子预算账为592保守旧账＋9.8023启动失败＋2295.1254续段＝2896.9277秒，原准确墙钟仍null。后两种子未启动。质量294/2268、cold881/6804、hot882/6804；全部6质量/12部署格INCOMPLETE，不支持总体性能判断。新失败在全父场景F&A检查、模型/native/H之前，不改场景或免费重试；optimizer0，原60未读。见[真实恢复结果与全部交付](rc3_s3_recovery_20261108/README.md)、[累计资源](rc3_s3_recovery_20261108/results/RESOURCE_ACCOUNTING.json)、[请求状态](rc3_s3_recovery_20261108/results/REQUEST_STATE_INDEX.json)与[下一步决定](rc3_s3_recovery_20261108/NEXT_STEP_DECISION.md)。

以下原IO中断段落保留为历史前缀；当前状态以上述恢复记录和活动执行状态为准。
<!-- S3_RECOVERY_CURRENT_END -->

**正式 S3 已实际启动，因日志 IO 故障中断；执行 REVISE，科学判定 NOT_EVALUATED_INCOMPLETE。** 用户批准准确 b348… 评估绑定并授权完整三种子后，首种子实际预扣163请求、完成162份有效计划；第163次在导航前更新HEAD时遇到Windows访问拒绝，二次错误记录和阶段闭合受阻。后两种子未启动，失败没有重跑或清账。132.83秒仅是已完成请求时间和，准确阶段墙钟不可还原，保守冻结592秒供明确恢复审查；optimizer0，原60未读。见 [本轮实际结果与全部证据](rc3_s3_results_20261108/README.md)、[真实报告](rc3_s3_results_20261108/EXPERIMENT_REPORT.md)、[资源账](rc3_s3_results_20261108/RESOURCE_ACCOUNTING.json) 和 [独立审查](rc3_s3_results_20261108/INDEPENDENT_REVIEW.md)。不能据这四例前缀判定质量或部署门槛。

**此前 S3 评估器工程验收为技术 GO，当时正式 S3 未执行。** 七方法、完整冷/热导航计时、固定权重身份、请求预扣与恢复、父布局聚类及原 AND 判定已实现。实际完成126/126旧DEV工程请求，累计112.330247秒、optimizer0、新教师0；RL_raw与RL_plus_H的18对原始路线一致，独立审查通过。几何来源校验已补齐并独立绑定V2评估源码；原S1-A/S1-B/S2不改，当轮S3目标与原60保持封存。见 [全部评估器交付](rc3_s3_evaluator/README.md)、[技术报告](rc3_s3_evaluator/S3_EVALUATOR_REPORT.md)、[准确源码绑定](rc3_s3_evaluator/S3_SOURCE_BINDING.json) 和 [一次性执行计划](rc3_s3_evaluator/S3_EXECUTION_PLAN.json)。这些原始工程文件保留不改；随后正式运行暴露的日志IO问题以上文本轮记录为准。

<!-- S2_CURRENT_BEGIN -->
**S2 科学执行状态：`COMPLETE_ALL3_FIXED_ENDPOINTS`，实际完成 3072 次正式更新。** seed11622026: RL512 / continued-SL512；seed11622027: RL512 / continued-SL512；seed11622028: RL512 / continued-SL512。S3模型谱系和工程评估器随后就绪，正式S3已授权且恢复后因新输入可行性失败停止，以上文本轮记录为准。S2开发记录是原18个DEV任务在256/512及三seed下的108次重复评价，不是108个独立案例，也不是S3的108目标；原S2阶段S3消费0的历史记录不改。

训练回报/成本曲线只作诊断，H收益单列；未证明DRL质量提升、泛化或SL对RL优势。原S1-A/S1-B与冻结科学证据保持；新增RL journal的路径专属LFS规则属于交付维护。见[本轮结果与全部报告](rc3_s2_results_20261107/README.md)、[下一步](rc3_s2_results_20261107/NEXT_STEP_DECISION.md)、[当前执行状态](rc3_global_local_execution_v3/ACTIVE_EXECUTION_STATUS.json)及[已核验发布收据](rc3_s2_results_20261107/PUBLICATION_RECEIPT.json)。

下面S1-B及更早段落保留其当时状态，当前以本S2记录及活动执行状态为准。
<!-- S2_CURRENT_END -->

<!-- S1B_CURRENT_BEGIN -->
**最新执行状态：S1-B三个种子已完成，共768次正式监督更新，S2技术GO但未授权/未执行。** 全256训练教师保存状态上，CE从3.8087/3.7942/3.5688降至1.9483/1.9656/1.9130，Top1为38.56%/38.60%/39.40%；这是部分教师模仿，不能宣称完整路线或DRL收益。新增scene/nav/native/H/teacher均0，原S1-A1482文件字节未变，S3/原60继续封存。见[实际训练权重、曲线与审查](rc3_s1b_results_20261106/README.md)和[当前准确授权状态](rc3_global_local_execution_v3/ACTIVE_EXECUTION_STATUS.json)。旧源码批准及原协议字节保留，新0f0ffd2源码批准与S1-B收据分别归档。

以下历史阶段段落按发生顺序保留，各阶段“当时未批准/未训练”等描述不代表当前状态；当前以上述新记录为准。
<!-- S1B_CURRENT_END -->

当前验证范围是二维静态批次任务与代理可达约束。真实机器人标定与 IK 暂缓。G2.3 的“局部窗口选择＋STOP”作为第三部分 DRL 核心的路线已结束追加训练，不能将这些记录解释为学习方法已经优于强规则。独立构造式研究已完成C0／C1和一次512更新C2小试：速度部分达标，质量及继续门槛未通过；随后完成冻结路线 C2.1：72 次共同后处理使 RL 与 Fixed3 的平均逐场景相对差缩至 +0.8170%，相同后处理下相对先验的学习增量区间仍跨零；没有追加训练或启动C3。归因见 [C2.1报告](rc3_constructive/REPORT_C21.md)，原小试详见 [研究目录](rc3_constructive/README.md) 与 [完整报告](rc3_constructive/REPORT_C012.md)。

C2.2 已完成一次匹配暖启动的raw/post目标对照：两组各128更新、共8192策略采样；新48机制场景上B相对A改善1.5491%，但相对起点平均改善−0.9084%、区间跨零，按事前规则判为无定论并停止。没有进入C3或查询60保留测试。见 [C2.2报告](rc3_constructive/REPORT_C22.md)。

当前同结构继续策略梯度训练支线已收尾，原第128次权重保留为研究参照，A/B仅作证据归档，不追加训练。严格候选间导航信息必要性方向仅记录立项门槛，原G1—G4尚未执行；后续独立R0/R1价值研究不构成该严格门槛的通过。见 [收尾报告](rc3_constructive/REPORT_CLOSEOUT.md) 与 [未来信息价值门槛](rc3_constructive/NAVIGATION_INFORMATION_GATE.md)。

覆盖—导航深度价值规划方案已完成评审，随后单独登记并完成R0：32例有界教师相对g_B＋H平均改善2.2507%，32例均正收益、最大单例贡献8.9448%，全部有效，实际2144/2304次顶层评价。R0仅通过教师机会门槛，未证明学习收益或改变既有默认方法；旧支线保持收尾；R1随后按独立任务完成，R2/R3未启动。见 [修订Word](RC3_研究模型与实验计划_20261105_评审修订版.docx) 与 [评审记录](research_proposals/20261105_nav_value_review/REVIEW.md)。用户原稿保持原字节。R0结果见 [完整报告](rc3_nav_value/REPORT_R0.md) 与 [逐场景表](rc3_nav_value/r0/per_scene.csv)，研究资源和独立审查在 [独立研究目录](rc3_nav_value/README.md)。

R1 已完成单种子有界筛选并收尾当前配置：实际9,286/16,384次顶层评价、6,144步优化；首轮两表示参考均未晋升，终点A=C、B=D。D对B改善0%，对暖起点改善区间跨零，质量与完整时间均未建立强规则部署优势。没有形成有效参考更新的API证据，也没有反证API一般无效。R2/R3未启动，60保留测试未读，默认Fixed3／Detour不变。见 [R1完整报告](rc3_nav_value/REPORT_R1.md) 与 [质量—时间表](rc3_nav_value/r1/quality_time.csv)。

R1研究证据冻结于提交 `df8cf28`。后续用户复核与候选研究边界登记在 [R1收尾补充](rc3_nav_value/r1_closeout/REVIEW_AND_DECISION.md)：优先验证教师动作识别与完整菜单预测可靠性，候选方向尚未启动。本次收尾维护不新增训练、规划或性能实验。

研究内容三的目标固定为：在大规模复杂10×10父场景中进行深度强化学习训练，冻结策略后快速规划不同规模局部任务。监督和规则仅作辅助与对照，不能替代DRL核心。新独立支线已完成 [S0全局—局部环境一致性](rc3_global_local/REPORT_S0.md)：3父图18窗口、63条未训练轨迹及精确重放，1次无更新梯度探针；S1/S2/S3正式训练和性能验证未启动。完整目标与后续协议见 [固定研究目标](rc3_global_local/RESEARCH_OBJECTIVE.md) 与 [研究目录](rc3_global_local/README.md)。S0提交`057830d`已随`codex/global-local-preflight`研究分支历史上传GitHub；旧研究结论不因软件检查通过而改变。

S1/S2正式执行前评审已完成：**正式训练REVISE，本轮工程准备PASS**。实际111物理布局满足预定独立划分；36端点基线与18全候选闭环参照通过；三个已存S0父图的9完整策略轨迹和3次反向通过，优化器更新为0。新增不可变输入/缓存与端点适配层，原S0/v1/v2原字节未改。v3仅为新提案，未激活；完整正式runner、batch/RNG/恢复与权重谱系验收仍须补齐。S3目标及原60测试未评价。见 [本轮全部交付](rc3_global_local_impl_review/README.md)、[实施判断](rc3_global_local_impl_review/IMPLEMENTATION_REVIEW.md) 与 [执行方案](rc3_global_local_impl_review/NEXT_STEP_DECISION.md)。用户明确要求GitHub交付后，已推送至`codex/global-local-preflight`，未合并main；[仓库内交付与远端核对记录](project_management/github_delivery_global_local_preflight/README.md)取代此前仅本地交付状态。

S1/S2正式训练器已实现，登记工程验收与独立复核通过：batch4梯度等价、命名随机流、S1/S2连续与恢复状态逐位相同、独立权重分叉及无授权拒绝。累计21次工程Adam、12次真实完整H，正式科学训练更新0。教师失败后隐式重试预算漏洞已修并零更新回归；两次绑定和旧源字节保留。**工程GO，正式执行仍REVISE**：v2保持活动冻结设计，v3及执行澄清待正式批准，不自动激活，不消费S3/原60。见 [训练器六项交付](rc3_global_local_trainer/README.md)、[工程验收](rc3_global_local_trainer/TRAINER_VERIFICATION.md) 与 [下一步决定](rc3_global_local_trainer/NEXT_STEP_DECISION.md)。成果在`codex/global-local-preflight`交付，不合并main。

此前v3候选定稿轮次完成八项差异澄清及独立阶段授权链验收，当时尚未批准、科学调用0。该轮9组75断言为零科学工程回归；原候选与旧v2历史指针继续保留原字节，后续真实批准和S1-A结果以下文及新的活动执行状态为准。候选SHA256为`0c5bc7d7a9557edf2af76f51a9f418550f0b4ce1ec6a8b47ddea95c6256e78bb`。见 [本轮交付](rc3_global_local_execution_v3/README.md)、[协议定稿](rc3_global_local_execution_v3/PROTOCOL_FINALIZATION.md)、[S1执行准备](rc3_global_local_execution_v3/S1_PREPARATION.md) 与 [下一步决定](rc3_global_local_execution_v3/NEXT_STEP_DECISION.md)。

S1-A此前授权核查曾因缺少两项明确决定停在准备阶段，该轮科学调用0，记录保持在 [授权核查报告](rc3_s1a_authorization_check/README.md)。后续用户已正式批准准确v3和仅S1-A，分别新建真实收据并激活执行版本，未覆盖旧v2指针或原候选字节。

**S1-A已实际完成**：256教师、native256+H256=512次评价、190.153795秒、缓存0/收费失败0/optimizer0；512份返回路线独立代理约束复核通过。平均教师J=129.77862746，额外H改善41例、不变215例，没有神经学习。审计发现nav保存缺少参与identity的`rejected_edge_count`，影响首96例的直接重载，因此**S1-B技术REVISE，未执行也未授权**。原D/F/A/路线及批准源码未改，没有重跑教师。见 [真实实验结果](rc3_s1a_results_20261106/README.md)、[256例明细](rc3_s1a_results_20261106/PER_TEACHER.csv)、[重载问题与最小修复](rc3_s1a_results_20261106/NAVIGATION_REPLAY_FINDING.md) 和 [下一步](rc3_s1a_results_20261106/NEXT_STEP_DECISION.md)。当前准确活动版本与权限见 [批准执行状态](rc3_global_local_execution_v3/ACTIVE_EXECUTION_STATUS.json)，S1-B/S2/S3和原60边界不自动解除。

**S1-A导航兼容修复已完成，S1-B技术GO**：实际生产路径256/256重放、512保存路线复核、96完整/160简化原身份保留，另6次同图LRU交错通过；原1482文件/564,969,808字节未变。28项格式/来源负测与阶段桥接33断言、独立5125断言通过，新scene/nav/native/H/model/optimizer均0。新执行源码绑定SHA为`0f0ffd2d2dda2fefefcfcb5cb26dc5197b5d11bd1332b0a5546992409049bfe0`，旧v3协议/435b绑定/原教师不改；新源码及S1-B正式执行仍需对应明确批准/授权。见 [实际修复与验收](rc3_navigation_compat_20261106/README.md)、[新源码绑定](rc3_navigation_compat_20261106/SOURCE_BINDING_UPDATED.json) 和 [获准即执行的阶段计划](rc3_navigation_compat_20261106/S1_B_EXECUTION_PLAN.json)。没有运行S1-B/S2/S3，原60继续封存。

## 获取完整文件

模型、训练数据与历史 ZIP 使用 **Git LFS**。先安装 Git LFS，然后克隆并获取大文件：

```powershell
git lfs install
git clone https://github.com/persiaf/rc3_2026_11_4.git
cd rc3_2026_11_4
git lfs pull
```

需要有该私有仓库的访问权限。仅下载 GitHub 自动生成的源码 ZIP 不作为本项目完整恢复方式；完整恢复应检出指定提交并执行 `git lfs pull`，再运行文件清单校验。

## 当前使用入口

在仓库根目录创建 Python 环境，安装运行依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r rc3_v5/requirements-runtime.txt
.\.venv\Scripts\python.exe -B -m rc3_v5.plan --method fixed3 --scale 10x10 --mode blocked --seed 771903 --output outputs/fixed3.json
.\.venv\Scripts\python.exe -B -m rc3_v5.plan --method detour --budget 8 --scale 10x10 --mode blocked --seed 771903 --output outputs/detour.json
```

配置预检入口为 `python -m rc3_v5_1.robot_input`。未填写有效标定的真实空模板返回 `blocked_inputs` 是预期状态。训练环境依赖与各阶段命令保留在对应目录中，不能用运行依赖替代全部历史训练依赖。

## 版本与证据入口

| 目录 | 用途与状态 |
| --- | --- |
| `rc3_baseline/`、`rc3_fast/`、`rc3_improve/`、`rc3_learning/` | 初期无学习、选择器与学习原型的历史记录 |
| `rc3_v3/`、`rc3_v4/` | 可靠规划演进及冻结主实验；v4 继续提供主算法比较证据 |
| `rc3_v5/` | 当前规划入口、共同约束、缓存及输入准备维护证据 |
| `rc3_v5_1/` | 当前机器人配置预检入口 |
| `rc3_rl_research/`、`rc3_rl_g2/`、`rc3_rl_g21/`、`rc3_rl_g22/`、`rc3_rl_g23/` | G0/G1 至 G2.3 的研究协议、训练、对照与诊断记录 |
| `rc3_g23_closeout/` | 已执行的收尾决定、只读复算和论文可用表述 |
| `rc3_constructive/` | 独立构造环境、插入基线、一种子REINFORCE小试、重放验证、冻结路线共同后处理归因及学习优势尚未确立的证据 |
| `paper/`、`paper_figures_v1/` | 论证组织、固定案例和展示材料 |
| `*_import/`、根目录历史 ZIP／审查文档 | 依赖来源、原始交付与审查证据，原样保留 |
| `project_management/`、`tools/` | 本轮备份、版本清单、清理收据与恢复工具 |

G2.3 测试包含 6 个独立场景、180 份输出；多步模型共 36 次测试，“严格降本且完整时间不超过 0.8 秒”为 0/36。每个最终多步模型为 512 次优化更新，3,840 是全部种子和对照的合计。有限次要学习增量与主目标失败同时保留，详见 [收尾报告](rc3_g23_closeout/REPORT_CLOSEOUT.md)。

## 备份与清理

按用户后续要求，本机只保留 `project_backups/2026-11-04_g23-closeout/full_before_cleanup.zip` 这一份完整清理前快照及必要说明／收据。它采用 ZIP 内 SHA256 内容去重存储，包含当时的原始文件和本机 Python 环境。与 GitHub 已保存内容重复的标准项目 ZIP 及无用操作日志已删除，具体记录见 `project_management/backup_retention_20261104.json`。

后续项目变更持续版本化，按对应任务授权推送并核对远端，默认不再生成额外整包备份或重复解压目录。维护约定保存在 [AGENTS.md](AGENTS.md)。本机快照不进入 Git；原始交付／审查 ZIP 继续作为研究证据保留。

清理仅针对可重建计算缓存、迁移测试解压副本和解释器字节码缓存。源码、已保存路径、检查点、训练日志、协议、审查材料、原始 ZIP、机器人资产及本机 Python 环境保留。训练缓存重建方法见 [清理与恢复说明](project_management/ARCHIVE_AND_RESTORE.md)。

前轮归档与清理没有新增训练；本轮构造式C2训练和性能记录单独保存在 `rc3_constructive/`，没有覆盖旧实验。文件通过 SHA256 清单核验。后续研发使用独立输出目录，分支及同步按AGENTS约定执行；Git不存储账号凭据、本机虚拟环境或额外整包备份。


## Integrated-RPPO-v1（独立研究）

[Integrated-RPPO-v1](rc3_integrated_rppo/README.md)：冻结SL骨干、残差Actor、独立Critic与受控PPO；按新任务P0—P5条件执行。原S0—S3及GL-v4/v5、两个缺失槽和负结论保留，原60继续封存。阶段与实际结论以模块报告为准。


### Integrated-RPPO small256 已收尾（2026-11-10）

首seed已实际完成256轮／1024 PPO Actor更新，预登记P2未通过：所选RL128对SL仅+0.2233%，对匹配监督256为−0.8682%；未启动其余seed／独立确认／正式计时／P5。固定DRL目标保持，原研究证据和60保留任务不变。

[实际报告](rc3_integrated_rppo/EXPERIMENT_REPORT.md) · [冻结Actor](rc3_integrated_rppo/ACTOR_INDEX.json) · [独立审查](rc3_integrated_rppo/INDEPENDENT_REVIEW.md) · [阶段与下一步](rc3_integrated_rppo/NEXT_STEP_DECISION.md)。
