# 已有训练、完整任务证据与停止记录

只读现有报告、训练器和允许的保存记录。旧S3仅阅读既有汇总报告/状态，不读取旧任务、逐请求路线或封存集合。各研究的输入、终点与均值分开；以下不拼成新的性能排名。

|阶段|真实完成与范围|支持的结论/停止边界|原证据|
|---|---|---|---|
|S1-A/S1-B|256局部强教师；三seed各SL256，总768监督更新；末点教师状态Top1约38.6%—39.4%|部分模仿，非路线/泛化优势；保留准确SL来源|`rc3_s1b_results_20261106/README.md`、WEIGHT_LINEAGE.json|
|S2|三个seed各全Actor RL512+继续SL512，科学更新合计3072；三seed共享阶段各约2955/3030/3254s|真实DRL已训练；训练回报/梯度不证明胜过SL/规则。不是只训练残差，也不是PPO统称|`rc3_s2_results_20261107/README.md`；`rc3_global_local_trainer/engine.py`|
|S3历史评价|既有汇总：15874有效/15876已扣请求，2历史缺失；5个完整质量格和9个完整部署格均未过；其余INCOMPLETE，全局AND=null|未建立监督/强规则的质量—时间优势；缺失不补，不改成“所有格完整失败”|仅`rc3_s3_scope_results_20261109/README.md`，本轮无旧S3任务语义读取|
|GL-v4/GL-v5|SL初始化后的全Actor实际训练已实施；v4三流A/B/C各128，v5G128加冻结贪心参考|均未建立共同SL及继续SL的可靠任务增量；不能把“主Actor可训练”当尚未试过的新机制|`rc3_score5_20261111/HISTORICAL_CLARIFICATION.md`及两GL NEXT_STEP_DECISION.md|
|R03|冻结SL+206维保守残差，PPO/完整回报，64/128评价，128停止；原2376完整请求、Actor384/Critic320消费保持|ON/OFF/监督的raw/H完整路线与FrozenSL相同；没有路线收益；256未运行|`rc3_conservative_refinement/integration_20261110/FORMAL_EXPERIMENT_REPORT.md`|
|R04-M1|原step128 G3=FAIL，256未运行，旧C1来源/账保持|不补C1、不改G3、不恢复旧M3|原M1停止报告；本轮只继承封存事实|
|R04-M2|fit-only B-global/B-cond及CG/CB残差候选真实执行；CB删除，模型仅归档|条件统计不等于学习残差收益；离线holdout已暴露；不重开Critic|`rc3_r04_m2_20261111/NEXT_STEP_DECISION.md`|
|冻结参考在线PG|B-global/B-cond作为动作无关训练参考；两RL及匹配SL至128，64/128均无有效完整路线增量|128封闭，256不延长；B-cond并未建立在线路线优势|`rc3_frozen_reference_pg_20261111/NEXT_STEP_DECISION.md`|
|固定32已封存|三LR CE实际323次：128+128+67；共享见证PASS32/32；CE14/32、Adam排序18/32、QP13/32；共享排序128尝试；QP32外层/112候选|全部已知状态拟合诊断，非路线评价。323/128/QP原账分别保留，诊断权重不得初始化；不续补1e-3|`rc3_task_level_decision_20261111/FIXED32_CLOSURE.md`及原账|
|D5修订|96完整教师+256局部教师；同任务4096×8；五张量PG4096和SL-U4096，科学8192更新、PG32768在线轨迹、SL32768已有标签重放|固定4096 H-J父配对收益对FrozenSL−2.6431%、对SL-U−3.7371%；P1 NO_DEVELOPMENT_SIGNAL；P2—P5 NOT_RUN|`rc3_d5_revision_20261112/EXPERIMENT_REPORT.md`、P1_DECISION.json|
|D6|准确SL256重新起点，A原损失/B因果剩余成本/SL各256，Actor768，完整请求4888、标签重放2048，Critic0|B对同轮A +0.7985%，但对FrozenSL−1.2305%、对SL−1.5834%，128和256均未过；不能以少退化过关|`rc3_d6_20261113/EXPERIMENT_REPORT.md`、NEXT_STEP_DECISION.md|
|R07|48互异训练父布局、144状态，288主请求+16固定复测=304；无更新；同前缀Top1/Top2再相同SL后续/H|H后33改善/76平/35退化；≥0.5%改善23/144，局部5/48；OPPORTUNITY_INSUFFICIENT。存在差异，但本配置机会未过门槛|`rc3_r07_20261113/EXPERIMENT_REPORT.md`、RESULTS.json|

## D5/D6的数值口径

固定66暴露开发任务、11物理父布局（SEL8+DEV3），每父6视图。旧质量门槛采用“每父先分别平均两方法J，再取相对差，最后父等权”，本轮`gain_of_parent_mean_J`复现旧数值。新结构分析另报“每任务相对差父内平均再父等权”，列`relative_gain`；二者不能互换，也不把全部均值比值当配对收益。

D5与D6都能改变实际停靠集合及顺序，故“动作完全不可改变”不是这两轮失败解释。D6最后相对FrozenSL少约0.985配置、却多2.349m导航，平均J多1.610；D5少约0.545配置、却多4.093m导航，平均J多3.684。这些是保存路线分量恒等式，不是“排序是唯一根因”。

R07所选48问题，已有强教师相对FrozenSL父等权每任务差距3.8139%；本轮精确重现该子集。扩展到全部352训练问题/96父的统计必须另列，不以新范围均值覆盖3.8139%。强教师不是所有问题都更优，也没有全局最优证书。

## 初始化与资源谱系

下一版唯一可复用神经基座为准确未微调SL256：

`rc3_integrated_rppo/runs/rc3_integrated_rppo_v1_20261110_01/actors/11622026/SL256_0256_1817f58aa650.pt`

文件SHA256 `60f5aa4e59983e92d565f0599823b710965cf2c96a041e7eb483d23a8eb0556c`；tensor SHA256 `1817f58aa6506456ff12f0c5fe0a914e156aef8af319f4687dde90782f55a63e`。本轮没有加载该权重；来源从已有登记读取。所有32诊断、D5/D6失败终点、CG/CB均只归档。

每一旧资源账保持原字节，不把本轮0更新回写为历史0，不把剩余额度视作R08授权。旧原60、旧S3、确认/最终集合边界保持；监督/规则只作初始化、辅助与对照，固定DRL目标不改。
