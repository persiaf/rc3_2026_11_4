# D5 data-revision 独立结果到主张审查

审查角色：独立 paper architect/reviewer。审查对象为 `d5-data-revision-20261112-r1`，不采用作者的结果解释作为判据。`claim_supported = no`；对本轮固定 P1 判定的置信度为 **high**。写作判定为 **provisional — no paper claim audit available**：`paper/PAPER_CLAIM_AUDIT.json` 不存在。本次已由独立代理完成审查，**不是 pending external review**。

## 判断与核算

待检验主张是：准确 SL256 初始化后，冻结表示、只训练原 Actor 五张量的完整长期回报 D5-PG，能在完整任务上优于 FrozenSL 及同更新/同任务暴露的 SL-U，并达到原 P1 门槛。原 `MASTER_PLAN.md:62` 要求4096主终点对两者的 HJ 父等权收益分别至少1%和0.5%，SEL、DEV各自均不负，且候选完整可行。**本轮完整执行了 P1，但主张不受支持，应按预登记规则直接 P6 收口。**

本角色自行读取9个开发评价 JSON 的594条逐请求压缩成员，核对压缩指针、原始JSON SHA、任务/父身份、raw/H指标和保存计划证书。两种表示的1188个计划检查均满足保存数值的 `J=L+0.75K`，对应计划长度、停靠数和证书长度一致。每个模型/基线都有66/66有效记录，11父，每父6请求；这验证保存证据的内部一致性，没有重新运行模型、几何验证、mask重放、H或规划。

统计严格先在每父内平均6个请求，再算 `(control_mean-candidate_mean)/control_mean`，最后对父等权。负号表示候选更差。结果不是汇总均值之比，也不是66个请求作为66个独立父。

| 固定4096 D5-PG对照 | 全11父 HJ收益 | SEL8 | DEV3 | 正收益父/11 |
|---|---:|---:|---:|---:|
| FrozenSL | -2.64308061% | -2.36190860% | -3.39287263% | 1 |
| SL-U4096 | -3.73708690% | -3.23751874% | -5.06926863% | 1 |
| Fixed3 | -4.93101404% | -4.61930835% | -5.76222922% | 0 |
| Detour16 | -6.43598317% | -6.19001298% | -7.09190369% | 0 |

1024/2048/4096三个固定非零终点对FrozenSL的HJ收益分别为-2.8983%、-4.8236%、-2.6431%；对同期SL-U分别为-2.9319%、-4.5824%、-3.7371%。没有一个非零登记终点出现总体正收益，且正式判断仍只使用4096，不挑选最优点。原始Actor路线的4096 raw-J收益对FrozenSL为-19.7069%，对SL-U为-28.0073%。共同H明显降低了保存路线的代价，但不能把这一后处理作用记成PG相对对照的改进。

同一4096终点，D5-PG/FrozenSL/SL-U/Fixed3/Detour16的请求平均HJ依次为141.57343/137.88965/136.27742/134.76431/132.92670。平均值可作描述，前表才是登记收益口径。SL-U对FrozenSL的父等权HJ收益为+1.0124%，只支持本轮监督对照的局部开发描述，不能转记为DRL成果。

D5-PG相对FrozenSL的raw与H停靠集合、访问顺序均在66/66请求改变；不存在“相同停靠集合、只换顺序”的配对。这个结果支持模型确实改变了解码行为，不能支持路线改善，也不能分离集合选择与排序的因果贡献。梯度、loss变化、输出改变及软件PASS均不能替代上表。

## 实现与来源

- `rc3_decoder_program_20261111/actor.py` 的 `configure/trainable_parameters/validate_optimizer` 与继承的 `rc3_score5_20261111/actor.py` 将D5更新限定在 `key.weight,dynamic.weight,nav.weight,query.weight,query.bias`，25152/65088标量。`policy.py:82-90` 表明它们直接作用于原query、候选key、剩余任务动态量与导航量的合法动作logits；39936标量与prefix GRU冻结，未增加残差头或状态ID输入。
- 原 `trainer.py:27-35` 使用完整未归一化的 `detach(J_H_sample-J_H_FrozenSL_same_task)*sum(logpi)`，批内平均；SL使用每episode的动作平均CE。Adam为1e-4、clip=1，无critic、优势归一化、额外A1幅度或更新接受guard。新 `training.py` 继承此更新数学，仅提供独立流/恢复封装。数据路径中PG走在线独立采样，SL读教师完整动作；FrozenSL参考不提供PG动作。
- 两条正式 `UPDATES.jsonl.gz` 各含连续1..4096，前后权重SHA链连续，第一步before均为准确SL256 tensor SHA `1817f58aa6506456ff12f0c5fe0a914e156aef8af319f4687dde90782f55a63e`。它们未从工程更新权重起步。保存导出的五张量变化/冻结字节检查另有 `INDEPENDENT_NUMERICAL_AUDIT.json`；本角色核查实现和更新链，不将该PASS当质量证据。
- 新96个完整教师对应96物理父，原Detour16 horizon16后统一固定H；无FrozenSL伪教师。保存标签和清单SHA一致，保存的当前Environment重放全部完成；256原局部标签逐字节SHA仍匹配本轮索引。没有重新调用教师或导航。`FULL_DATA_CENSUS.json` 的父训练任务数70–252、候选208–732是训练暴露规模描述，不是新父性能或机器人认证。
- 自行检查三份不可变日程均为4096×8，每批4完整+4局部，局部(2,1,1)/(1,2,1)/(1,1,2)轮换；每seed完整槽16384、三局部槽5462/5461/5461，352唯一任务，跨seed暴露多重集相同。生产 `science.py:148,205,218-227` 使同seed PG/SL读取同一任务表。两条实际更新日志都有4096×8唯一episode ID；全部32768实际PG任务/mask的逐条对应由独立数值审计补证，本角色未再运行该大循环。相同任务暴露不意味着PG与SL的状态/动作分布或计算成本相同。
- 当前R04B绑定、science-03实际依赖校验、当前文件三方在12个必要源码上SHA相同，包含Actor、Trainer、policy/environment/H及新data/runtime/science/training/evaluation/clocking。实际环境为Python3.10.20、Torch2.12.1+cu126、NumPy2.2.6、CPU单线程确定性。源码绑定仍保留部分旧run_id/base字段，必须用 `revision_run_id`、本轮协议/批准、科学身份和 `ACTIVE_EXECUTION_SOURCE.json` 联合解释，不能把旧来源包当成本轮授权。Git提交/远端未由本角色检查。

## 恢复、资源与已知缺口

科学执行记录不是“无中断单次运行”。science-01在740更新后的741批slot3扣账后出现WinError5；失败请求保留，恢复继续同问题，重用同Actor前三条已存轨迹。science-02在1771后的登记维护前已存1772批8条轨迹，未尝试该次优化器；随后从相同模型/Adam/RNG/采样Generator/游标继续。恢复收据、生产实现及连续更新链相互一致，累计恢复己方轨迹11条；独立审计未发现重复实际科学优化器更新。这里支持实际恢复工程成立，不支持性能提升或任意故障下的普遍恢复保证。

本角色独立计数新 `EVENTS.jsonl` 得到33939请求尝试、33938完成、1失败、8197优化器尝试。其中科学8192、工程新增5；新用途请求为teacher192/reference352/engineering32/eval594/train32769。加继承15请求、5优化器尝试，得到科学结束原始总账33954请求（33952完成、2失败）与8202 Actor尝试，和 `RESOURCE_SCIENTIFIC_FINAL_RAW.json` 一致。工程总59请求/10尝试，未越过60/24；科学为4096 PG+4096 SL，critic=0。用途和阶段计数交叉，不能把它们相加：工程期最初12条参考也属于reference用途。SL的32768次教师回放不是32768条新完整规划请求。

该科学结束快照累计27279.319022秒：工程6515.670951秒、训练池20763.648071秒，数据335.318755秒是训练池内子账，不能再次加到总墙钟。PG训练流含参考分摊的全墙钟16139.334602秒、SL流4055.000980秒，均低于8h；系统睡眠2363.847163秒仍包含在全墙钟和预算内。R04B仅将SL-C匹配改为活跃墙钟，未退款。科学03监督器真实确认进程树退出，status COMPLETE、耗时11833.438秒；前两个中断的墙钟也均保留。

仍须准确保留以下限制，不能用软件PASS掩盖：

1. `RESOURCE_SCIENTIFIC_FINAL_RAW.json.recording_limits` 明确1772批维护时未持久化的部分forward/backward尾部细计数为UNKNOWN；继承的原13请求内部搜索及前置forward细项也为UNKNOWN。最早可证明新源码之前的零散读取时间未测。总监督墙钟、失败与实际optimizer可核算，不能据此声称完整FLOPs/forward细账已无缺口。
2. `original_forward_calls/feature_states/feature_s/request_io_s` 是继承字段，不能当本轮全部新forward或延迟；本轮encode/score计量也不能补齐上述UNKNOWN尾部。内部搜索inclusive/exclusive不能叠加。未保存的原生DP细分不能填0。
3. 原始快照是“科学结束”账，不含随后失败数值审计的保守276.950322秒、成功审计的监督器全进程墙钟715.594秒及本轮后处理。最终交付总账应另外纳入真实后处理且保留失败；本角色没有修改资源账。初版审计list/tuple比较错误属于审计器失败，不能改写成科学数据失败。
4. `STATUS_WORKER_FINAL_RAW.json` 的phase/status/closed/P1_gate已表明P6收口，但 `next_action` 仍是启动P1、`scientific_result` 仍为null，并保留旧error。这些是原始状态遗留字段；汇总交付应明确引用最终gate和关闭原因，不能按旧next_action继续运行。原始文件应保留。

## 本轮能写与不能写的结论

可写：本轮补齐了合法96完整教师与原256局部标签的共同日程；D5-PG和同五张量SL-U从同一SL256出发完成首seed4096更新；在暴露的11父66请求开发网格上所有保存计划通过代理模型证书，但D5-PG的固定终点对FrozenSL及SL-U均退化，未通过P1，故按计划终止后续阶段。

不能写：D5-PG改善完整规划质量；冻结范围优于全Actor训练；RL优于监督；已获得独立泛化或跨seed稳定性；实现部署加速；已满足真实机械臂安全/IK/关节轨迹约束。SEL8本身是训练物理父的新任务，DEV3虽父布局与训练不重叠，却已暴露用于开发筛选，既非确认24父也非最终24父。9格共594行不是594个独立父。

完整请求开发时间可原样报告，但输入冷热不同、包含压缩归档且截止于后续pointer/event/resource写入前；日志尾部仍在作业墙钟内。FrozenSL/PG/SL的p50/p95不能直接作为部署速度或纯网络forward比较。Fixed3/Detour16的raw是含原生内部搜索的native-final，不是神经no-H消融；两者raw与H相同不表明共同H没有作用。

SL流保存了名为 `SL_C_candidate_0.actor.pt` 的4096候选槽，但 `compute_ratio=0.2943252497`、`compute_matched=false`、`C_complete=false`。它与U同权重仅因P1截止，不能称匹配C已完成或已评价。Full-PG、补足SL-C、后两seed、部署优化、确认/最终层均NOT_RUN；缺失是条件门槛未准入的结果，不是这些阶段成功或失败。旧v1仍是技术数据未完成、P1—P5 NOT_RUN，不能与本修订实际科学否定混写。

建议主张改为：**在补齐完整教师、严格匹配任务暴露后，原Actor五张量的完整回报PG完成了固定首seed试验，但在开发完整规划上未优于FrozenSL和匹配监督更新；本次负结果界定了该冻结解码器配方的证据边界。** 本次结果不能给出退化的单一因果机制，更不能否定所有DRL方法。

## 路由

`NO_DEVELOPMENT_SIGNAL → P6_CLOSEOUT`。本周期不启动补充诊断、新算法、新seed或新预算，不恢复固定32/A1/Critic旧支线。`next_experiments_needed=[]`；缺失的归因、复现、独立层与部署证据仅登记。后续写作只可形成范围明确的负结果/工程证据草稿，并保持“provisional/no paper claim audit”。图表质量审查待作者提供本轮图和caption后补充，不改变本判断。

可复核独立计算为同目录 `reviewer_saved_numeric.py` 和 `REVIEWER_NUMERICAL_RECOMPUTATION.json`。审查辅助脚本曾以“批内任务ID必唯一”作未预登记假设而停止；随后删除该额外假设，保留episode ID唯一要求，完整运行通过。另一个只读结构预览曾对None调用len而停止。两者均未触及科学输入或执行模型，也未生成新科学结果。


## 图表与状态后续审查

三幅本轮图的数据/选择规则与状态规范化已独立审查，见同目录FIGURE_REVIEW.md与FIGURE_REVIEW.json。原始状态遗留字段的报告歧义已由规范化STATUS.json解决；图表仅有英文图注/单位的小修，科学结论与P6路由不变。
