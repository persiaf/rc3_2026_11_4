# D5候选实施与数据准入收口

结论：**五张量工程验收PASS；完整任务科学周期P0→P6技术数据未完成，P1—P5 NOT_RUN**。

任务书要求同一清单、50%完整10×10任务的PG/继续SL对照。准确合法256教师池只有4×3=86、6×6=84、8×8=86，完整10×10=0。`DATA_ADMISSION.json`未通过，无法按批准分布构造有完整标签的共同32768槽位日程。本轮不生成新教师，不用不同分布SL冒充匹配，不改门槛。

已实际实现原Actor评分模块微调：key.weight、dynamic.weight、nav.weight、query.weight、query.bias，共25152/65088标量；39936表示/GRU参数冻结。沿用原输入、评分、mask、转移、覆盖、端点、J=L+.75K、H和贪心。新Trainer采用任务书Adam1e-4、clip1、完整J差×全动作logprob，未沿用旧B-cond、优势RMS、A1幅度或更新guard。

工程实际通过9项检查：准确SL256与独立存储；完整及三局部尺寸初始化logits/mask/raw/H路线等价；严格五张量/FullActor优化器身份；no_grad采样缓存重新进入可微key映射；真实PG梯度及更新/旧缓存拒绝；已有局部标签SL五张量非零梯度和冻结字节；FullActor原参数梯度连通；Actor-only冻结导出等价和用途隔离；checkpoint恢复后同一条已采集轨迹的下一更新精确一致。

恢复证据边界：检查了模型、Adam、RNG状态、游标和请求身份，以及同一后续轨迹的更新一致。没有独立重新采样两条恢复前后轨迹来比较采样流，也未实现/执行科学作业恢复驱动。当前运行入口提供P0、状态与条件阻塞返回；**不能称完整科学训练、冻结推理和统一评价CLI已全部交付**。Actor导出/加载及请求函数可用，已由P0实际调用。

原P0：13次新完整请求，5次真实Adam；准确保留前置1次失败请求/0优化器。独立审查发现旧写账适配把逐请求H/search字段清空以避免重复累计；原字节不改。R02仅修记账接口，1次普通4×3 FrozenSL请求、0优化器验证：H1、two_opt4、joint_search1与全局增量一致。实际累计15次完整尝试=14完成+1失败，Actor5、Critic0。未新增场景、导航、教师、native或保留集调用。

13条旧工程记录的H按冻结terminal源码和非空raw完整路线恢复到更正表；原逐请求搜索细账未保存，明确UNKNOWN。新接口逐请求计数与全局增量一致。原P0成功宏观encode35、score1130，R02另encode1/score12；原字段483只覆盖请求路径，不能当全阶段forward。前置失败未保存的forward/search细账仍UNKNOWN。

只读独立数值复核275项：14条新完成记录的J=L+.75K、raw/H完整输出字段、覆盖分配、保存的合法mask与完整return；5条更新记录；2个新模型payload与准确SL基座的冻结字节；原53行表/封存与旧Actor/Trainer的Git身份。新增forward、训练、规划、H均0。

完整逐请求raw/H J/L/K、成功与失败、完整时间、停靠集合及访问顺序变化见 `ENGINEERING_REQUEST_SUMMARY.csv`、`CORRECTED_REQUEST_RESULTS.csv` 和原压缩记录。这些是普通任务工程观测，不作为任务级科学收益。FrozenSL、SL-U/C、D5-PG、Full-PG、Fixed3、Detour16的正式开发/确认/最终比较全部NOT_RUN，指标为null。

监督器至R02累计78.357480秒，包括前置失败和登记；只读审查5.342908秒另计，最终作业总账见RESOURCE_ACCOUNTING。纯文档及Git维护分列，不伪装新训练。9项工程PASS不能用于保证4096×8科学吞吐：本轮未测完整逻辑batch，吞吐准入尚未到达。

旧S2、GL-v4、GL-v5已做SL起点全Actor训练；相应历史说明已补入老师材料。53行表、固定32关闭状态、旧负结果和权重原字节保留。没有将未运行阶段写成科学成功或失败。
