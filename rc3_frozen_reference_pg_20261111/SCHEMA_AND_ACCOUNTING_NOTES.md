# 容器语义与计账说明

新Actor-only与训练checkpoint使用本轮独立A1 schema，校验准确角色、工程/科学阶段、SL256来源、条件表和源/日程身份。部署包没有条件表、Critic或Adam，冻结参考不参与部署选点。

完整轨迹容器复用已验收的只读collector格式，并增加本轮method/role、stage和correction_semantics。旧容器中的V_old=0及critic_predictions=null是未启用Critic的占位字段，**不是本轮训练参考**；Trainer不读取V_old。本轮真正用于更新的G、B-global、B-cond、两种中心化回报及A_sample，均在ONLINE_UPDATE_DIAGNOSTICS.jsonl.gz内逐状态记录，并由新输出审计独立复算。

Actor/FrozenSL的raw与共同H后路线、J/L/K和完整约束证据分别保存。原生Fixed3/Detour16沿用其完整原生recipe/search后处理，raw列采用原生最终路线，不是另做一个“去掉原生后处理”的实验；没有额外共同H。

H_invocations字段捕获显式调用的共享baselines.postprocess前后路线。原生recipe的two_opt/joint_search走原生求解入口，不经过该函数，因此不能把H_invocations解释为原生全部后处理调用链。原生完整最终路线、native_info以及recipe/construct/two_opt/joint_search计数和包含/独占时间均实际保存；不据此声称单独测出了原生H增益。协议中“全部原生H前后路线”的预期覆盖应收窄到这一实际边界，保留原协议原字节，不以补跑或改写记录填补。

REQUEST_TIMING.csv测量从请求开始至完整归档、索引和完成journal之后；计时收据自身写入仍计入阶段墙钟。REQUEST_INDEX.csv的request_wall_s在索引写入前取得，主时间对照使用更外层REQUEST_TIMING.csv。每请求包含已保存输入/导航恢复、模型权重加载、provider/新增特征、策略/native、H、导出验证和主要I/O；一次性Python/Torch导入及阶段初始化另在完整阶段墙钟中。这里是已有运行时内的冷模型加载，不等同每次重新启动Python进程。文件系统缓存未控制；已验证字节缓存命中、实际哈希字节数和输入恢复消费如实记录。

监督标签重放不是完整规划请求，已有768次重放单独记录其输入、状态数和总耗时；不是新教师请求。Actor机会、真实Adam调用及失败/拒绝不因回滚或工程入口修复清零。工程模型不进入科学初始化。
