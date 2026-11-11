# R04验收与进入联合PPO的条件

**当前允许验收的是方案与零优化器工程原型；联合PPO当前为NOT_READY/NOT_AUTHORIZED。** 本文门槛是新提案，不改写R03门槛或历史停止记录。

## G0：历史与来源

必须保留R03在128停止、256未开始、ON/OFF完整路线未变、C1未全面胜条件常数、C2的H代价、29.79%多合法状态理论锁定及99.53%共同能量。原60/旧S3任务访问均0，旧源码、实验、权重及根LFS规则字节不变。明确“数值更好”不等于已经完成统计显著性检验。

本轮检查所有指定诊断代码及CSV的来源hash、完整算术、父划分和口径。旧审计函数的再次使用只算维护复核，不伪称第二份完全独立实现；新增分窗口/阶段streaming矩统计单独标识。没有重跑模型或任务性能。

## G1：本轮独立工程

- 37个人工fixture单元测试全部通过；期望错误mask/非法输入用例必须明确抛错，不能静默接受。
- 零头在保存SL logits上合法值bitwise相同且mask相同；非法值不进mean；单合法输出0；所有合法残差abs<=0.25。
- pre共同平移不变、common方向梯度为0；必须保留非零post-tanh均值反例和post-center可能破坏幅度界的反例。测试通过仅证明公式性质。
- 原guard接受/拒绝及fake-state精确恢复均通过；不得写成真实Actor/Adam更新验收。近tie的合法首选翻转也可能被guard接受，仍需未来完整路线验证。
- 排序及actual-action pre/post ratio使用同一mask/候选ID，不能把pre ratio1写成无更新；缺collection likelihood则显式缺项。
- C1分量单位、signed H、初始n、raw恢复及旧loss等价通过；不再二次除n。268维不改变，返回schema和metadata不能进入旧tensor日志循环。
- 条件统计只接受fit物理父、固定可观测键；验证污染/未来H/最终路径字段拒绝；unknown bin走fit-global；三分量和组合value严格一致。
- 新Actor-only mapping往返logits/排序一致，旧schema/critic/Adam混入必须拒绝；人工冻结encoder梯度为0且train切换不启用。真实SL加载、真实路线、实机与生产文件loader未被本轮测试代替。

实际测试、失败和消费以ENGINEERING_REGISTRATION/RESOURCE_ACCOUNTING/UNIT_TEST_RESULTS/OFFLINE_INTERFACE_RESULTS/EVIDENCE_AUDIT为准，不能预填成功。

## G2：科学集成前的生产阻断

旧export/loader和checkpoint语义字段必须共同升级；同形状absolute/relative或raw/normalized权重不能自动互换。必须实现并审查typed C1适配、schema恢复、条件表hash（若启用）、Actor与Adam候选快照及严格回滚、基座永久冻结、累计RNG/游标/查询/资源恢复。原R03生产入口继续封存。

需另行登记真实optimizer工程用例、容差与总额并获对应授权，验证接受、拒绝、暂停/恢复和actor-only完整请求等价；本轮所有optimizer为0，fake-state测试不足以使G2通过。此要求来自项目AGENTS的真实optimizer工程分账规定，不是要求现在训练。

## G3：未来C1离线校准门槛

需要新的精确数据/源码注册及阶段授权。使用未参与前轮50父校准的预先固定16个R03父布局作新离线验证，仍不能称未见SL/DRL场景；仅元数据选择，不读结果后替换。任何尺寸不足3个验证父，结论为INSUFFICIENT，不混入state数伪增样本。

固定步数/seed/初始化/训练fold/批次；同时保存0/128/256和两种聚合。C1的value MSE对训练折条件常数，两聚合点比值均<=0.95，按物理父配对重采样95%上界均<1；同时优于C0，归一L/K/H各相对C0不得>1.10，每尺寸value点比值<=条件常数。任何一项未满足即不进入联合PPO。初始物理预测不同必须报告，不能作纯梯度因果声明。

新门槛拟合采用state SUM/episode均值以匹配拟议PPO；报告MSE的状态均值并不改变拟合归约。该多项门槛是预先固定的保守项目决策标准，不是已经校准为全局5%错误率的多重检验，也不替代未来确认性统计设计。

**现有C1两聚合均未胜条件常数，所以G3目前未满足。** 这是本轮评审得到的实际判断；不通过放宽门槛或直接换成C2获得GO。若提议条件基线，需新独立实验批准，并和其自身常数及同结构global-anchor residual比较，不能自动追加配置。

G3作为全部四臂pilot的前置条件是主动收窄研究风险的项目选择，并非Actor居中数学上依赖C1胜过lookup；若G3失败，记录为当前联合pilot不放行，不能将H2直接写成已被否证。

## G4：进入新四臂pilot

只有G0/G1/G2/G3均满足、精确R04协议和独立阶段授权同时存在时，才允许新pilot。基座、mask、H、奖励及幅度不变；4个PPO臂与2个匹配监督控制同起点、同机会预算，每臂用自己的当前策略在线生成轨迹。没有任何权限可以由本目录PENDING提案、工程PASS或旧R03剩余额度推导。

pilot保留raw/H后J/L/K、完整请求/启动/IO、任务覆盖和独立验证、实际排序/ratio、guard接受与拒绝、Critic分量校准、资源。与历史native成本只能作固定输入质量参照，不能用旧计时冒充同期公平速度。完整同期native计时若需要，另行明确授权。

pilot的正向结论需按父布局比较完整成本且优于FrozenSL和各自匹配继续监督，并报告完整时间代价、失败与置信区间；不能以MSE、动作翻转、loss下降或guard未拒绝代替。单seedpilot只能提出机制证据，不足以证明普遍泛化或真实双臂IK。
