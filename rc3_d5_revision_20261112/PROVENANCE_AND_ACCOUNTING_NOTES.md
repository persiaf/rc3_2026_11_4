# 来源与计账口径

本修订基准是`20478a4dde2888f86abb9526c9d39ab71a27ee52`，独立run ID为`d5-data-revision-20261112-r1`。本次实际请求、数据和授权不能与旧v1混同。

## 四层身份

1. 原准确SL256是`rc3_integrated_rppo/runs/rc3_integrated_rppo_v1_20261110_01/actors/11622026/SL256_0256_1817f58aa650.pt`。文件SHA256为`60f5aa4e59983e92d565f0599823b710965cf2c96a041e7eb483d23a8eb0556c`，tensor SHA256为`1817f58aa6506456ff12f0c5fe0a914e156aef8af319f4687dde90782f55a63e`。其历史路径包含RPPO字样，但本次只读取已验明的base-only SL256 Actor，不加载旧RL或诊断权重。
2. `20bf35f40`提交在新教师调用前固定了96个完整输入身份、共同日程、当前数据生产源码与预算。`DATA_SOURCE_BINDING.json`复用了原来源包的结构，其`base_commit/run_id/protocol_sha256`保留旧源集语境；`revision_run_id`、新源码SHA、`REVISION_PROTOCOL.json`和本次`AUTHORIZATION.json`才标识本次数据生产。旧字段不是把本次操作授权为旧v1，也不是本修订Git基准；原文件不回写。
3. 科学身份`441f2281d59145476f1ee957ba0a742ede8ef8cacb213ae016191b5631df6df1`在实际优化前固定。原Actor、Trainer、环境、H、数据、日程和门槛均有hash；正式流独立从上述SL起点建立，工程权重禁止进入。
4. 当前执行源码由`ACTIVE_EXECUTION_SOURCE.json`指向R04B。R02只修复用途与阶段计数器重名，并复用已完成的工程事务；R03修复Windows原子文件替换及异常收据写入；R04B将总墙钟预算与SL-C活跃计算匹配分开，并修复Python3.10对系统事件100ns时间戳的兼容。保留原科学身份，训练数学与固定终点不变。原失败源码、资源快照、740/1771检查点和恢复登记均单独留存，未退款。

## 可直接核算的新增消费

`EVENTS.jsonl`在工作前写入request/optimizer尝试，成功记录含精确压缩归档位置；失败属于分母。teacher/reference/train/eval是用途，engineering/training等是阶段，两套计数不能相加，否则会重复计算工程期间首次生成的12条参考。

新增`encode_calls/encode_s/score_calls/score_s`覆盖本修订实际Actor调用，含采样、反向重算和评价；缓存命中、重建、输入准备和完整请求时间另记。`original_forward_calls/feature_states/feature_s/request_io_s`是继承字段，不能把未更新的旧值描述为本次完整新计时，也不能用相减得到未测延迟。

研究归档的实际压缩字节和记录位置保存在逐请求指针中。当前开发`full_request_s`包含压缩及归档写入，截止于后续pointer/event/resource日志写入之前；所有日志尾部成本仍在监督作业墙钟内。该口径不是纯网络前向时间。开发FrozenSL、原生规则和训练终点评价的输入冷热状态不完全相同，不能据此宣称部署加速。后续部署阶段若获准，才使用登记同期冷热网格；未执行则明确NOT_RUN。

内部搜索细账实际记录construct、two_opt、joint_search、recipe四类被监测函数的次数和inclusive/exclusive时间，嵌套时间不能直接相加。原生接口返回的其他细节（例如未保存的DP细分调用）无法从归档精确恢复，应标UNKNOWN；native调用次数及完整原生请求墙钟仍有实际证据，不把UNKNOWN填0，也不为补日志重跑教师。

## 实际中断与维护

- 工程首次准入因计数器重名误报61；日志证明实际累计44请求、7次Actor尝试。核正仅撤销重复字段，29条新请求和2次实际更新全部保留，后续最终工程消费为59请求、10次Actor尝试。
- P1完成740次更新后，第741批slot3的请求已扣账，但原子资源账替换遭WinError5，模型/H尚未运行。该请求保留为失败。恢复保留740模型/Adam/RNG/游标及前三条同Actor轨迹，只补缺失槽位；实际重试同一问题另扣请求。
- 系统事件确认睡眠2363.847163秒，原累计墙钟已完整扣账。1771后登记维护停止，1772批8条己方完整轨迹均已保存、优化器尚未尝试。R04B用Windows活跃墙钟计SL-C匹配；预算仍使用含睡眠的原总墙钟。兼容探针第一次因7位小数解析失败、第二次通过，累计完整CLI耗时1.8488335秒，零前向/优化/规划。维护时未持久化的部分反向/前向细计数标UNKNOWN，真实优化器次数与墙钟保留。

工程实现的墙钟另以`ENGINEERING_WALL_SUPPLEMENT.json`和`R03_RESOURCE_RECONCILIATION.json`补入原6h池。最早可证明的新源码时间之前的零散读取未测，不假称为0；已记录并行工程和科学工作按原计划加总计费。P1流内训练时间不加入上传或等待，原有数据、参考首次费用和真实中断消费均保留。

这些口径限制不改变原质量门槛，也不把任何技术未完成、未运行阶段或模型约束验证写成新父布局泛化、部署速度或真实机械臂关节轨迹认证。
