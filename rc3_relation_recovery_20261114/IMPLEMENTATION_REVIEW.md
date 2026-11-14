# 实施审查：ENGINEERING_PASS

本结论覆盖持久化修复及独立工程验收。原M2科学状态仍为TECHNICAL_INCOMPLETE，恢复协议是待批准提案。

新代码保存在本目录，原common/runtime/observe、M2协议、停止决定、历史教师和权重均不修改。[HISTORICAL_BINDING.json](HISTORICAL_BINDING.json)登记原证据字节；[NEW_SOURCE_BINDING.json](NEW_SOURCE_BINDING.json)绑定原98项及新增执行组件。

## 事务状态

| 边界 | 已确定事实 | 重启后允许的处理 |
|---|---|---|
| begin日志fsync | 已消费一次请求/optimizer尝试 | 保留扣账；没有完成记录就停止 |
| 请求记录写出 | 结果字节存在，尚无完成事务 | 显式核对请求身份和SHA；可只补记录事务，不能据此复用计算图 |
| complete日志fsync | 该请求/optimizer终态唯一 | 从日志派生快照，不能再追加failed终态 |
| checkpoint字节、STATE写出 | 存在新模型候选边界 | 仍需checkpoint_commit日志；不能仅凭较大STATE自动推进 |
| checkpoint_commit日志fsync | 游标与完整checkpoint身份一致 | 快照缺失时可从日志重建该游标 |

[persistence.py](persistence.py)采用单写者OS字节锁、序列号和哈希链日志。快照错误不会撤销已完成事件。每个请求ID只允许一个begin和一个计算终态；同一complete补发只更新快照。重试仅位于os.replace，不重跑请求、不调用optimizer、不重追加日志。持续权限错误有界停止，旧目标不删除，pending保留诊断。日志append的fsync若结果不确定则停止，不自动重试或修剪尾部。

[observe.py](observe.py)只打开published目录中的不可变status文件。发布采用唯一文件名，正在被读取的旧快照不再被替换或清理。监控可能显示旧进度，不能用其授权恢复。日志、STATE和checkpoint是恢复依据。

[runtime.py](runtime.py)是明确版本化的原运行时副本。AST对照确认优化器、apply_update、replay、随机设置及SL损失与反向次序保持原样；sl_update仅给回放请求ID加`recovery_v1_`前缀。另两处修改为request_record持久化边界和checkpoint文件替换重试。模型仍导入原RelationActor。[对照清单](RUNTIME_DIFF_REVIEW.json)及测试34保留证据。

[recovery_adapter.py](recovery_adapter.py)准确加载原SL238模型/Adam/RNG，同时保留1910次历史消费和全部原ID，不调用旧restore的“1904等于1910”检查来强行跳过。新适配器只接受明确登记的6个额外历史请求差异。[recovery_runner.py](recovery_runner.py)要求独立新协议批准和M2a恢复阶段授权，使用原进程树监督器及固定累计剩余墙钟；本轮只测试了拒绝未授权入口，未运行科学分支。

新入口没有M2b路径。恢复目录固定为原init下面的recovery_v1_20261114，并以原账为genesis来源；目录存在就拒绝第二次启动。任何再次中断均停止，不自动使用已消费额度重跑。

## 验收范围

最终36项独立测试通过，包括真实Windows读句柄冲突、真实只读属性拒绝、注入式事务中断、双写者拒绝、记录损坏、checkpoint损坏和双槽覆盖。没有模型前向、真实optimizer、正式教师回放或新父任务。工程测试证明这些持久化边界可处理；没有宣称已实际恢复科学模型并完成SL512，也没有DRL性能证据。
