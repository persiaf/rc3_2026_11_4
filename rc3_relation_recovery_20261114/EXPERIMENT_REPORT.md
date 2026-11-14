# M2a持久化工程报告

**结论：ENGINEERING_PASS；原科学运行仍为TECHNICAL_INCOMPLETE。**

本轮只运行独立临时目录中的合成事务。第一次测试发现冲突complete输入未统一抛出预期ValueError，改为先核对幂等事件身份后检查实际记录；错误轮次保留。后续在接口接线、双层异常处理、双槽和拒绝未授权入口补齐后，最终36项全部通过。

| 轮次 | 测试数 | 失败/错误 | 结果 | 合成begin方法调用 |
|---|---:|---:|---|---:|
| 01 | 27 | 0/1 | FAIL保留 | 62 |
| 02 | 27 | 0/0 | PASS | 62 |
| 03 | 32 | 0/0 | PASS | 65 |
| 04 | 36 | 0/0 | PASS | 69 |

累计4轮、122项测试执行、258次合成begin方法调用（包含预期被拒绝的尝试）。这些是内存/JSON计数器操作，其中名为optimizer的事件没有调用任何真实优化器。已用完登记的4轮suite预算，本轮不继续扩展测试。

各轮开始前均保存源码哈希并先追加[SUITE_DEBITS.jsonl](engineering_results/SUITE_DEBITS.jsonl)。每轮详细结果、进程输出和准确测试源码绑定位于[engineering_results](engineering_results)。临时目录限定在本目录`.engineering_tmp`，用例结束后仅删除本用例新建子目录；未在正式账本上复现。合成结果保留结构化证据，未保存无用临时副本。

最终关键观察：普通读句柄持有时单次replace得到WinError5，释放后成功；新实现真实瞬时冲突第二次尝试成功；持续拒绝最多6次、不删除旧目标；complete后的快照失败重建为1次消费、1次complete、0次计算failed；optimizer完成但checkpoint事务未提交时游标保持原值。独立发布快照被持续读取时40个合成请求正常完成。历史具体占用者仍未知。

只读审计核对了SL238文件、模型张量、Adam和RNG哈希；Adam各状态step均为238。检查点包含1904次教师消费和238次optimizer，原始日志另有第239批6次消费。1910份请求字节及保存的验证证书、全部对应教师来源哈希均核对通过，未重新计算验证或回放教师。详见[SL238_RECOVERY_AUDIT.json](SL238_RECOVERY_AUDIT.json)。

原第239批前5次回放已经反向传播，第6次记录写出后持久化失败，未将lp返回。第239批没有optimizer调用，临时梯度和计算图未进入checkpoint，不能从请求JSON恢复它们。受控方案必须重新完整计算8个任务，并在新ID下收费。

实际工程墙钟、只读审计工具墙钟及科学零调用分列在[RESOURCE_ACCOUNTING.json](RESOURCE_ACCOUNTING.json)；它们不被算成新增科学训练消费。原科学37.02分钟及1910请求保留在恢复提案累计起点。
