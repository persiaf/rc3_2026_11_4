# M2a SL238受控恢复实际报告

**结果：COMPLETE_RECOVERED。** 本轮只执行一次获批M2a恢复，M2b和DRL未运行。

| 资源 | 历史 | 本轮新增 | 累计 | 上限 |
|---|---:|---:|---:|---:|
| 教师请求 | 1910 | 2192 | 4102 | 4102 |
| optimizer尝试 | 238 | 274 | 512 | 512 |
| 完成更新 | 238 | 274 | 512 | 512 |
| 墙钟（秒） | 2221.312 | 2552.594000 | 4773.906000 | 14400 |

本轮沿用获批提案规范化SHA256 `40a01bf510b3addf474f428708b6c8ee93cf9f53ef30f1ba444afd8f5b0b380c`，执行源码保持106项绑定。准入独立核对SL238模型/Adam/RNG、原账、2148个原请求/optimizer ID及完整剩余日程；授权在启动前提交并核对远端。

原始第239批6次消费保留。恢复从第239批完整8次计算重新开始，所有新回放使用recovery_v1前缀，不复用旧计算图、不抵扣旧账。每次回放保留教师来源、动作及合法性证书；每次更新保留梯度、模型前后哈希和版本。监控仅读取published不可变快照，原账本未被监控并发打开。

最终游标为512。精确日程、请求唯一性、模型链、checkpoint、Adam/RNG、资源和历史原字节检查详见[CLOSED_RUN_AUDIT.json](CLOSED_RUN_AUDIT.json)。原始TECHNICAL_INCOMPLETE及complete/failed重叠事件保留；历史raw request_failed=1不可解释为本轮新增失败。

完整请求耗时在[TEACHER_REQUESTS.csv](analysis/TEACHER_REQUESTS.csv)，实际SL损失和梯度在[TRAINING_UPDATES.csv](analysis/TRAINING_UPDATES.csv)。耗时字段覆盖计算、请求记录和账本快照，后续时间事件及发布尾部由阶段墙钟计入；完整外层监督时间包含冷启动与收尾。GPU峰值未专门记录，不借用M1峰值。

原始运行目录：[recovery_v1_20261114](../rc3_relation_drl_20261113/science/relation_navcover_m2_v1_20261113/init/recovery_v1_20261114)。原科学SL512 COMPLETE仍不存在；恢复完成凭据采用独立RECOVERY_COMPLETE，不冒充无中断的原初始化收据。

本轮SL初始化结果不证明DRL收益；没有运行LOCAL66/FULL12性能评价、M2b对照、额外种子或新父测试。

监督收据确认工作进程exit_code=0、未超时。观察根进程退出时，作业内仍报1个活动进程，监督器按root_exited_descendants_remain执行该作业收尾清理，随后确认活动进程归零；status=COMPLETE。这不是超时终止训练。原始监督事件全部保留。

本轮没有新增计算failed或persistence_failed终态；底层瞬时replace重试次数未单独记录，因此不把“无新增失败”扩张为“从未发生瞬时文件冲突”。恢复的前6个位置按额外技术重算消费，累计backward_calls=4101包含历史5次未提交的反向计算；实际参与512次已提交更新的样本位置为4096。
