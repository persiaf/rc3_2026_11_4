# M2正式科学实验：TECHNICAL_INCOMPLETE

运行锚点日期2026-11-13，结束于北京时间2026-11-14 00:01。科学ID：`relation_navcover_m2_v1_20261113`。批准基准：`9c878928e7f619bce25ddd00792757431ca705a1`；准确协议SHA256：`7ca20d48177b156ef4183f8d58f8354f20198513ecad66e161ff4d0b5e41a445`。

**结论为TECHNICAL_INCOMPLETE。** M2a完成238次科学SL更新，在第239批第6个教师回放的资源账保存处异常退出。没有SL512完成收据，M2b未获创建条件，未启动。此结果不能判定关系DRL有效或无效，也不是2048质量门槛的NO_DEVELOPMENT_SIGNAL。

## 1 实际执行和交付覆盖

|要求|实际结果|
|---|---|
|SL512初始化及来源|仅完成SL238；保存准确部分checkpoint、模型/Adam/RNG和来源，SL512标NOT_COMPLETED|
|DRL2048与继续SL2048|均NOT_RUN，更新0，未创建M2b阶段授权|
|0/512/2048固定评价|全部NOT_RUN；评价请求0|
|六方法完整对照|全部NOT_RUN；缺失成本与时间为空，不填0|
|LOCAL66与FULL12|分别保留固定清单，均0题评价，不混合统计|
|原始数据、图表、资源和停止决定|1910份教师回放、238份更新、原账/异常/监督记录、部分checkpoint、CSV及F1/F4实际图表已保留|

|方法|LOCAL66，端点0/512/2048|FULL12，端点0/512/2048|
|---|---|---|
|FrozenSL|NOT_RUN / NOT_RUN / NOT_RUN|NOT_RUN / NOT_RUN / NOT_RUN|
|RelationInit|NOT_RUN / NOT_RUN / NOT_RUN|NOT_RUN / NOT_RUN / NOT_RUN|
|RelationSL|NOT_RUN / NOT_RUN / NOT_RUN|NOT_RUN / NOT_RUN / NOT_RUN|
|RelationRL|NOT_RUN / NOT_RUN / NOT_RUN|NOT_RUN / NOT_RUN / NOT_RUN|
|Fixed3|NOT_RUN / NOT_RUN / NOT_RUN|NOT_RUN / NOT_RUN / NOT_RUN|
|Detour16|NOT_RUN / NOT_RUN / NOT_RUN|NOT_RUN / NOT_RUN / NOT_RUN|

## 2 故障证据

[worker原日志](../rc3_relation_drl_20261113/science/relation_navcover_m2_v1_20261113/init/controls/attempt_0001/WORKER_STDOUT.log)显示调用链：`sl_update → replay → request_record → Ledger.complete → Ledger.save → put → os.replace`。Windows对`RESOURCE_ACCOUNTING.json.pending → RESOURCE_ACCOUNTING.json`返回`PermissionError: [WinError 5] 拒绝访问`。

这是资源账持久化失败，未见教师非法、来源不匹配、NaN、CUDA OOM或预算超时的异常证据。[监督收据](../rc3_relation_drl_20261113/science/relation_navcover_m2_v1_20261113/init/controls/attempt_0001/SUPERVISOR_RECEIPT.json)为`WORKER_FAILED`，worker exit=1，timeout=false，未请求强制终止，最终活动进程数0。原资源状态`SUPERVISOR_STOPPED`表示监督层记录了异常结束，不能解释为监督器主动杀进程。

已确认的直接原因是原子替换被系统拒绝；具体占用者或权限条件未取证。本轮[监控脚本](observe.py)以普通文件读取同一账本，可能与替换发生短暂冲突，应列入候选原因，不能排除其他文件访问或权限因素，也不能将其确认为唯一根因。Windows共享删除权限涵盖重命名；这一机制说明了读句柄冲突为何值得排查，但不是本次锁持有者的证据。参见[Microsoft CreateFileW](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew)及[MoveFileExW](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-movefileexw)。

本轮没有修改冻结的持久化源码，没有重试科学阶段或执行新的故障复现实验。

## 3 原始计数与失败重叠

|项目|实际值|口径|
|---|---:|---|
|SL optimizer尝试/完成|238 / 238|239未调用optimizer|
|唯一教师请求尝试|1910|预算消费仍为1910，不是1911|
|原账request_completed|1910|原始complete事件计数，保留原值|
|原账request_failed|1|同一请求同时出现在complete与failed事件中|
|派生的非失败请求|1909|从complete ID集合排除failed ID，未回写原账|
|保存的教师记录与valid证书|1910|全部记录hash匹配；包括落盘失败前已保存的第1910份回放记录|
|完整请求时间记录|1909|失败请求未写出正常full_request_time，不用局部时间冒充|
|encode / score|1910 / 112478|正式SL教师强制回放中的真实模型调用|
|backward|1909|最后请求在返回log概率前报错，尚未执行其backward|
|新教师 / 新H / 原生求解 / DRL在线轨迹|0 / 0 / 0 / 0|本轮仅SL初始化|
|已访问登记训练ID|352|重复回放不算新增场景|

失败ID为`init_0239_teacher_5`。其请求JSON、teacher validation与complete事件已保存，随后账本原子替换失败，异常处理追加failed事件。因此不能用1910与1计算互斥成功率，不能通过减少completed或退款来“修好”历史账。精确事件和来源见[失败请求详情](FAILED_REQUEST_DETAIL.json)、[教师记录CSV](analysis/TEACHER_REPLAYS.csv)和[原始EVENTS](../rc3_relation_drl_20261113/science/relation_navcover_m2_v1_20261113/init/EVENTS.jsonl)。

## 4 部分模型和恢复边界

最后原子提交的[STATE](../rc3_relation_drl_20261113/science/relation_navcover_m2_v1_20261113/init/STATE.json)游标238，覆盖1904教师请求及238次optimizer。该checkpoint的模型SHA为`e4002cd07a31430d76ce391a7f5915e4a127093bccc2a94a5194a6b24462a3d9`；文件SHA为`e6b7672a3184b9e65e90805c8773fcbdd473df0f0b837967f040e41c6f721c9e`。所有Adam参数step=238，模型/Adam/RNG哈希与sidecar一致，见[部分状态审计](M2A_PARTIAL_AUDIT.json)。

完整权重和优化器保存在[current_slot0.pt](../rc3_relation_drl_20261113/science/relation_navcover_m2_v1_20261113/init/current_slot0.pt)，前一槽位237也保留。它是科学SL238部分状态，不能称为SL512或供本轮M2b使用。

checkpoint之后`init_0239_teacher_0`至`_5`六个请求已扣账。即使最终pending为空，当前账本ID集合仍比checkpoint多六个，冻结恢复条件不满足；不能只加载238并免费重放这六个请求。未执行恢复，未创建M2b授权。

## 5 时间、图表和证据边界

原资源账累计外层墙钟为**2221.312秒（37.0219分钟）**，低于14400秒上限。监督收据本身记录2221.288313秒；外层账包含最后核算尾部，二者不相加。worker最后failed事件的账本时间为2194.294541秒，其时间原点与外层不同，不能把两数之差直接解释为收尾或未计耗时。

[F1训练曲线](figures/fig1_training.pdf)与[F4资源曲线](figures/fig4_resources.pdf)只反映已发生的SL拟合及事件消费。损失下降不证明规划改善。无完整评价数据，所以F2收益图与F3六方法性能图不生成；[方法覆盖表](analysis/METHOD_TABLE.csv)保留缺口。数据来源和逐图脚本见[图表说明](figures/README.md)。

98项已批准源码/数据/模型绑定在停止后仍匹配；协议和训练身份未变。用户论文、原60、旧S3及关闭支线保持原状态。下一步见[停止决定](NEXT_STEP_DECISION.md)，本轮到此结束。
