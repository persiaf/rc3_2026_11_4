# 持久化故障诊断

确认的历史直接原因是`os.replace(RESOURCE_ACCOUNTING.json.pending, RESOURCE_ACCOUNTING.json)`被Windows以WinError5拒绝；历史占用者没有被记录，不能确定就是observe进程。此次独立复现证明同样的普通Python读句柄足以触发该错误。

## 原事务缺陷

原[common.py](../rc3_relation_drl_20261113/common.py)的`put`先写固定名称pending并fsync，然后只尝试一次os.replace。`Ledger.complete`先从pending移除请求、增加completed、追加并fsync complete事件，最后调用save。调用者[runtime.py](../rc3_relation_drl_20261113/runtime.py)在`request_record`中已经成功保存请求JSON及其哈希。

当最后一步账本替换失败时，外层replay的except调用`Ledger.fail`。旧fail即使请求已从pending消失，也增加failed并追加failed事件。因此`init_0239_teacher_5`同时出现在complete和failed事件中。其请求记录和独立验证证书已经保存，但该次replay没有把lp返回给sl_update，第239次optimizer从未调用。旧账的1910与1是有重叠的事件计数，不是1911次请求。

准确原始记录见[FAILED_REQUEST_DETAIL.json](../rc3_relation_m2_20261113/FAILED_REQUEST_DETAIL.json)、[原日志](../rc3_relation_drl_20261113/science/relation_navcover_m2_v1_20261113/init/EVENTS.jsonl)及[工作进程错误](../rc3_relation_drl_20261113/science/relation_navcover_m2_v1_20261113/init/controls/attempt_0001/WORKER_STDOUT.log)。原日志不修剪、不改判。

## 文件占用证据及限度

旧[observe.py](../rc3_relation_m2_20261113/observe.py)直接对活动RESOURCE_ACCOUNTING.json调用Path.read_text，并读取STATE。只读不等于不会影响Windows重命名：FILE_SHARE_DELETE控制其他句柄的删除和重命名共享，限制持续到句柄关闭。[Microsoft CreateFileW文档](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew)

独立测试01启动已知PID的子进程，用普通`Path.open('r', encoding='utf8')`读取合成文件并保持句柄。主进程调用旧式单次os.replace得到WinError5，旧目标内容保持；释放读句柄后同一次文件替换成功。测试02用新有界重试，首次失败后释放子进程读句柄，第二次替换成功。实际PID、错误码、每次尝试和耗时保存在[最终原始测试结果](engineering_results/ATTEMPT_04.json)。保持句柄是用于确定复现时序的注入，不代表历史监控曾长时间持有句柄。

本次只读查询显示原目标文件属性为Archive而非ReadOnly，目录具有Administrators FullControl及Authenticated Users Modify权限。此前1909次及失败后的收尾写入成功，也不支持整个目录持续禁止写入的解释。这些是当前权限和历史行为证据，无法排除当时短暂ACL/杀毒/索引程序影响。测试05独立把合成目标设为只读，同样触发持续WinError5并有界停止，说明单凭错误码不能唯一归因。

没有历史文件访问跟踪，停止的进程无法被事后查询出当时句柄。此次可控子进程已提供实际占用者及释放前后对照，故未安装或启动广域文件监控；这不增加对历史占用者的确定性。

## 修复对应关系

| 故障点 | 新处理 | 独立证据 |
|---|---|---|
| 监控打开被替换的账本 | 只读唯一名称的已发布快照，不回退到活动账本 | 测试14、15、29 |
| 瞬时替换拒绝 | 同一pending字节最多6次替换，等待总和0.62秒 | 测试02—05、21 |
| complete已经持久化，快照失败 | 根据日志重建一次complete；单独记录persistence_failed | 测试06、08、28 |
| 记录写出但complete未写出 | 保持已扣账pending，显式核对记录后只补持久化 | 测试07、23、27 |
| optimizer已完成但STATE/提交事件未完成 | 不从optimizer计数推导游标；持久化边界核对后才提交 | 测试16—18、24—25、30、32、35 |
| 不确定追加或日志尾部撕裂 | 不重试append、不截断，停止并保留 | 测试10—12、27 |

原子替换成功不等于包含计算、记录、日志和快照的多文件事务已经全部提交；因此新实现明确分开这些阶段。[Python os.replace文档](https://docs.python.org/3/library/os.html#os.replace)
