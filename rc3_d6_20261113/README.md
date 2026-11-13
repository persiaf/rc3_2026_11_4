# RC3-D6长序列信用分配对照

从539ead5ff52cb110a91736d6550cb988171d4d9b建立的独立研究。复用D5的32768条保存轨迹做成本分解，随后从准确SL256重新训练A（原D5损失）、B（因果剩余成本）和匹配继续SL，各256更新。旧D54096负结果、教师、权重和用户文件保持。

[数学](MATHEMATICS.md) · [实施审查](IMPLEMENTATION_REVIEW.md) · [精确协议](PROTOCOL.json) · [共同日程](COMMON_SCHEDULE.json) · [来源](REGISTERED_INPUTS.json) · [资源账](RESOURCE_ACCOUNTING.json)

```powershell
& .\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_d6_20261113.run --run rc3-d6-causal-tail-20261113-r1
& .\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_d6_20261113.run --resume rc3-d6-causal-tail-20261113-r1
& .\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_d6_20261113.run --status
```

恢复不会清零任何账；已成功的phase跳过，失败收据保留并要求具体修复，不把失败标成PASS。固定128段结束保存完整检查点，256段实际重启恢复。若发生优化器已扣但无完整检查点，停止为技术未完成，禁止免费重做。

首个optimizer前固定：256是主终点；B对FrozenSL和匹配SL父等权H后J均改善≥0.5%，SEL与DEV均不退化且完整代理可行。128只固定报告；B对A另报，不能以A退化替代两个主比较。未通过结束本候选，不加学习率、宽度、更新或种子。当前66案例均已暴露，仅开发筛选。

## 已完成的实际交付

A/B/匹配SL各256完成，4888请求/768 Actor/2048标签重放，失败0。主终点B对FrozenSL、SL的父等权H-J收益−1.2305%、−1.5834%，对A+0.7985%，NO_DEVELOPMENT_SIGNAL，结束本配方。

[完整路线报告](EXPERIMENT_REPORT.md) · [数学与实现](MATHEMATICS.md) · [9个实际模型](MODEL_INDEX.json) · [792请求表](COMPLETE_TASK_RESULTS.csv) · [两聚合/分尺寸](AGGREGATE_RESULTS.csv) · [父配对](PARENT_PAIRED_GAINS.csv) · [停靠与顺序](STOP_SET_ORDER_CHANGES.csv) · [成本分解](COST_DECOMPOSITION_VALIDATION.json) · [逐动作复核](OLD_PER_ACTION_COST_AUDIT.json) · [旧规模损失/裁剪](D5_TRAJECTORY_CREDIT.csv) · [新在线参考诊断](ONLINE_CREDIT_DIAGNOSTICS.csv) · [数值复核](INDEPENDENT_NUMERICAL_AUDIT.json) · [研究决定](NEXT_STEP_DECISION.md) · [结果到主张](RESULT_TO_CLAIM.md) · [给老师的短材料](TEACHER_DECISION_BRIEF.md)。

首算前协议/源码提交c0e3cd08897615a4137c9efeecc2c5860fd104fe。正式A/B/SL只读同一个COMMON_SCHEDULE，当前A前256的模型和损失与旧D5前缀精确一致，但实际进行了新的采样和更新。没有借用旧D5轨迹训练。当前候选不提升为生产默认。

## 远端发布核验

实际payload提交e04bc30c837e36bb33caa898fbbd9139fb0ac836，新增27个LFS对象、149973365字节已从远端下载并逐一核对SHA256和大小。详见[发布收据](PUBLICATION_RECEIPT.json)、[远端LFS逐对象核对](LFS_REMOTE_READBACK.json)、[冻结文件保持](PRESERVATION_CHECK.json)与[最终墙钟结账](FINAL_WALL_ACCOUNTING.json)。最终元数据提交不改变已核对的模型和轨迹。
