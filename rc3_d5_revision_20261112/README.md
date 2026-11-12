# D5完整教师数据修订与科学执行

本目录是从`20478a4dde2888f86abb9526c9d39ab71a27ee52`建立的独立执行版本，run ID为`d5-data-revision-20261112-r1`。旧`rc3_decoder_program_20261111`的技术停止和P1—P5 NOT_RUN原字节保留。本目录的状态不能回填为旧版本已经执行。

当前数据阶段已完成：96条新增完整10×10强教师覆盖96个训练物理父布局，旧256条局部教师文件不变；352个问题均通过当前Environment的完整动作重放。三seed各有4096×8共同日程，每批4完整、4局部，同seed的D5-PG、Full-PG和D5-SL读取同一日程。**P1已完成：PG4096、SL-U4096；原固定终点门槛未通过，NO_DEVELOPMENT_SIGNAL→P6。** H后J父配对收益对FrozenSL为−2.6431%、对匹配SL-U为−3.7371%，SEL8/DEV3均负。P2—P5均NOT_RUN，本身份已封闭。

[完整报告](EXPERIMENT_REPORT.md) · [给老师的短材料](TEACHER_DECISION_BRIEF.md) · [结果到主张](RESULT_TO_CLAIM.md) · [阶段状态](STAGE_RESULTS.json) · [模型索引](MODEL_INDEX.json) · [逐请求表](tables/COMPLETE_TASK_RESULTS.csv) · [资源账](RESOURCE_ACCOUNTING.json) · [独立复核](INDEPENDENT_NUMERICAL_AUDIT.json)

## 来源与参数

- [登记协议](REVISION_PROTOCOL.json)、[96问题身份](FULL_TASK_REGISTRATION.json)、[完整教师来源](FULL_TEACHER_MANIFEST.json)、[联合索引](JOINT_DATA_INDEX.json)、[共同日程和使用次数](SCHEDULE_EXPOSURE.json)。教师为原Detour16强教师加固定公共H；PG不读取教师动作作为提示。
- D5只更新`key.weight`、`dynamic.weight`、`nav.weight`、`query.weight`、`query.bias`，25152/65088个标量。表示和prefix GRU冻结；使用原Actor输入、评分和mask，不使用detached206残差入口。
- 正式流从准确SL256独立克隆。Adam学习率1e-4、裁剪1、每批8条、完整gamma=1回报、熵权重0；没有A1幅度/永久SL guard或Critic。五张量PG损失为`mean(detach(J_sample_H-J_FrozenSL_same_problem_H) * sum(logpi))`，同问题FrozenSL参考永久冻结。
- [原始科学身份](SCIENTIFIC_IDENTITY.json)与[当前执行源码](ACTIVE_EXECUTION_SOURCE.json)分开保存。R02修复用途/阶段计数器重名，R03修复Windows原子文件替换，R04B分开累计墙钟和SL-C活跃计算匹配时钟并兼容系统事件时间戳。模型、更新公式、数据、日程和质量门槛不变。原失败、错误计数原件和[无退款核正收据](COUNTER_CORRECTION_RECEIPT.json)保留；系统睡眠不从总账或8h流预算退款。

## 实际命令与恢复

在仓库根目录的PowerShell使用已登记Python环境：

```powershell
& .\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_d5_revision_20261112.run --status
& .\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_d5_revision_20261112.run --continue-authorized-plan d5-data-revision-20261112-r1
& .\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_d5_revision_20261112.run --resume d5-data-revision-20261112-r1
```

第二条用于已授权执行的首次进入；第三条用于原身份中断后的继续，不能与活动作业同时启动。控制器检查监督器退出收据、原授权、源码、数据、参考及准入，不靠手改PASS推进。已经封闭的执行身份只返回封闭状态。

`streams/<seed>/<role>/LATEST.json`原子绑定交替checkpoint。checkpoint含模型、Adam、全局RNG、独立采样Generator、游标、已用轨迹和查询/资源快照。未提交批次只复用参数hash相同的己方完整在线轨迹；真实额外优化尝试和中断时间不退款。工程模型绝不作为正式初始化。

## 条件阶段与证据

P1第一seed分别完成D5-PG4096、D5-SL-U4096，固定0/1024/2048/4096导出及66个已暴露开发问题评价。step0质量由同一SL256同期基线复用。主判断只用4096；父布局等权H后J相对FrozenSL改善至少1%、相对SL-U至少0.5%，SEL与DEV均不退化，完整约束验证全部通过。未达到4096的技术中断不解释为科学失败。

只有原门槛通过才进入Full-PG/时间匹配SL-C、另外两seed、部署计时及新父布局确认/最终评价；详见冻结原主计划和登记协议。不读原60、旧S3、关闭支线或提前解封确认/最终集。不同方法本轮开发评价的输入缓存冷热不同，其实际时间只作测量记录，部署速度结论须由后续同期冷热控制计时支持。

原总上限96h、300000请求不变。教师子账最多3600秒、104 native及104统一H，其中额外8对只允许同输入技术恢复。工程原15请求/5次Actor尝试及失败完整继承。新增请求用途按teacher/reference/train/eval分开；完整轨迹、服务分配、raw/H J/L/K、验证、计时、特征和I/O见逐请求归档及资源账。

保存输出的独立数值核算在活动科学作业结束后运行，不新增模型前向、训练或规划：

```powershell
& .\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_d5_revision_20261112.audit_saved_results
```

它只核对本次新增产物，不重做历史大CSV审计。Git/LFS发布收据在本轮交付时绑定实际远端SHA与新增对象，不把缓存清理作为科学条件。

## 发布验收

本轮数据/模型结果提交为`a7b471bf62768cd9eadc1e377f2c9a942e280c7c`，科学运行源码提交为`f31cf3a71d2d28b6a5fe9b2c8f5fde7bc97afe09`。已从远端向全新独立缓存下载并逐一核对542个本轮新增LFS对象、1,153,742,108字节，含3个仅存在于中间提交的版本；当前树539个。主分支未合并，旧v1/关闭支线及用户文件未提交改动。

[发布收据](PUBLICATION_RECEIPT.json) · [逐对象远端核对](LFS_REMOTE_READBACK.json) · [新增对象清单](LFS_NEW_OBJECT_INVENTORY.json) · [原字节/提交范围](PRESERVATION_CHECK.json) · [缓存维护记录](CACHE_MAINTENANCE_RECORD.json) · [最终资源结账](DELIVERY_ACCOUNTING_FINAL.json)。最后的发布元数据提交沿用同一组模型/LFS对象；其远端SHA在本轮最终答复核对。

总墙钟账含科学实耗、全部中断和睡眠、后处理完整日历窗口及并行独立审查窗口。最终120秒元数据收尾按保守上界计费，明确区别于已实测运行；最早未测读取和未持久化细项继续UNKNOWN。未用缓存删除换取准入。
