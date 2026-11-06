# 新执行源码与原教师来源的阶段衔接

本轮已实现生产衔接，未执行S1-B。新协议批准/阶段授权收据不存在时，生产入口在任何模型初始化或运行写入前拒绝。原S1-A授权不能承担S1-B授权；原v3协议字节、435b源码绑定、教师池和RUN_MANIFEST均不改写。

## 执行来源与数据来源

`--source-binding`显式选择新的不可变`SOURCE_BINDING_UPDATED.json`。准确当前源码、registry、H、运行环境和新的零科学验收均须匹配；新用户源码批准收据和S1-B阶段授权均须绑定新manifest的SHA。省略该参数仍使用原绑定，改变后的源码会被原绑定拒绝，不会静默升级。

新manifest中的`legacy_teacher_replay`包含清单路径/SHA和完整旧science_binding。生产准入核对清单中的原实验目录、S1-A COMPLETE、POOL_INDEX、RUN_MANIFEST、原源码manifest、原真实用户批准及各自精确SHA。StageChain只有S1-A前驱允许匹配该明确旧binding，其余S1-B/S2必须匹配新执行binding。原完成记录及512次查询资源不重写、不重置、不重复消费。

TeacherPool和阶段上游检查核对原池路径、索引SHA、256标签SHA、完整路线/端点/任务及终局成本；SceneProvider按已冻结清单重放各输入。监督阶段provider强制read_only，缺失输入不能触发生成。新权重、Adam、RNG、恢复游标和日志仍绑定新代码，不把旧教师源码伪装成新训练源码。

首次获准执行新S1-B时，在原实验目录新增`SOURCE_REVISION_<new_source_manifest_sha>.json`，保留原RUN_MANIFEST。该记录跨三个seed和后续独立授权阶段保持稳定；后续S2应复用同一新源码批准，仅另建阶段授权。改源码或再次改科学身份不能绕过已完成/中断阶段的链路检查。

## 固定S1-B入口

未来收到准确新源码批准和S1-B执行授权后，使用真实收据路径运行；下列占位符不构成许可：

```powershell
.\rc3_learning\.venv\Scripts\python.exe -B -m rc3_global_local_execution_v3.runner --stage S1-B --seed 11622026 --experiment-id rc3_v3_global_local_20261106_01 --output rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01 --existing-run --source-binding rc3_navigation_compat_20261106/SOURCE_BINDING_UPDATED.json --protocol-approval <真实新源码批准收据> --stage-authorization <真实S1-B阶段授权收据>
```

第一seed完整结束后，同入口依次使用11622027、11622028；前序seed未完成会拒绝后序。授权可一次明确覆盖三个固定seed，授权前驱`S1-A`固定为原COMPLETE文件SHA。每seed256次SL优化器更新、batch4、累计1800秒，新教师/native/H为0。检查点每32更新保存，完整交付包含untrained与update_0256模型、Adam、RNG、日志、integrity sidecar及关闭记录。它们不是当前已生成的模型。

安全边界中断后，仅同一授权、相同seed/阶段/来源及精确checkpoint可用`--resume`继续；既有dirty日志、FAILED、已完成stage或授权变更拒绝重启，累计墙钟含之前已用资源及当前恢复IO。生产恢复继承现有严格Tensor/Adam/RNG验证，不在本轮创建模型验证夹具。

## 已执行的零科学检查

`bridge_regression.py`检查真实旧池的阶段门槛、TeacherPool.validate和S1-A完成链，校验新旧来源分离；错协议/registry/H/实验/source/清单SHA/旧批准、错误seed、缺失许可和借用旧S1-A许可均拒绝。隔离纯字节RUN_MANIFEST夹具仅验证构造器保持旧文件和新checkpoint配置来源，stage scope为空，无正式执行许可。

模型初始化/前向、backward、Adam.step、scene生成、nav求解、native和H均由sentinel封锁。机器回执单独记录当前source map、checker SHA、断言、负向拒绝和墙钟；后续源码变化必须实际重跑并另存回执，不能手改旧回执source map。
