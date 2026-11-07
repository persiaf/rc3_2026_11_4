# S2独立授权后的准确执行方案

**技术GO；当前未授权、未执行。** 原批准v3/0f0ffd2源码保持不变。可一次明确授权三个固定seed，随后按11622026→11622027→11622028执行，无需逐seed重复申请。

新S2阶段收据必须绑定现有独立源码批准收据、协议、Registry、H、实验ID和下面四份已存在上游完成哈希；完整参数和精确端点见[S2_EXECUTION_PLAN.json](S2_EXECUTION_PLAN.json)。

| 上游 | COMPLETE SHA256 |
|---|---|
| S1-A | `595716b2844223613c43477d6d26788df89ccbbfe49db0b32655634c4d726228` |
| S1-B/11622026 | `5c86800ae865c9d37143e3aa25261f1a4f12f71f6c337d8adaa9171e5232f167` |
| S1-B/11622027 | `1d9ffb24d732910a8aa5bd7db3d1d415da477a85f19e349081b2cdc0a02359f0` |
| S1-B/11622028 | `3fb6b4bb5d1a85a78f2e051d6ed2d78fe62f0b08ca3f8ce71de3c41c1fd3b7aa` |

每seed从自己的SL256分叉RL与continuedSL，fresh Adam；冻结SL及原untrained保留。每seed512RL+512continuedSL、batch4、终局调用最多8192、阶段累计7200秒；三seed最多3072更新、24576终局调用、21600阶段秒。prepare/input/cache/H/development/验证/IO均在阶段账中，不能另开免费预处理。

只在用户新增明确S2授权、真实收据已保存并核验后，将以下占位文件替换为真实授权路径运行；本轮没有执行这些命令。

```powershell
.\rc3_learning\.venv\Scripts\python.exe -B -m rc3_global_local_execution_v3.runner --stage S2 --seed 11622026 --experiment-id rc3_v3_global_local_20261106_01 --output rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01 --existing-run --source-binding rc3_navigation_compat_20261106/SOURCE_BINDING_UPDATED.json --protocol-approval rc3_global_local_execution_v3/authorizations/20261106_S1_B/SOURCE_REVISION_APPROVAL.json --stage-authorization <NEW_EXPLICITLY_APPROVED_S2_STAGE_AUTHORIZATION.json>
```

```powershell
.\rc3_learning\.venv\Scripts\python.exe -B -m rc3_global_local_execution_v3.runner --stage S2 --seed 11622027 --experiment-id rc3_v3_global_local_20261106_01 --output rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01 --existing-run --source-binding rc3_navigation_compat_20261106/SOURCE_BINDING_UPDATED.json --protocol-approval rc3_global_local_execution_v3/authorizations/20261106_S1_B/SOURCE_REVISION_APPROVAL.json --stage-authorization <NEW_EXPLICITLY_APPROVED_S2_STAGE_AUTHORIZATION.json>
```

```powershell
.\rc3_learning\.venv\Scripts\python.exe -B -m rc3_global_local_execution_v3.runner --stage S2 --seed 11622028 --experiment-id rc3_v3_global_local_20261106_01 --output rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01 --existing-run --source-binding rc3_navigation_compat_20261106/SOURCE_BINDING_UPDATED.json --protocol-approval rc3_global_local_execution_v3/authorizations/20261106_S1_B/SOURCE_REVISION_APPROVAL.json --stage-authorization <NEW_EXPLICITLY_APPROVED_S2_STAGE_AUTHORIZATION.json>
```

源码哈希、上游或checkpoint不匹配/非有限/不可行/超限即保存已花资源并停，不换seed、不补训练或重置账。中断仅按同授权安全checkpoint恢复；不选择开发最佳权重，不执行S3或原60。S1训练集Top1约39%不是S2效果保证，独立RL增量仍须正式数据。
