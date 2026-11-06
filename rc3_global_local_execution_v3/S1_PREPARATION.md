# S1-A / S1-B 正式执行准备

本页为待授权入口，不是执行命令的当前授权。固定DRL目标不变，S1监督仅为后续S2初始化和匹配对照。

## S1-A：共享付费教师池

输入为`REGISTRY.json/S1_teachers`完整256条，见CSV逐例清单。仅来自96×8训练银行，不使用开发或S3作教师。生产provider按冻结生成/重采样、F与逐臂A、实际几何/来源绑定、全父目标连接可行性检查后提取目标窗口。所有输入准备和导航都计时；困难输入不能换seed、删例或重抽。

每例端点Detour16 native一次，再额外完整H，保存native输出、H完整路线、任务/逐臂指派、验证、终局J/L/K、原动作与最终标签、内部搜索和完整费用。最多256 native+256 H=512顶层调用，1800秒，0optimizer。H缓存命中独立计账，不能伪称神经收益。完整pool/index被冻结后供所有seed共享；不完整池不能进入S1-B。

## S1-B：三种子监督初始化

按11622026、11622027、11622028固定顺序分别启动。每seed实际创建并保存独立untrained权重，然后batch4、固定256次masked teacher CE更新，1800秒。重复标签读取不调用教师/H；新教师和奖励查询上限0。保存固定256终点及完整Adam/RNG/数据索引/资源日志，不选最佳checkpoint或成功seed。

当前没有生成三份正式untrained权重，也没有将旧工程检查点当正式模型。真实S1-B授权后才执行初始化。以后S2逐seed从对应SL固定终点精确clone冻结SL/RL/继续SL，后两者fresh Adam。

## 独立入口（批准后才可执行）

公共前提是项目内准确协议批准文件、`EXECUTION_SOURCE_BINDING.json`和源码匹配的工程PASS。`<...>`是尚待用户决定的文件/实验ID，不能直接执行占位命令。

```powershell
.\rc3_learning\.venv\Scripts\python.exe -B -m rc3_global_local_execution_v3.runner --stage S1-A --experiment-id <experiment_id> --output rc3_global_local_execution_v3/runs/<experiment_id> --protocol-approval <approved_protocol_receipt> --stage-authorization <S1_A_authorization>

.\rc3_learning\.venv\Scripts\python.exe -B -m rc3_global_local_execution_v3.runner --stage S1-B --seed 11622026 --experiment-id <experiment_id> --output rc3_global_local_execution_v3/runs/<experiment_id> --existing-run --protocol-approval <approved_protocol_receipt> --stage-authorization <S1_B_authorization>
```

同一S1-B授权允许所列三seed按序分别调用；后续依次指定11622027、11622028。合法恢复额外提供`--resume <本stage已关闭checkpoint>`并保持原授权。S2必须另行授权并引用教师及全部S1-B完成收据，使用同一experiment_id，见`AUTHORIZATION_CHAIN.md`。

## 风险与停止

768个新task draw尚未实际生成或全体验证；本轮只核metadata和守卫接口。完整教师成本、训练后轨迹长度、导航/日志/检查点IO仍有资源风险。任一阶段不可行、非有限、超时或来源不匹配就保留失败/不完整记录并停止，不追加预算或轮数。

推荐总预算保持S1-A1800 + S1-B三seed各1800 + S2三seed各7200 = 28800秒阶段上限，包含实际准备与IO，不是预计耗时。科学optimizer上限仍3840，S3独立封存。真实机械臂IK/扫掠/关节轨迹尚未由代理F&A证明，不能将当前准备写成完整机器人验证。
