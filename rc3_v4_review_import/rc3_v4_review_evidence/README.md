# 第四版验收与维护证据包

本包与用户原第四版交付包配合使用。原包保持冻结，本包提供独立重算、限定补评和可审查的维护副本，没有再训练模型。

## 主要结论

- 原855份方案、357记录、29项实验冻结哈希和89组汇总/比较统计均能复现；全包270文件哈希一致。
- 原版34项无需Torch的单测实际重跑通过。原报告40项不等于本次重跑40项。
- 小规模225份方案全部复验，45组合重算得到44个离散最优与1个unsupported。
- 固定8次修复时，Structured相对Detour降本0.7699%，但相对Control、旧MLP的额外收益区间跨零，且时间未达到原采用门槛。
- 补做冻结旧MLP对Detour的当前机器400 ms软预算对照：18场景、108次，均通过验证；旧MLP平均降本0.2880%，95%场景区间[0.0268%,0.5818%]。实际规划中位449.38与446.53 ms。此为冻结后单机描述性补充，仍为软预算，两法各54/54次超时。
- 外部共同可用臂补丁20项回归通过；图嵌入缓存144步输出、108次完整方案一致，在原18验证场景显示提速潜力。两者均在隔离文件中，未替换原评测源码。

## 目录

|目录|内容|
|---|---|
|experiments|855份方案实际复验、统计重建、CI与门槛、PNG/SVG科学图|
|exact|225方案验证、45组合精确复算及脚本|
|training|教师数据与模型核对、非Torch测试、图嵌入缓存适配器及诊断|
|external|外部接口、回调及原保守误拒绝反例|
|external_patch|修订external.py、unified diff、20项回归、原新SHA|
|mlp_budget_supplement|冻结旧MLP同预算108次补评完整方案和计时|

审查文档为`rc3_v4_acceptance_review.docx`。科学图解释原Windows结果，不包含后来的Linux补评；补评单列于自身README与JSON/CSV。

## 复现

所有主要脚本都支持`--source-root`，指向原ZIP解压后同时含rc3_v4、rc3_v3、rc3_fast、rc3_improve、rc3_baseline的目录；输出使用新目录。Python与NumPy/SciPy等依赖沿用原项目，不需要安装Torch来运行这里的审计和纯NumPy推理。

```bash
python -B experiments/audit_v4.py --source-root /path/to/v4 --output-dir /path/to/audit
python -B exact/audit_exact.py --source-root /path/to/v4 --output-dir /path/to/exact
python -B external_patch/test_external_common_arm.py --source-root /path/to/v4 --output-dir /path/to/patch-tests
python -B training/cache_diagnostic.py --source-root /path/to/v4 --output-dir /path/to/cache --scenes 18 --repeats 3
```

训练核对与补评的完整运行参数见各目录README或审查报告。审查脚本可能读取原数值数据缓存；`training/safe_numeric.py`仅允许已列明的NumPy数值构造，不运行任意pickle全局对象。PT文件未加载。

## 采用维护改进

1. 在原项目的开发副本中审查并应用`external_patch/external_common_arm.patch`。如果原文件已改动，先核对原新SHA和差异；不要强行覆盖已有修改。补丁只修改共同可用臂分配，仍须运行独立终验与外部checker。
2. 图缓存是可选适配器，导入`training/graph_embedding_cache.py`中的`CachedGraphPolicy`后，按原`runtime.run(problem,'learned',policy,8)`接口运行。权重与F/D/任务候选语义在Env内固定；变更地图、任务或权重必须失效缓存或新建Env。
3. 新维护源码使用新版本号和清单；保留原v4模型、正式结果与清单。不要把旧855份正式结果改标为新源码的实验结果。

## 证据边界

通过验证只说明给定代理模型或外部表下的方案一致性。AlohaMini1身份已有文档引用，但仍缺一致、可运行的资产和服务约束配置。双臂同步、升降过渡、机械臂连续运动和物理抓取不在当前证书范围。

本包未复制约296MB的原交付解压数据。复现原855份与训练数据核对需要原交付包；新108份补评计划完整包含在本包。统计以场景为单位，重复与种子不充当新增独立场景。
