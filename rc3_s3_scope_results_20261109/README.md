# S3 scope_r1 正式恢复实际交付

**13817/13817新增请求成功，新增失败0；累计15874份有效结果、2个历史缺失。** 三seed按序封账，未追加训练或读取原60。5个完整质量格及9个完整部署格全部未通过原门槛；另外1质量格／3部署格INCOMPLETE，全局AND为null。当前RL没有建立对监督或强规则的质量与时间优势。

- [正式结果与全部表格](EXPERIMENT_REPORT.md)、[结果到主张](RESULT_TO_CLAIM.md)、[下一步决定](NEXT_STEP_DECISION.md)
- [逐请求J/L/K、raw/H成本与完整时间CSV](results/REQUEST_METRICS.csv)、[七方法质量—时间](results/METHOD_QUALITY_TIME.csv)、[三seed质量](results/QUALITY_BY_MODEL_SEED.csv)
- [六个质量格和十二个部署格](results/SCIENTIFIC_ANALYSIS.json)、[匹配比较与父布局区间](results/MATCHED_COMPARISONS.json)、[计时版本](results/TIMING_BY_REVISION.csv)
- [全部原始结果索引与哈希](results/RESULT_ARTIFACT_INDEX.json)、[请求状态与历史失败](results/REQUEST_STATE_INDEX.json)、[只读覆盖/同臂审计](results/CONSTRAINT_SOURCE_AUDIT.json)
- [累计及本轮资源](RESOURCE_ACCOUNTING.json)、[完整模型来源](results/MODEL_SOURCE_AUDIT.json)、[协议/源代码/授权身份](EVIDENCE_IDENTITY.json)
- [独立审查](INDEPENDENT_REVIEW.md)、[技术执行复核](IMPLEMENTATION_REVIEW.md)、[Git版本审计](GIT_VERSION_AUDIT.json)

生产运行完整输入、来源修订、Journal、时钟、封口和13817份新结果在[scope_r1运行目录](../rc3_s3_connectivity_20261109/runs/rc3_v3_global_local_20261106_01/scope_r1/)。模型/NPZ/大Journal遵循Git LFS，请检出提交后执行`git lfs pull`。不生成额外整包备份，不覆写S1-A/S1-B/S2及旧S3失败证据。

评估源码SHA256 `311a58bd867e95c852ec3246c54a398468eb526db391fbb6d12239c6a3379949`，与训练源码身份分开。输入范围为v3+scope_r1，不能称未修订原v3完整网格通过。原阶段精确历史墙钟仍null，592保守预算结转保留；两失败不填补。科研结果限于静态代理约束和本配置，不声称实机IK认证或DRL一般无效。
