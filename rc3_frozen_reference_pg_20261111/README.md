# 冻结参考在线PG对照

**已完成有界实际运行，当前配置STOP，不延长256。** PG-cond在64/128均未改善FrozenSL，也未优于PG-global；对匹配继续SL未通过预登记质量门槛。研究目标仍为父场景DRL及冻结Actor完整局部规划。

- 工程：19完整请求／6真实Actor尝试，累计42.48秒（外层登记执行41.23秒＋元数据检查）；上限24／16／300秒。
- 科学：三组各128机会，384次Actor更新、1536在线轨迹、768已有标签重放；64/128两终点共792评价，完整请求2328/2328成功，Critic0。
- 科学外层2490.28秒；含新增报告与独立主张审查合计3073.73秒，上限7200秒。新场景／导航求解／新教师／原60／旧S3任务均0。
- 新输出独立数值复核1326384项PASS；独立主张审查claim_supported=no、confidence=high（限本配置停止判定）。

| 交付 | 内容 |
|---|---|
| [实验报告](EXPERIMENT_REPORT.md) | 源/数据/单位、设计、实际结果、归因边界 |
| [64/128结果](ENDPOINT_RESULTS.md) | 六方法完整成本、时间、主比较父布局区间 |
| [实际科学权重](execution_v2/science/models) | 三组0/64/128训练checkpoint，64/128 Actor-only部署包 |
| [逐请求索引](execution_v2/science/REQUEST_INDEX.csv) | 全2328次J/L/K、raw/H、来源、归档位置与主要I/O |
| [完整路线归档](execution_v2/science/REQUESTS.jsonl.gz) | 原始/最终路线与完整约束、父/视图身份、native/H/内部搜索证据 |
| [完整请求计时](execution_v2/science/REQUEST_TIMING.csv) | 含归档/索引/完成journal；模型启动、特征和I/O另列 |
| [时间与新增特征汇总](REQUEST_TIMING_SUMMARY.csv) | 实际66案例冷模型请求，非仅Actor-forward |
| [全部配对统计](execution_v2/science/QUALITY_CONTRASTS.csv) | 两cohort及分尺寸，raw/H J、L/K和完整时间 |
| [在线参考汇总](ONLINE_REFERENCE_SUMMARY.csv) | G、BG/BC、中心化回报、有效残差/间隔、ratio/KL/TV |
| [完整更新诊断](execution_v2/science/ONLINE_UPDATE_DIAGNOSTICS.jsonl.gz) | 145309逐状态诊断，384机会guard与应用后状态 |
| [guard/事务日志](execution_v2/science/EVENTS.jsonl) | 真实扣账、接受/拒绝、回滚、来源及监督重放 |
| [数值复核](execution_v2/science/INDEPENDENT_NEW_OUTPUT_AUDIT.json) | 新输出独立路径PASS，未读历史大CSV |
| [真实工程验收](execution_v2/engineering/CHECKS.json) | 19请求/6真实尝试、更新/回滚/恢复/角色/导出 |
| [结果到主张](RESULT_TO_CLAIM.md) | 独立reviewer判定、限制与停止路由 |
| [独立审查trace](paper/review-traces/experiment-result-to-claim/2026-11-11_run01/) | prompt、原评语、结构化判定、路由、status |
| [累计资源账](RESOURCE_ACCOUNTING.json) | 工程/科学/失败/后处理与旧账身份分列 |
| [准确协议](PROTOCOL.json) | 首次实际调用前登记，原字节保留 |
| [实际源码绑定](execution_v2/SOURCE_BINDING.json) | A1依赖修复后的准确来源；科学实现字节未变 |
| [固定日程](SCHEDULE.json) | 首128机会×6问题/6标签及66评价案例 |
| [记录范围说明](SCHEMA_AND_ACCOUNTING_NOTES.md) | 旧collector占位、native H范围、完整计时与缓存边界 |
| [历史边界核对](HISTORICAL_BOUNDARY_VERIFICATION.json) | M1/M2/R03/旧设计小元数据与基准原字节一致 |
| [Git/LFS发布核对](GIT_LFS_VERIFICATION.json) | 远端SHA和实际新增LFS原字节 |
| [下一步](NEXT_STEP_DECISION.md) | 停止当前配置，无自动扩展或最终测试 |

冻结M2组件表SHA：`495790b78e383bf78072d613da2e5b0a100c1aa391b3055eec19121e244b3d6b`；b=−bL−0.75bK+bH，不再除n，始终不更新。神经Critic离线扩展保持结束，旧C0/C1/CG/CB不训练。

原生Fixed3/Detour16使用完整原生管线；原生raw栏是其最终管线路线。原生全部H中间链未单独捕获，最终路线和内部调用完整保存，不作独立native H归因。该限制不影响Actor/SL/PG-global主比较。此轮仅单seed已暴露开发筛选，不是新独立泛化或实机IK确认。

基准`16f5d05952c03c7ddf20623ddcc59b09c0011040`。M1 G3FAIL、M2删除CB及旧停止/权重原字节保留，不恢复旧M3/R03，不合并main，不纳入用户Word。最初0请求0更新的A1来源目录漏登记失败保存在本目录根engineering；准确原基准源码补登记与原消费一并保留。
