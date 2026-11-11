# R04-M2 条件锚点Critic可删除性实验

**已完成：CG/CB各128步，共256次Critic更新。当前CB残差按预登记标准删除，候选停止。** 主要收益来自fit-only条件统计，残差另有点改善，但父布局等权区间和6×6不退化门槛未通过。训练期优先保留B-cond条件分量参考，DRL与冻结Actor完整规划目标不变。

| 交付 | 内容 |
|---|---|
| [实验报告](EXPERIMENT_REPORT.md) | 设计、数据身份、归因与限制 |
| [step0/128结果](ENDPOINT_RESULTS.md) | 两聚合Value/L/K/H、分尺寸、主比较区间 |
| [完整指标CSV](attempts/02/M2_METRICS.csv) | fit/holdout、全部尺寸、原始及归一分量 |
| [父布局配对区间](attempts/02/PAIRED_PARENT_INTERVALS.csv) | 20000次配对bootstrap，全部比较/尺寸/分量 |
| [真实模型](attempts/02/models) | CG/CB的step0与128权重、Adam、RNG、游标、表和资源快照 |
| [条件组件表](attempts/02/FIT_ONLY_COMPONENT_ANCHORS.json) | 33fit父估计，未知条件回退fit-global |
| [新状态输出](attempts/02/M2_STATE_PREDICTIONS.csv.gz) | 55944行可复核证据；原数据直接复用 |
| [独立数值复核](attempts/02/INDEPENDENT_NUMERICAL_RECHECK.json) | 5880项PASS，额外优化器0 |
| [结果到主张](RESULT_TO_CLAIM.md) | 独立reviewer判定与缺失在线证据 |
| [资源账](RESOURCE_ACCOUNTING.json) | 新旧消费、首次导入失败与全部后续工作 |
| [阶段计时](TIMING_SUMMARY.json) | 真实Torch/数据加载/拟合/端点评价/复核 |
| [准确协议](M2_PROTOCOL.json) | 首次计算前固定的标准、256更新/1800秒上限 |
| [实际源码绑定](attempts/02/SOURCE_BINDING.json) | 准确环境入口、数据/基座/源码身份 |
| [数据复用核验](attempts/02/DATA_REUSE_VERIFICATION.json) | 保存268维数据的116轨迹/7992状态哈希链 |
| [C1来源](attempts/02/C1_REFERENCE_IDENTITY.json) | 旧C1固定step128来源；重训0 |
| [监督器收据](attempts/02/control/SUPERVISOR_RECEIPT.json) | 完整进程树退出，活动进程0 |
| [Git/LFS核对](GIT_LFS_VERIFICATION.json) | 远端SHA、4模型和1新证据LFS对象 |
| [下一步](NEXT_STEP_DECISION.md) | 停止候选，无自动追加或联合PPO |

M1的G3FAIL、256未运行及全部旧账保持冻结；原60、旧S3、R03/旧M3和用户Word保持边界。本轮仅已暴露holdout的开发机制筛选，离线MSE不是路线收益或新的独立泛化确认。

首次默认Python未安装Torch，0更新0数据的失败保存在本目录根；准确M1解释器的新入口在attempts/02单独登记，原协议及科学源码不变，两次入口累计计费。基准`7aeab00d50793a3bd8f35adf99e74581e29350fb`，不合并main。

结果已推送GitHub，发布提交`2c56a5e518b163be352febb138d60abf3ce3c74d`；从全新LFS临时存储下载5对象逐一核对SHA256和字节数PASS，临时存储已移除，详见发布收据。收据自身随后单独提交。
