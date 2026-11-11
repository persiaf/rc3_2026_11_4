# 本轮交付索引

独立分支：`codex/rc3-critic-rank-diagnostics-20261111`；基准：`83b0df9ceb6b43c5e04de883db1376558393af4d`。

| 用户要求的文件 | 内容 |
| --- | --- |
| [ACTOR_RANK_RESPONSE.csv](ACTOR_RANK_RESPONSE.csv) | 8150个原动作前状态；合法gap、±0.25理论界、6种权重/特征条件、残差及居中能量、排序、pre/post ratio |
| [CRITIC_MATCHED_CALIBRATION.csv](CRITIC_MATCHED_CALIBRATION.csv) | 三配置0/256及原保存预测描述；L/K/H、归一分量和value；pooled/父等权、常数参照及偏差分解 |
| [CALIBRATION_SPLIT.json](CALIBRATION_SPLIT.json) | 50个物理父布局33fit/17holdout，全部轨迹归属及分层规则 |
| [MECHANISM_FINDINGS.md](MECHANISM_FINDINGS.md) | 五个问题的明确回答、支持与反证 |
| [RESOURCE_ACCOUNTING.json](RESOURCE_ACCOUNTING.json) | 全部真实计算、来源读取/哈希、失败和418.6573秒累计墙钟 |
| [NEXT_MINIMAL_REVISION.md](NEXT_MINIMAL_REVISION.md) | 仅C0→C1最小候选，不自动开始联合PPO |

其他可审查证据：

- 首算前身份：PREREGISTRATION、TRAJECTORY_SELECTION、BATCH_SCHEDULE、两份authorizations、IMPLEMENTATION_REVIEW与独立STATIC_REVIEW。
- 源码：actor.py、critic.py、data.py、run.py、ledger.py、register.py、summaries.py、independent_recheck.py；recover_environment.py仅修复精确零计算环境失败，同账接续，原登记源码未改。
- 原始新输出：ACTOR_CANDIDATES.csv.gz约448MB；CRITIC_STATE_PREDICTIONS.csv.gz约15MB；CRITIC_STRATIFIED_CALIBRATION.csv；TRAINING_LOG.csv；FIT_ONLY_REFERENCES.json；models/初始化及三个256终点。
- 独立全量数值复核：NUMERICAL_RECHECK.json，41,591,999检查PASS；完整来源重建及缺项在RECONSTRUCTION_RECEIPT与MISSING_SAVED_FIELDS。
- 主张审查：INDEPENDENT_MECHANISM_REVIEW.md/.json、findings.md、RESULT_TO_CLAIM及paper/review-traces/experiment-result-to-claim/20261111_run01。
- 失败保存：attempts/ATTEMPT01_RESOURCE_ACCOUNTING.json及ENVIRONMENT_RECOVERY_RECEIPT；events.jsonl包含原失败、接续及完整实际消费，不能作为新预算重播。
- Git/LFS：GIT_START_AUDIT、LFS_OBJECTS及后续PUBLICATION_RECEIPT。7项LFS共509,462,148字节；必须核对远端后才报告同步。

旧科研源码、R03目录及原根.gitattributes保持原字节；原60/旧S3未读取任务。原用户Word、新增论文文件、研究情况ZIP/目录未纳入提交。仓库级项目清单仅追加本轮证据，不刷新原用户文件哈希。没有整包备份、分支删除、强推或main合并。
