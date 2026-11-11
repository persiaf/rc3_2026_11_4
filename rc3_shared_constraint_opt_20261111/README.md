# 固定A1表示上的共享约束优化诊断

**实际完成，质量FAIL13/32，关闭本固定状态诊断支线。** 32最终软约束QP全部COMPLETE；112实际参数候选、32接受更新、80精确恢复。旧Adam hinge18/32、CE14/32保持，不声称等算力。隔离22位置构造核验PASS，只作标签相关存在性检查。

核心含来源登记扣账62.6488秒；含完整独立数值/主张审查的总扣账上界418.09/1800秒。实际QP32、dual迭代9050、Actor参数候选112，绝不写optimizer=0。Critic/新任务/导航/新教师/native/H/完整请求全0。

- [SHARED_OPTIMIZATION_REPORT.md：完整报告](SHARED_OPTIMIZATION_REPORT.md)
- [LOCAL_SUBPROBLEM_LOG.csv](LOCAL_SUBPROBLEM_LOG.csv) · [ACTUAL_STEP_RESPONSE.csv](ACTUAL_STEP_RESPONSE.csv) · [STATE_MARGIN_RESULTS.csv](STATE_MARGIN_RESULTS.csv)
- [RESOURCE_ACCOUNTING.json](RESOURCE_ACCOUNTING.json) · [NEXT_STEP_DECISION.md](NEXT_STEP_DECISION.md)
- [实际诊断权重与原始日志](diagnostic/) · [54份子问题证书](diagnostic/SUBPROBLEM_CERTIFICATES.jsonl.gz) · [每状态实际候选输出](diagnostic/CANDIDATE_STATES.jsonl.gz)
- [优化终点](OPTIMIZATION_RESULT.json) · [隔离22位置路径](ISOLATED_PATH_PROBE.json) · [新输出摘要](NEW_OUTPUT_SUMMARY.json)
- [独立审查五件trace及数值复核](paper/review-traces/experiment-result-to-claim/2026-11-11_run01/) · [结果到主张](RESULT_TO_CLAIM.md)
- [事前协议](PROTOCOL.json) · [精确源码](SOURCE_BINDING.json) · [实施审查GO](IMPLEMENTATION_REVIEW.md) · [真实Torch](diagnostic/ACTUAL_ENVIRONMENT.json) · [监督器](diagnostic_control/SUPERVISOR_RECEIPT.json)
- [历史字节核对](HISTORICAL_BOUNDARY_VERIFICATION.json) · [Git/远端LFS字节核对](GIT_LFS_VERIFICATION.json)

不追加损失/配置/训练，不自动启动在线DRL。共享权重不作优化初值/目标/方向，全部模型禁止部署及正式初始化；无泛化或路线收益结论。未合并main，未提交用户Word或无关文件。

结果提交`e7d4dc3aacb31c14b7c8be06f9adc830622b9aa3`已推送且远端SHA匹配；从远端重新下载核验5个新增LFS对象、26,051,133字节，SHA256/大小/指针全部PASS。后续提交只记录此核对收据和索引。
