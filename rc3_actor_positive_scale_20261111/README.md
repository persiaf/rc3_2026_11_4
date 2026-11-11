# Actor阳性对照及条件步幅在线试验

**已实际完成阳性对照；三档均未通过事前门槛，在线阶段NOT_RUN。** 32个完整A1排序见证状态上，3e-5为1/32，3e-4为14/32；1e-3在51次接受后连续16次guard拒绝，于第67次尝试停止，保留14/32。新增LR有部分训练态学习信号，但没有达到80%阳性标准，不能触发在线实验。

共323次Actor尝试，核心登记/运行197.89秒；包含完整独立审查和小表检查的扣账上界见资源账。Critic、新教师、新场景、新导航求解、native、H与完整规划请求均0。所有实际诊断权重禁止正式初始化；未生成在线权重或路线结果。

- [阳性对照完整报告](ACTOR_POSITIVE_CONTROL.md)
- [实际A1排序见证及身份](FEASIBLE_RANK_WITNESSES.json) · [封存状态与见证参数](positive/SELECTED_TEACHER_STATES.pt)
- [逐尝试结果](UPDATE_SCALE_RESULTS.csv) · [完整guard与回滚](positive/UPDATE_GUARDS.jsonl.gz) · [实际诊断权重](positive/models/)
- [条件在线表：NOT_RUN且指标为空](CONDITIONAL_ONLINE_RESULTS.csv) · [触发决定](TRIGGER_DECISION.json)
- [资源账](RESOURCE_ACCOUNTING.json) · [真实Torch/运行环境](positive/ACTUAL_ENVIRONMENT.json) · [Windows监督器收据](positive_control/SUPERVISOR_RECEIPT.json)
- [结果到主张](RESULT_TO_CLAIM.md) · [下一步决定](NEXT_STEP_DECISION.md) · [独立审查五件trace](paper/review-traces/experiment-result-to-claim/2026-11-11_run01/)
- [独立NumPy见证复核](positive/WITNESS_INDEPENDENT_RECHECK.json) · [保存终点重载复核](positive/ENDPOINT_RECHECK.json) · [新增表一致性](OUTPUT_CONSISTENCY_REVIEW.json)
- [事前协议](PROTOCOL.json) · [实施审查](IMPLEMENTATION_REVIEW.md) · [日程](SCHEDULE.json) · [准确执行源码](SOURCE_BINDING.json)
- [历史边界核对](HISTORICAL_BOUNDARY_VERIFICATION.json) · [新增LFS远端字节核对](GIT_LFS_VERIFICATION.json)

本轮仅已知训练状态学习能力检查；逐状态可行不证明共享权重联合可行，没有泛化或路线收益结论。旧PG128停止、M1/M2失败/删除结论及资源原字节保留，不自动延长步数、改变幅度/guard或重开Critic。独立研究分支不合并main，不提交用户Word和其他无关文件。

已推送结果提交`981c6892fe8fc5df5414198de9f95bb209082e5f`，并从远端全新下载核验8个LFS对象、合计9,975,977字节，SHA256、大小与Git指针全部一致。后续提交只记录此核对收据和索引。
