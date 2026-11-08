# Research findings: S3 persistence repair

- Source: `SOURCE_BINDING.json`, exact SHA256 `34fbe6c4d8b396681769ff5843a66151de5a4e7c4c300e6792d8528facc6de79`.
- Intended engineering claim: resume only unexecuted IDs under retained legacy evidence, charges and explicit missingness.
- Independent verdict: **yes**, confidence high within bounded engineering scope; technical **GO**.
- Evidence: final15testsPASS with source consistency and observed scientific entries0; independent5synthetic probesPASS; original180files and12checkpointbytes match. Three must-fix persistence/resource findings and two minor idempotency/accounting findings closed before source freeze.
- Missing evidence: actual resumed scheduling/resources and any S3 scientific quality/deployment outcome. Reviewer has executed no scientific requests.
- Routing: accept exact scoped output through one recovery revision, then perform already-authorized continuation. Do not add training, model-performance gates, imputation or repeated v3 approval.
- Paper claim audit unavailable; this does not confirm paper performance claims.

## Actual continuation stopped: independent final failure review

- Evidence: source02 `767f13bd4619bef37501db997296982baf6b1565cb25ea8cb964c2e39d381bea`; `INDEPENDENT_FINAL_FAILURE_REVIEW.md`; run03 read-only audit,15069 checks PASS in5.5891564s, scientific calls0.
- Engineering claim supported **yes**: original180 bytes retained; new1896 debits/1895 successes/one new input failure; total2059/2057/two failures; original terminal IDs not replayed; six real same-event anchor retries and complete parent failure clock closure.
- Combined claim_supported **partial**, confidence high within evidence scope. Complete S3 quality/deployment claim **no, incomplete evidence**: quality294/2268, cold881/6804, hot882/6804; six quality and twelve deployment cells all incomplete. This is not a model-performance failure verdict.
- New S3_042 failure occurs in full-parent closed-depot endpoint-aware feasibility before local/model/native/H. Saved F&A covers233/233 tasks globally; D was not persisted, so exact endpoint-filtered task IDs and local4×3 infeasibility are not established. No navigation was recomputed.
- Routing: retain all evidence and stop scientific execution pending a precise disposition of this new real input failure. No seed/budget/model/method changes, failed-slot imputation, new training, or change to the fixed DRL research objective. Paper claim audit unavailable; paper-writing verdict provisional.
# 2026-11-08 实际恢复停止后的研究发现

独立 result-to-claim 判定partial/high：持久化修复及原账继承得到工程与实际运行支持；完整S3质量/部署主张仍无完整证据，NOT_EVALUATED_INCOMPLETE，不能写成模型性能失败。真实恢复新增1896预扣/1895成功/1新输入失败；原180文件及163已消费ID保持，累计2059预扣/2057成功/2失败/13817未执行。

新失败位于S3_042全父闭合depot端点连通有效服务检查，早于局部提取及模型/native/H。F&A全局233/233有同臂覆盖，未保存D，不能进一步断言局部无解或具体失联任务。保持原DRL目标、模型终点、种子、预算和H；停止后续请求并保留负/不完整结果，先作新失败明确处置，不重训或筛选模型。

依据：[实际报告](EXPERIMENT_REPORT.md)、[独立审查](INDEPENDENT_FINAL_FAILURE_REVIEW.md)、[结果主张](RESULT_TO_CLAIM.md)。历史工程评审记录如下，原绑定评审不改。
