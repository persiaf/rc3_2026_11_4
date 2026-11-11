# Research findings — Actor positive control

- Intended claim: 有通过预注册标准的最小新增LR，足以触发在线步幅对照。
- Verdict: no; confidence high，仅限本轮标准与触发判定；独立review COMPLETE。
- Positive evidence: 32真实A1/完整guard见证；3e-4拟合14/32优于3e-5的1/32。
- Limit: 都未达80%；1e-3在67尝试时因16连续拒绝停止。逐状态见证非共享权重联合可行性保证，不能归因全部是LR。
- Routing: online NOT_RUN，结果字段空白，不增加步骤/幅度/放松保护，不追加新实验。诊断权重永久禁止正式初始化。
- Evidence: ACTOR_POSITIVE_CONTROL.md、POSITIVE_CONTROL_SUMMARY.json、TRIGGER_DECISION.json及独立review trace。
- Paper use: provisional — no paper claim audit available；无泛化/路线收益结论。
