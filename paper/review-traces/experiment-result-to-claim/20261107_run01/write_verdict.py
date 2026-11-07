"""Write this reviewer's verdict from closed independent saved-data checks."""
from pathlib import Path
import datetime
import hashlib
import json

TRACE = Path(__file__).resolve().parent
ROOT = TRACE.parents[3]
REPORT = ROOT / 'rc3_s2_results_20261107'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    text = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
            if isinstance(value, dict) else value)
    with path.open('x', encoding='utf8', newline='\n') as f:
        f.write(text)


checks = read(TRACE / 'summary_checks.json')
assert checks['status'] == 'PASS' and not checks['failures']
summary = read(REPORT / 'analysis/ANALYSIS_SUMMARY.json')
audits = [read(REPORT / f'audit_seed_{seed}.json') for seed in (11622026, 11622027, 11622028)]
assert all(a['status'] == 'PASS_CLOSED_SEED' and not a['failures'] for a in audits)
preservation = read(REPORT / 'FROZEN_BASELINE_AUDIT.json')
assert preservation['status'] == 'PASS'
assert preservation['baseline_commit'] == 'f69de88efa9655cf20633b61b6b0b46dd600666c'
assert preservation['mismatch_count'] == preservation['read_error_count'] == 0
seed_rows = []
for audit in audits:
    seed = audit['seed']
    x = summary['per_seed'][str(seed)]
    terminal = x['resource_ledger']['terminal']
    seed_rows.append(dict(seed=seed, RL_updates=x['RL_updates'],
        continuedSL_updates=x['continuedSL_updates'], shared_stage_wall_s=x['stage_wall_s'],
        RL_trajectory_requests=terminal['RL']['terminal_requests'],
        RL_paid_attempts=terminal['RL']['terminal_attempts'],
        development_paid_attempts=terminal['development']['terminal_attempts'],
        terminal_cache_hits=terminal['RL']['terminal_cache_hits'],
        terminal_failed=sum(v['terminal_failed'] for v in terminal.values()),
        raw_mean_J=x['raw_mean_J'], H_mean_J=x['H_mean_J'], mean_H_contribution=x['mean_H_gain'],
        RL_grad_before_mean=x['phase_diagnostics']['RL']['gradient_before']['mean'],
        RL_grad_before_min=x['phase_diagnostics']['RL']['gradient_before']['min'],
        RL_grad_before_max=x['phase_diagnostics']['RL']['gradient_before']['max'],
        RL_clipped_updates=x['phase_diagnostics']['RL']['clipped_updates_n'],
        all_zero_advantage_instances=x['all_zero_advantage_instances_n'],
        entropy_decision_weighted_mean_nats=x['entropy_decision_weighted_mean'],
        RL_relative_L2_change_from_S1=audit['parameter_deltas']['RL']['relative_L2_change_from_S1'],
        continuedSL_relative_L2_change_from_S1=audit['parameter_deltas']['continued_SL']['relative_L2_change_from_S1']))
claim = dict(
    intended_claim='大规模复杂父场景DRL训练后，冻结策略可快速完成不同规模局部任务的完整约束规划，并具有可确认的质量—时间收益。',
    claim_supported='partial', narrow_S2_completion_claim_supported='yes', confidence='high',
    what_results_support=[
        '同一获准v3与源码下，三个固定种子各完成RL512和continuedSL512；总3072次真实S2优化器更新，固定终点与完整Adam/RNG/日志可追溯。',
        '保存日志、路线、代理约束验证及资源账闭环；三条own-seed S1来源及fresh-Adam分叉未混用，真实付费终局调用均在上限内。',
        '训练中存在非零策略梯度和优势信号，参数发生变化；这些是训练执行和数值记录结论。'],
    what_results_dont_support=[
        '尚不能证明RL512优于frozenSL、continuedSL512、untrained或Fixed3/Detour，也没有独立泛化或部署质量—时间收益证据。',
        '训练任务随更新改变，首末区间与训练曲线是未配对描述，不能解释为因果性能提升；H收益不能记为网络收益。',
        '代理路线验证不等于实际机械臂IK、关节轨迹或实机安全认证。',
        'S2完成不等于S3生产评估器已实现，不自动授权保留测试。'],
    missing_evidence=[
        '符合既有冻结协议的S3评估器及其工程验证：相同任务/端点/H、冷暖完整时间、父场景统计单位、预定AND门槛。',
        '未来单独授权的S3固定终点对照：三seed的untrained/frozenSL/RL512/continuedSL512与强规则质量、时间和可行性。'],
    suggested_claim_revision='已按批准协议完成三种子大规模混合父场景/局部窗口DRL训练并冻结RL512端点；保存数据的来源、数值及资源验收通过。其冻结推理的质量—时间表现仍待S3验证。',
    next_experiments_needed=['先在不访问封存任务的条件下补全和验证S3评估器，再依独立授权运行既定S3；不追加S2更新、不更换模型、不改研究目标。'],
    routing_action='supplement_after_S3_evaluator_and_separate_authorization',
    paper_claim_audit='unavailable', writing_verdict='provisional — no paper claim audit available',
    pending_external_reviewer_delegation=False,
)
review = dict(schema='rc3-s2-independent-review-v1', status='PASS_S2_CLOSED',
    reviewer_role='Independent delegated paper architect/reviewer and saved-summary auditor',
    experiment_id='rc3_v3_global_local_20261106_01',
    completed_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    protocol_sha256=summary['protocol_sha256'], source_manifest_sha256=summary['source_manifest_sha256'],
    S2_authorization_sha256='4dba652b2156b88848c4c1c49826952414a5b6d08baa7ce250d710b3ea312423',
    S2_completion_readiness='GO', S3_model_lineage_readiness='GO',
    S3_execution_readiness='REVISE_PENDING_FROZEN_PROTOCOL_EVALUATOR_AND_SEPARATE_AUTHORIZATION',
    S3_authorized=False, S3_executed=False, original60_semantic_access=0,
    scientific_optimizer_model_inference_scene_navigation_teacher_H_calls_during_review=0,
    prior_independent_raw_seed_audit_assertions=sum(a['checks_total'] for a in audits),
    prior_independent_raw_seed_audit_failures=0,
    new_independent_summary_checks=checks['checks_total'], new_independent_summary_failures=0,
    analysis_checks=summary['analysis_checks'], analysis_finite_checks=summary['finite_checks'],
    checkpoints_audited=102,
    totals=dict(RL_updates=1536, continuedSL_updates=1536, S2_optimizer_updates=3072,
        RL_trajectory_requests=18432, RL_paid_attempts=18431, development_paid_attempts=108,
        paid_terminal_attempts=18539, terminal_cache_hits=1, terminal_failed=0,
        shared_stage_wall_s=sum(a['shared_stage_wall_s'] for a in audits),
        stage_wall_cap_each_seed_s=7200, paid_terminal_cap_each_seed=8192),
    development_denominator=dict(unique_declared_DEV_tasks=18, checkpoints=[256,512], seeds=3,
        repeated_evaluations=108, independent_cases_is_not_108=True, sealed_S3_queries=0),
    seeds=seed_rows,
    source_and_preservation_review=dict(approved_source_files=34, byte_mismatches=0,
        preservation_receipt_sha256=sha(REPORT / 'FROZEN_BASELINE_AUDIT.json'),
        scope='Original tracked scientific evidence byte audit; .gitattributes inspection preceded the additive S2-only LFS rule; mutable metadata and live user Office edits are explicitly excluded.'),
    visual_QA=dict(status='PASS', PNG_files_visually_inspected=9, paired_PDF_files_hash_verified=9,
        independent_PDF_rasterization=False,
        findings='All nine PNGs show actual closed curves, readable labels/units and expected seed ordering. Captions distinguish smoothing, descriptive intervals, H attribution, shared wall clocks and dependent draws. PDFs are paired exports of the same Matplotlib figure source; no separate all-page PDF rendering is claimed.'),
    material_risks=[
        'RL全1536次更新均触发norm1裁剪，原始范数均值约559–617、最大约2466；说明梯度尺度很大，不能仅凭finite或完成更新认为学习有效。本轮遵守固定配方，不事后修改。',
        '训练回报无配对比较和冻结端点比较；三种子观察分布、熵与H贡献不同，未证明收益稳定性。',
        '已保存模型谱系完备，但当前authority/runner仅S1-A/S1-B/S2，TrainingRegistry仅train/dev；S3执行准入仍REVISE。'],
    maintenance_exceptions=[
        '早先只读seed27审计遇到缺省0的稀疏navigation键，独立报告脚本改用get(key,0)后重读；未重跑科学阶段。',
        '只读分析导出checkpoint时phase重复键导致TypeError；补充脚本改为先核phase一致再合并字段。正式源码、模型、协议不变，未新增科学消费。',
        '失败维护尝试未单独测得的墙钟不补造；现有三份成功raw-audit计时仅覆盖相应循环，不代表全部维护总耗时。'],
    limitations=[
        'This final reviewer rechecks closed audit receipts and derived summaries; full raw route/checkpoint scans are the three separately saved prior independent seed audits, not repeated here.',
        'Raw audit assertion counts, derived analysis checks and new summary checks overlap; they must not be summed as unique observations or independent cases.',
        'Saved geometric/proxy validation verdicts and static route arithmetic are not new geometry checks, physical IK or joint-trajectory certification.',
        'Remote synchronization/LFS publication is reviewed by the separate release receipt; this scientific review does not predeclare a successful push.'],
    claim_verdict=claim,
    artifact_sha256=dict(checks['artifact_sha256'], **{
        (TRACE / 'summary_checks.json').relative_to(ROOT).as_posix():sha(TRACE / 'summary_checks.json'),
        (TRACE / 'verify_saved_summary.py').relative_to(ROOT).as_posix():sha(TRACE / 'verify_saved_summary.py'),
        (REPORT / 'independent_audit_s2.py').relative_to(ROOT).as_posix():sha(REPORT / 'independent_audit_s2.py'),
    }))
table = '\n'.join('| {seed} | {RL_updates} | {continuedSL_updates} | {shared_stage_wall_s:.3f} | {RL_paid_attempts} + {development_paid_attempts} | {terminal_cache_hits} |'.format(**r) for r in seed_rows)
md = f'''# S2 独立审查

结论：`PASS_S2_CLOSED`。正式 S2 的三种子固定训练端点和保存证据验收通过。
S3 模型谱系就绪为 `GO`；S3 执行准入为 `REVISE`，原因是当前生产源码尚无 S3 评估器，且本轮无 S3 执行授权。

| seed | RL更新 | continuedSL更新 | 共享阶段墙钟 s | RL＋DEV真实终局调用 | 缓存命中 |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

总计 3072 次 S2 optimizer 更新，18432 条 RL 轨迹请求，18539 次真实终局评价与 H／导出／验证，1 次终局缓存命中，0 次终局失败。逐 seed 均低于 7200 秒与 8192 次硬上限。总共享阶段墙钟为 9237.6241247 秒；各 phase 的累计 ledger/时钟不能重复相加。

本 reviewer 复核已有三份逐保存文件独立审计：合计 7,362,042 条断言、0 失败、102 个 S2 完整检查点。它们覆盖 own-seed SL256 来源、fresh Adam 分叉、模型/Adam/RNG/游标、日志前缀与 checkpoint 关闭边界、合法 mask、三起点、奖励/LOO/混合目标、保存路线的代价/折线算术与代理验证记录、调用及预算。

本轮最终独立汇总验证另执行 55,468 条检查，0 失败：直接核验 34 个获准生产源码字节、协议/registry、真实 S2 授权与前驱 COMPLETE SHA；从 CSV 重算各 seed 的原始及 H 后 J/L/K 和 H 贡献，核对固定 512 游标、账本、开发分母和所有图源/输入/输出 SHA。完整断言见 [summary_checks.json](../paper/review-traces/experiment-result-to-claim/20261107_run01/summary_checks.json)。检查计数互有重叠，不能相加成独立样本量。

保存数据分析自身 375893 条检查、其中 283587 条 finite 检查均为零失败。18432 条轨迹指标均连接到精确保存终局记录，没有代数回填。这里没有运行新的模型/optimizer/推理、场景、导航、教师或 H。

开发评价的分母为原 18 个 DEV 任务，在 update256/512、三 seed 共重复 108 次；不是 108 个独立场景，更不是 S3 的 108 个封存目标。直接检查 authority.STAGES、runner choices 和 TrainingRegistry，仅支持 S1-A/S1-B/S2 与 train/dev。不能从训练端点齐备推出 S3 评估器已可执行。

九张 PNG 已逐一目检，曲线端点、单位、图注、seed 顺序和布局均通过；配对九份 PDF 的字节和同源生成绑定通过，没有把 PNG 目检说成另做了全 PDF 渲染。平滑窗口、观察分位区间、共享时钟与 H 的归因均明确。各图为训练诊断，未用于开发集选权重或替换固定 RL512 终点。

原冻结证据使用已有全字节审计 PASS，原科学结果未覆盖；该审计中的 `.gitattributes` 为新增 S2-only LFS 规则之前的检查，不能宣称后续属性文件毫无变化。原生产源码 34 份现在再次核验完全一致。原 60 与 S3 任务没有在本审查中进行语义读取。

数值风险必须保留：全部 1536 次 RL 更新均触发 norm1 裁剪，原范数均值约 559–617，最大约 2466。梯度有限、端点改变和训练完成说明机制运行，并不证明学习有效；本轮未据此更改学习率、奖励或算法。训练输入随更新变化，三种子熵与回报/H贡献波动只作描述。旧只读审计的稀疏键错误和分析的 phase 重复键错误均在补充报告脚本解决；没有科学重跑，未量测的维护失败墙钟不补造。

本审查确认保存的代理约束验证与路线算术，未重新执行几何/可达求解；不能写成真实机械臂 IK、关节轨迹或实机安全认证。GitHub 和 LFS 是否完成，以独立发布回执的实际远端核对为准，本审查不预先宣告同步。

## 结果到主张

整体性能主张为 `partial`，confidence `high`；窄范围“按协议完成三种子 S2 并冻结可追溯端点”为 `yes`。

{claim['suggested_claim_revision']}

尚未证明 RL512 优于 frozenSL、continuedSL512、untrained 或 Fixed3/Detour；未证明泛化、部署速度或独立 DRL 收益。原始到 H 后的下降归于固定 H。训练曲线和 SL CE 下降不能替代上述证据。

下一步保持固定研究目标，先补全并工程验证符合既有冻结协议的 S3 评估器；S3 的真正目标访问与运行须独立明确授权。本轮不追加 S2、不挑选其他 checkpoint、不启动 S3 或解封原60。

`paper/PAPER_CLAIM_AUDIT.json` 不存在，论文写作判定标记为 `provisional — no paper claim audit available`。独立 reviewer 委派已实际完成，不冒充外部论文数据审计。完整 prompt／raw verdict／parsed verdict／routing／check script 见 [review trace](../paper/review-traces/experiment-result-to-claim/20261107_run01/)。
'''
write(REPORT / 'INDEPENDENT_REVIEW.json', review)
write(REPORT / 'INDEPENDENT_REVIEW.md', md)
write(TRACE / 'raw_reviewer_response.md', md)
write(TRACE / 'raw_reviewer_response.json', review)
write(TRACE / 'parsed_verdict.json', claim)
write(TRACE / 'routing_action.json', dict(routing_action=claim['routing_action'],
    pending_external_review=False, paper_claim_audit='unavailable',
    writing_verdict=claim['writing_verdict'], experiments_automatically_started=False,
    S3_authorized=False, fixed_research_goal_preserved=True,
    reason='S2 execution evidence is complete; frozen inference and matched baseline performance remain untested.'))
write(TRACE / 'status.json', dict(status='COMPLETE', final_independent_review_status='PASS_S2_CLOSED',
    claim_supported='partial', narrow_S2_completion_claim_supported='yes',
    reviewer_delegation='Actual delegated independent reviewer', external_paper_claim_audit='unavailable',
    source_of_raw_audit='Three separately saved independent closed-seed audits',
    raw_audit_assertions=7362042, new_summary_assertions=55468,
    new_scientific_calls=0, protected_semantic_access=0,
    review_md_sha256=sha(REPORT / 'INDEPENDENT_REVIEW.md'),
    review_json_sha256=sha(REPORT / 'INDEPENDENT_REVIEW.json')))
print(json.dumps(dict(status=review['status'], claim_supported=claim['claim_supported'],
    review_json_sha256=sha(REPORT / 'INDEPENDENT_REVIEW.json'),
    review_md_sha256=sha(REPORT / 'INDEPENDENT_REVIEW.md')), ensure_ascii=False))
