"""Publish the independent judgment from completed, saved-only audit artifacts."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

TRACE=Path(__file__).resolve().parent
HOME=TRACE.parents[3]
ROOT=HOME.parent


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(p,v):
    p=Path(p)
    assert p.parent==TRACE or p in [HOME/'INDEPENDENT_REVIEW.json',HOME/'INDEPENDENT_REVIEW.md']
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n' if not isinstance(v,str) else v,encoding='utf8')


numeric=read(TRACE/'independent_numeric_audit.json')
extra=read(TRACE/'additional_saved_data_checks.json')
assert numeric['status']=='PASS_SAVED_EVIDENCE_AUDIT' and extra['status']=='PASS'
checks=numeric['checks'];dev=checks['development'];dep=checks['deployment'];training=checks['training']
comparisons=dev['comparisons']
assert dev['candidate_gate']=='REVISE'
now=datetime.now(timezone.utc)
review_start=datetime(2026,11,9,8,29,29,tzinfo=timezone.utc)
claim_parts={
 'actor_exact_quality_and_observed_speed':dict(claim_supported='yes',confidence='high',technical_verdict='GO',
    claim='In this one-machine, one timing seed, six fixed DEV windows from one parent, D1/D2 preserved exact routes/costs/assignments and recorded shorter complete-request medians for neural methods.',
    support='12/12 actor exports are exact; 756/756 requests follow the fixed grid/order; 42 case-method outputs are identical over lifecycle/cache/repeats; 540 encodings are fresh.',
    limit='Observed timing only. Fixed D0→D1→D2 order and native-control timing drift prevent an unqualified causal/hardware/generalization speed claim. Include startup and separate cold/hot results.'),
 'B_vs_A':dict(claim_supported='partial',confidence='medium',technical_verdict='REVISE',
    support='Raw equal-parent relative gain is positive in all three parents; mean and descriptive bootstrap interval exceed zero.',
    not_supported='The registered common-H final-quality improvement criterion fails. Positive raw comparison cannot promote the candidate or establish a first-action-only cause.'),
 'B_vs_SL':dict(claim_supported='no',confidence='medium',technical_verdict='REVISE',
    support='The grand absolute case-mean raw and postH J are lower for B in this saved DEV sample.',
    not_supported='Raw equal-parent interval spans zero and postH equal-parent mean is negative with interval spanning zero; no registered gain over the shared SL256 start.'),
 'overall_DRL_research_goal':dict(claim_supported='no',confidence='high',technical_verdict='REVISE',
    support='The saved screen verifies implementation/data/weight lineage and exact deployment artifacts while retaining the fixed DRL objective.',
    not_supported='This single-seed 128-update DEV screen does not establish convergence, robust DRL improvement, generalization to unseen layouts/scales, S3 success or real-arm IK/joint trajectory safety.')}
missing=[
 'PostH improvement over BOTH matched original A and the common SL start under the preregistered equal-parent statistic.',
 'Independent optimizer seeds and more independent physical parent layouts; the 18 cases are nested in only three parents.',
 'A separately preregistered factorial/one-factor design if causal attribution to forced starts versus mixture weights is desired; B changed both jointly.',
 'Counterbalanced/repeated lifecycle measurements for a causal speed statement and an exact process-to-first-response measurement if that latency is a delivery target.',
 'Broader frozen-policy evaluation under its own stage authority; this review does not authorize new S3/old60 access or additional training.',
 'Real robot calibration, IK and arm/joint swept-trajectory evidence before a physical reachability/safety certificate.',
 'A supplied paper and paper/PAPER_CLAIM_AUDIT.json for non-provisional paper writing.'
]
revision='Exact actor-only export and parameter residency reduce observed lifecycle overhead without changing saved DEV plans. The joint all-policy/equal-weight PG variant improves raw planning cost relative to original A on this small DEV screen, but fails the preregistered final postH candidate gate against A and the common SL256 start.'
routing=dict(action='supplement_after_design_revision',candidate_gate='REVISE',confirm_paper_writing=False,
    immediate='Close this bounded experiment and retain exact actors plus full negative/mixed mechanism evidence. Keep the DRL research target and old frozen evidence.',
    proposed_next='Prepare a revised DRL protocol using this raw/postH discrepancy and gradient clipping evidence; choose a mechanism question and independent-seed/parent validation before seeking any new execution authorization.',
    new_science_executed_by_review=False,new_execution_authorization_granted=False,
    no_supervised_replacement=True)
verdict=dict(schema='rc3-independent-result-to-claim-verdict-v1',experiment_id='rc3_actor_pg_screen_20261109_01',
    reviewer_role='independent delegated paper architect/reviewer',claim_supported='partial',confidence='high',
    confidence_scope='High for audited saved facts and failure to satisfy the registered gate; mechanism/generalization inference remains limited to one seed and three layouts.',
    technical_verdict='GO',technical_scope='Completed bounded execution is technically admissible and saved evidence consistent; GO does not authorize any next optimizer or held-out request.',
    scientific_candidate_verdict='REVISE',parts=claim_parts,
    what_results_support=revision,
    what_results_dont_support='A confirmed postH DRL quality improvement, isolated forcing causality, convergence, unseen-layout generalization, old S3 success, real robot joint-trajectory certification, or an unqualified causal speedup.',
    missing_evidence=missing,suggested_claim_revision=revision,next_experiments_needed=routing,
    paper_claim_audit=numeric['paper_claim_audit'],writing_verdict=numeric['downstream_writing'],
    pendingExternalReview=False,external_review_basis='A distinct delegated reviewer independently inspected source and recomputed saved artifacts; no pending reviewer delegation.',
    review_trace='paper/review-traces/experiment-result-to-claim/20261109_run01/',
    source_binding_sha256='57b0d0d2236b4b1f942a393ce2fc41e6408f4e7ec899df038bdfc4b19a65eee6',
    protocol_sha256=sha(HOME/'PROTOCOL.json'),baseline='925a0a1a8957299e3e849b29ac501a99212aa902',
    audit_script_wall_s=numeric['audit_wall_s']+extra['audit_wall_s'],
    review_wall_s=(now-review_start).total_seconds(),review_started_utc=review_start.isoformat(),review_finished_utc=now.isoformat(),
    new_scientific_calls=numeric['new_scientific_calls'],
    evidence=dict(primary_audit='paper/review-traces/experiment-result-to-claim/20261109_run01/independent_numeric_audit.json',
        additional_audit='paper/review-traces/experiment-result-to-claim/20261109_run01/additional_saved_data_checks.json'),
    key_numbers=dict(deployment_requests=756,exact_actor_models=12,training_updates=256,training_terminal_requests=3072,DEV_requests=54,
        DEV_parent_clusters=3,DEV_cases=18,optimizer_seeds=1,comparisons=comparisons,absolute_model_means=extra['absolute_case_means'],
        resources=checks['resources'],complete_resource_reconciliation=extra['complete_resource_reconciliation']),
    audit_limitations=[
        'No new model forward/backward allowed: gradient values and logged likelihoods were not regenerated. Audit checks full saved tensors, finite/norm/clip relations, exact loss/return/LOO arithmetic, source code and Adam delta/moment consistency.',
        'Saved plan validation certificates were checked for consistency; validation/planning/navigation/H was not rerun.',
        'Historical evidence was byte-hashed without interpreting held-out task/result contents.',
        'The skill reference review-tracing.md/assurance-contract.md was absent at its documented relative path. Explicit prompt/raw/parsed/routing/pending/manifest requirements were fulfilled directly.',
        'Git/remote publication is the parent executor responsibility; this reviewer did not change Git or production files.'
    ])

lines=[
 '# 独立结果与主张审查',
 '',
 '**结论：证据审计 GO；本轮候选晋级 REVISE；总体 `claim_supported=partial`。** 这里的 GO 仅表示已完成的有界执行与保存证据可采，不授权新训练、旧 S3 补跑或保留测试解封。',
 '',
 '十二个 actor-only 导出与原始参数逐项完全相等，本机固定 DEV 请求中 D1/D2 记录到加载生命周期加速且规划输出精确等价。B 相对 A 的 raw 成本有正向小试证据，但未传递为预登记的 postH 父布局相对收益；B 对共同 SL256 起点也未通过。不能据此确认 DRL 性能提升、收敛或 S3 成功。',
 '',
 '## 审查身份与边界',
 '',
 '- 独立委派的 paper architect/reviewer；非执行者自行确认。`pendingExternalReview=false`。',
 '- 基准 `925a0a1a8957299e3e849b29ac501a99212aa902`；源码绑定 `57b0d0d2236b4b1f942a393ce2fc41e6408f4e7ec899df038bdfc4b19a65eee6`。',
 '- 独立读取 202 个绑定源码、1,392 个输入来源、12 个导出、756+54 原始评估、3,072 个训练终局与 256 份更新完整梯度/权重。49,638 个历史证据文件共 9,899,434,805 字节只核原始字节 SHA，未解释旧60或旧S3任务内容。',
 '- 本审查模型 forward、rollout、optimizer、backward、场景/任务生成、导航求解、native、H、教师、旧S3重跑全部为 0。旧实验失败与缺口不回填。',
 '- 没有提供论文或 `paper/PAPER_CLAIM_AUDIT.json`：`paper_claim_audit=unavailable`，写作判断为 provisional；实际结果报告仍可交付。',
 '',
 '## 精确制品与部署结果',
 '',
 '全部 12 个导出均核对名称、shape、dtype、逐值 `torch.equal`、整体及逐参数 hash；原训练 checkpoint、sidecar、commit 共 36 文件保留。actor 仅含原模型参数、架构与来源，未量化、剪枝或蒸馏。每个actor文件268,504字节；用于计时的seed11622026 RL512完整checkpoint为37,950,130字节，其他角色完整checkpoint大小不同，因此各角色提速幅度应分别报告。',
 '',
 '独立重建 PCG64/11642026 的 756 请求顺序，与每个请求的预扣账、完成日志、原始文件 hash 一致。42 个 case×method 组合在 D0/D1/D2、cold/hot、3 重复下 raw/final 路线、J/L/K、来源和 assignments 精确相同；108 对 RL_raw/RL_plus_H 的 raw 路线一致。540 次神经请求均重新编码，432 次部署共同 H 与方法定义相符。',
 '',
 '| 生命周期 | 启动至 ready (s) | 请求中模型加载 | 驻留命中 |',
 '|---|---:|---:|---:|'
]
for mode in ['D0','D1','D2']:
    d=dep['modes'][mode]
    lines.append(f"| {mode} | {d['startup_s']:.6f} | {d['model_loads']} | {d['resident_hits']} |")
lines += ['', 'D0 为完整 checkpoint 逐请求加载，D1 为 actor-only 逐请求加载，D2 在启动阶段加载 4 个 actor，键为 `(seed, actor SHA256)`，仅模型参数常驻。D2 的动态编码、历史状态和路线不跨请求缓存。',
          '', '下表为完整请求 median / p95 秒，每单元 18 请求；包含任务解析与来源校验、导航/view、模型访问、fresh encode、构造、H、导出验证及制品写入/hash。hot 导航预热发生在请求计时外，但计入阶段墙钟。嵌套组件不可相加。',
          '', '| 方法 | 导航 | D0 median / p95 | D1 median / p95 | D2 median / p95 |', '|---|---|---:|---:|---:|']
tim={(d['method'],d['mode'],d['navigation']):d for d in dep['timings']}
methods=['Fixed3_endpoint_adapter','Detour16_endpoint_adapter','untrained','frozen_supervised','continued_SL','RL_raw','RL_plus_H']
for m in methods:
    for nav in ['cold','hot']:
        vals=[]
        for mode in ['D0','D1','D2']:
            d=tim[m,mode,nav];vals.append(f"{d['median_s']:.6f} / {d['p95_s']:.6f}")
        lines.append('| '+m+' | '+nav+' | '+' | '.join(vals)+' |')
lines += ['', '以旧 RL512 的 RL_plus_H 为例，D0→D2 的 cold median 为 1.342769→0.697890 s（1.924×），hot 为 1.104739→0.465004 s（2.376×）；p95 分别为 1.588658→0.834002 s 与 1.506505→0.535066 s。',
          '', '**解释范围：** 只测一台机器、一个计时 seed、同一父布局的六个旧 DEV 窗口，顺序固定 D0→D1→D2。native 对照也发生时间漂移，尤其 p95，因此不能将所有差异归因于 actor 制品或外推通用硬件因果加速。D2 相对 D1 的增量应从表中独立看待。启动成本不能隐去。启动与首请求 span 之和不含中间 receipt/predebit IO，且首方法不同，不能称为精确进程到首响应时间。',
          '', '## 匹配策略梯度与谱系',
          '', 'A/B 的 0 号完整 checkpoint 均为同一 SL256 精确参数 `1817f58aa6506456ff12f0c5fe0a914e156aef8af319f4687dde90782f55a63e`，Adam state 为空、查询游标为0、named RNG为空。两组 128×4 个 specs/view identities 完全匹配，64 full + 64 local 批次；各自实际完成 128 更新与1,536 终局请求。新推理 actor 精确等于各自128终点。',
          '', '独立逐条核对了 3,072 轨迹的角色、seed 派生、action/mask/logprob、reward=-J、LOO baseline/advantage 与加权目标；A primary 首 logprob 保留，1,024 个 forced 首动作不计 logprob；B 的1,536个 policy 首动作均有完整 logprob，三流 seed 独立、等权。两源均显式 detached LOO。每更新25个参数的完整前后裁剪梯度均有限且与范数、clip1、保存 Adam 增量一致；每32更新 checkpoint 的权重、Adam矩/step、RNG、缓存/请求预算和游标均连续。',
          '', '两组均128/128更新触发clip1、0次zero_signal；裁剪前范数A为79.778912–2466.050051、B为48.781604–1597.049654。由已保存梯度与freshAdam独立计算的权重增量最大绝对误差约5.96e-8。普遍裁剪是应纳入后续DRL设计解释的事实，但不能据此直接认定性能差异的原因。',
          '', '审查没有用模型重算 logits/梯度：这里确认的是绑定源码、保存的数学关系、完整张量与更新谱系；不能用这些软件事实证明规划变好。B 同时改变 forcing 与 mixture weights，不能单独归因于首动作。',
          '', '## 固定终点 DEV 结果',
          '', '54 请求=18例×SL/A/B；18例属于3个真实父布局，各6个尺寸/端点窗口，只有1个优化seed。主统计先在每父布局内对6例求平均成本，再计算相对收益 `(control-B)/control`，最后对3父布局等权平均。以下95%区间独立以 PCG64/11632026、20,000次父布局 bootstrap 复算；仅为描述性区间。',
          '', '| 比较 | 指标 | 父布局平均相对收益 | 95%描述区间 | 预登记正向条件 |', '|---|---|---:|---:|---|']
for c in comparisons:
    lines.append(f"| B vs {c['control']} | {'raw J' if c['metric']=='raw_J' else 'postH J'} | {100*c['mean']:+.6f}% | [{100*c['lower95']:+.6f}%, {100*c['upper95']:+.6f}%] | {'通过' if c['exploratory_positive'] else '未通过'} |")
lines += ['', '**两项 postH 条件均未通过，故候选判定 REVISE。** raw B>A 的结果不能代替最终共同H交付质量，也不能用“比A少退化”证明B优于共同起点。独立补算A对SL的raw父布局平均相对收益为−7.344875%（描述区间[−8.823322%,−5.624070%]），说明只看B对A会高估相对共同起点的进步。3簇bootstrap不能扩大为18独立布局或3优化seed的证据。',
          '', '同时，18例的绝对成本均值为：', '', '| 模型 | raw J | postH J | postH L | postH K |', '|---|---:|---:|---:|---:|']
for d in extra['absolute_case_means']:
    lines.append(f"| {d['model']} | {d['raw_J']:.9f} | {d['J']:.9f} | {d['L']:.9f} | {d['K']:.6f} |")
lines += ['', 'B 的 postH 绝对均值较 A 降低0.130061%、较 SL 降低0.186978%，所以不能写成“绝对平均J恶化”。绝对均值之比按各父布局基线成本隐式加权，与预登记的等权父布局相对收益不同；二者方向差异属聚合定义差异。不能事后改用绝对均值推翻原筛选。完整父布局、单例及尺寸/端点表见审计JSON及原结果CSV。',
          '', '## 资源与失败核对', '', '| 阶段 | 完成请求/更新 | 墙钟 / 上限 (s) | 状态 |', '|---|---|---:|---|']
phase_counts={'DEPLOYMENT':'756 requests / 0 updates','A128':'1,536 terminals / 128 updates','B128':'1,536 terminals / 128 updates','DEV54':'54 requests / 0 updates'}
for phase,d in checks['resources']['phases'].items():
    lines.append(f"| {phase} | {phase_counts[phase]} | {d['wall_s']:.6f} / {d['cap_s']} | COMPLETE |")
lines += ['', '训练+DEV机制额度为3,072+54=3,126，不与756部署额度混用。全部科学完成请求合计3,882；实际新导航435（训练0，使用已保存导航）、共同H3,558、native216；教师0。训练terminal cache两组均0命中、0失败，所有预扣账与完成一致；没有重抽、换seed、补抽或重置旧资源的保存证据。工程验收另账8项PASS、1.510073 s/120 s、真实optimizer/模型推理/导航/native/H/教师0；独立审查未重跑这些测试。',
          '', f"本次独立数值审计脚本墙钟 {numeric['audit_wall_s']:.6f} s；补充审计 {extra['audit_wall_s']:.6f} s。从首个审查UTC时钟至出具本报告约 {verdict['review_wall_s']:.1f} s，均为维护/审查账，新增科学调用0。", 
          '', '## 主张与下一步路由',
          '', '| 分项 | claim_supported | confidence | 判定 |', '|---|---|---|---|',
          '| actor精确质量及本机观测速度 | yes（限定上述测试范围） | high | GO |',
          '| B相对A的规划提升 | partial（只支持raw） | medium | REVISE |',
          '| B相对共同SL起点的注册提升 | no | medium | REVISE |',
          '| 广义DRL快速高质量/泛化/真实机器人认证 | no | high（明确未获证实） | REVISE |',
          '', '最小可用表述：'+revision,
          '', '保留精确actor制品、完整A/B原证据与负/混合结果，关闭本次有界小试。下一步先围绕raw与postH不一致、普遍裁剪和混合采样机制修订DRL设计，再准备独立优化seed与真实父布局验证；如果要做forcing因果解释，应另行设计单因素/因子对照。该路由不执行新训练，也不授予旧S3或旧60权限。研究目标继续是大规模复杂父场景DRL训练、冻结后完整局部规划，监督只作初始化/辅助/对照。',
          '', '真实标定、IK、机械臂扫掠路径与关节轨迹认证仍缺；本轮只核保存的代理same-arm F&A、导航与完整任务覆盖验证，不能提升为真实机械臂认证。',
          '', '独立可复核产物：[主数值审计](paper/review-traces/experiment-result-to-claim/20261109_run01/independent_numeric_audit.json)、[补充聚合/资源核对](paper/review-traces/experiment-result-to-claim/20261109_run01/additional_saved_data_checks.json)、[完整审查轨迹](paper/review-traces/experiment-result-to-claim/20261109_run01/TRACE_MANIFEST.json)。',
          '']
report='\n'.join(lines)
write(HOME/'INDEPENDENT_REVIEW.md',report)
write(HOME/'INDEPENDENT_REVIEW.json',verdict)
write(TRACE/'raw_response.md',report)
write(TRACE/'parsed_verdict.json',verdict)
write(TRACE/'routing_action.json',routing)
write(TRACE/'pending_external_review.json',dict(pendingExternalReview=False,reviewer_role='independent delegated paper architect/reviewer',
     satisfied=True,paper_claim_audit=numeric['paper_claim_audit'],writing_verdict=numeric['downstream_writing']))
write(TRACE/'findings.md','# Research findings\n\n- Experiment: rc3_actor_pg_screen_20261109_01.\n- Intended claim: exact actor-only lifecycle speedup and matched PG improvement under the fixed DRL objective.\n- Verdict: partial; confidence high for scoped audited facts.\n- Actor artifacts: yes, exact and observed faster within the fixed deployment test.\n- B vs A: partial, raw only; postH gate fails.\n- B vs common SL: no registered improvement.\n- Overall DRL generalization/convergence/real-arm claim: not established.\n- Routing: revise DRL design and supplement only under future explicit stage authority; no new science in review.\n- Paper claim audit unavailable; downstream writing provisional.\n- pendingExternalReview: false.\n')
inputs=['PROTOCOL.json','AUTHORIZATION.json','SOURCE_BINDING.json','INPUT_REGISTRY.json','DEPLOYMENT_GRID.json','ENGINEERING_VERIFICATION.json',
        'IMPLEMENTATION_REVIEW.md','FROZEN_PRIOR_EVIDENCE.json','actors/ACTOR_INDEX.json','results/RESOURCE_ACCOUNTING.json','results/MECHANISM_COMPARISONS.json',
        'results/DEV_MODEL_SUMMARY.csv','results/STARTUP_AND_FIRST_RESPONSE.json']
artifacts={p.name:dict(path=p.relative_to(HOME).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in TRACE.iterdir() if p.is_file() and p.name!='TRACE_MANIFEST.json'}
for p in [HOME/'INDEPENDENT_REVIEW.md',HOME/'INDEPENDENT_REVIEW.json']:
    artifacts[p.name]=dict(path=p.relative_to(HOME).as_posix(),sha256=sha(p),bytes=p.stat().st_size)
manifest=dict(schema='rc3-review-trace-manifest-v1',skill='experiment-result-to-claim',trace='full',reviewer='independent delegated reviewer',
    prompt='prompt.md',raw_response='raw_response.md',parsed_verdict='parsed_verdict.json',routing_action='routing_action.json',
    pendingExternalReview=False,created_utc=now.isoformat(),paper_claim_audit=numeric['paper_claim_audit'],new_scientific_calls=0,
    inputs={name:dict(sha256=sha(HOME/name),bytes=(HOME/name).stat().st_size) for name in inputs},artifacts=artifacts,
    manifest_self_hash_excluded=True,source_modules_imported=False)
write(TRACE/'TRACE_MANIFEST.json',manifest)
print(json.dumps(dict(claim_supported='partial',technical='GO',candidate='REVISE',pendingExternalReview=False,trace_artifacts=len(artifacts)),ensure_ascii=False))
