"""Freeze the source/decision and maintain Git-visible reports, no grid execution."""
import json
import subprocess
import time
from datetime import datetime
from rc3_s3_evaluator.common import ROOT,read,digest,label,write_new,now,require
from rc3_s3_connectivity_20261109.scan import HOME

VERSION='2026.11.09-s3-connectivity-target-scope-ready'
BASE='c4219abe6838a9d5416cb311c09a0565ef21ac8b'


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT).decode('utf8').strip()


def freeze():
    old=read(ROOT/'rc3_s3_recovery_20261108/SOURCE_BINDING_02.json')
    outer=read(HOME/'independent/REGRESSION_RESULT.json');child=read(HOME/'independent/CHILD_RESULT.json')
    review=read(HOME/'INDEPENDENT_REVIEW.json')
    require(outer['status']==child['status']=='PASS' and child['tests_run']==8 and child['checks']==6067,'Regression not passed')
    write_new(HOME/'REGRESSION_RESULT.json',dict(schema='rc3-s3-scope-independent-regression-summary-v1',
        status='PASS',tests=child['tests_run'],assertions=child['checks'],outer_wall_s=outer['elapsed_s'],
        nested_child_wall_s=child['elapsed_s'],time_rule='Outer includes child; never sum nested durations',
        exact_registration_sha256=outer['registration_sha256'],source_snapshot_consistent=outer['source_snapshot_consistent'],
        observed_scientific_function_entries=outer['scientific_calls'],formal_S3_requests=0,
        frozen_files_checked=2515,old_semantics_comparisons=18,
        independent_outer_result_sha256=digest(HOME/'independent/REGRESSION_RESULT.json'),
        independent_child_result_sha256=digest(HOME/'independent/CHILD_RESULT.json')))
    sources=dict(old['source_sha256'])
    for p in HOME.glob('*.py'):sources[label(p)]=digest(p)
    source=dict(schema='rc3-s3-target-scope-execution-source-r1',baseline_commit=BASE,
        experiment_id='rc3_v3_global_local_20261106_01',protocol_sha256=old['protocol_sha256'],
        registry_sha256=old['registry_sha256'],H_identity=old['H_identity'],
        training_source_sha256=old['training_source_sha256'],training_model_provenance_separate=True,
        original_evaluator_sha256=old['old_evaluation_source_sha256'],
        original_persistence_source_binding_sha256=digest(ROOT/'rc3_s3_recovery_20261108/SOURCE_BINDING_02.json'),
        model_index_sha256=old['model_index_sha256'],environment=old['environment'],
        input_contract_amendment_sha256=digest(HOME/'INPUT_SCOPE_AMENDMENT.json'),
        frozen_evidence_index_sha256=digest(HOME/'FROZEN_EVIDENCE_INDEX.json'),
        baseline_import_sha256=digest(HOME/'BASELINE_IMPORT.json'),
        regression_result_sha256=digest(HOME/'REGRESSION_RESULT.json'),
        diagnostic_witness_sha256=digest(HOME/'CONNECTIVITY_WITNESS.json'),
        source_sha256=dict(sorted(sources.items())),
        scope='Uniform S3 input target-scope contract and retained-ledger continuation only; original model/MDP/reward/H/native/training source untouched')
    write_new(HOME/'SOURCE_BINDING.json',source)
    evidence=['INPUT_SCOPE_AMENDMENT.json','BASELINE_IMPORT.json','FROZEN_EVIDENCE_INDEX.json',
        'CONNECTIVITY_WITNESS.json','CONNECTIVITY_GEOMETRY_EVIDENCE.json','diagnosis/FINAL.json',
        'REGRESSION_RESULT.json','INDEPENDENT_REVIEW.json']
    decision=dict(schema='rc3-s3-042-one-time-scope-recovery-decision-r1',utc=now(),
        status='IMPLEMENTED_VERIFIED_READY_NOT_EXECUTED',
        authority_basis='Existing real S3 stage authorization plus actual user2026-11-09 conditionalA instruction to implement uniform target-scope repair and explicit retained-ledger recovery; this task performs diagnosis/regression only, not remaining scientific grid',
        user_pre_read_unknown_generated_source_hash_claim=False,
        original_v3_design_reapproved=False,original_protocol_rewritten=False,
        original_stage_authorization_sha256='08d5631b13acf95d73a6854789a5d08c8897714a789e1e7ff928a7eb4bf26a8e',
        source_binding_sha256=digest(HOME/'SOURCE_BINDING.json'),
        input_scope_amendment_sha256=digest(HOME/'INPUT_SCOPE_AMENDMENT.json'),
        evidence_sha256={label(HOME/p):digest(HOME/p) for p in evidence},
        selected_evidence_branch='A_OUTSIDE_TARGET_ONLY',refill_failed_slots=False,
        inherited_debits=2059,inherited_success=2057,inherited_failed_slots=2,
        inherited_first_seed_budget_wall_s=2896.9277158999903,inherited_original_exact_wall_s=None,
        remaining_requests=[3233,5292,5292],remaining_wall_s=[11503.07228410001,14400,14400],
        diagnostic_fees_separate=True,new_paid_refill_attempts=0,
        previous_two_failure_slots='Retain as missing, disposed for continuing OTHER unexecuted IDs only; do not create model results for failed requests',
        scientific_impact='Input admission changed to v3+scope_r1; same old accepted local inputs by original extractor and18saved comparisons. Retain explicit version tags and incomplete original-grid limitations.',
        current_task_executed_grid_requests=0,default_entry='python -B -m rc3_s3_connectivity_20261109.resume',
        explicit_future_execute='python -B -m rc3_s3_connectivity_20261109.resume --execute')
    write_new(HOME/'RECOVERY_DECISION.json',decision)
    print(dict(source_binding_sha256=decision['source_binding_sha256'],recovery_decision_sha256=digest(HOME/'RECOVERY_DECISION.json'),source_files=len(sources)))


def entry_check():
    start=time.perf_counter()
    p=subprocess.run([str(ROOT/'rc3_learning/.venv/Scripts/python.exe'),'-B','-X','utf8','-m','rc3_s3_connectivity_20261109.resume'],cwd=ROOT,capture_output=True,text=True,encoding='utf8',timeout=120)
    write_new(HOME/'ENTRY_CHECK.json',dict(utc=now(),status='PASS' if p.returncode==0 else 'FAIL',returncode=p.returncode,
        complete_source_budget_read_only_check_wall_s=time.perf_counter()-start,stdout=p.stdout,stderr=p.stderr,
        scientific_grid_requests=0,new_navigation_calls=0,model=0,native=0,H=0,optimizer=0,execute_flag_used=False))
    require(p.returncode==0,'Default production entry check failed')
    print('Default production source/budget entry check PASS; --execute was not used.')


def reports():
    bind=digest(HOME/'SOURCE_BINDING.json');reg=read(HOME/'REGRESSION_RESULT.json');diag=read(HOME/'diagnosis/FINAL.json')
    prep=read(HOME/'PREPARATION_RESOURCE.json');search=read(HOME/'SEARCH_RESOURCE.json');ledger=read(HOME/'LEDGER_IMPORT_RESOURCE.json');geo=read(HOME/'GEOMETRY_RESOURCE.json');entry=read(HOME/'ENTRY_CHECK.json')
    resource=dict(schema='rc3-s3-042-diagnosis-and-scope-repair-resource-v1',date='2026-11-09',
        current_task_actual_navigation_calls=1,current_task_formal_S3_requests=0,
        new_task_generation=0,new_FA_generation=0,model_initialization_and_inference=0,native=0,H=0,teacher_queries=0,optimizer_updates=0,original60_semantic_reads=0,
        diagnostic=dict(preregistered_navigation_cap=1,actual_navigation=1,wall_cap_s=300,all_read_compute_save_wall_s=diag['all_diagnostic_wall_s'],
            navigation_function_wall_s=read(HOME/'CONNECTIVITY_WITNESS.json')['navigation_wall_s'],original_D_sha256=None,
            kind='DIAGNOSTIC_RECOMPUTATION_NOT_ORIGINAL_D_BYTE_RECOVERY',separate_from_formal_stage=True),
        maintenance_phases=dict(artifact_inventory_read_search=search,source_and_frozen_hash_registration=prep,
            read_only_ledger_import=ledger,read_compute_save_geometry_addendum=geo,
            independent_regression_outer_wall_s=reg['outer_wall_s'],independent_regression_calls=reg['observed_scientific_function_entries'],
            default_production_entry_check=entry),
        time_accounting_rule='Nested navigation/child times are diagnostics within outer phases, never added. Diagnosis/search/engineering/import/default-check/IO and Git are separate from retained formal S3 ledger.',
        pre_search_shell_inspection=dict(exact_combined_wall_s=None,scientific_calls=0,
            scope='Initial local source/Git inspection and two failed direct fetches preceded instrumented search. Tool-reported command durations exist in execution transcript; no exact combined monotonic stage clock was recorded and none is invented.'),
        observed_end_to_end_since_search_start_s=(datetime.fromisoformat(now())-datetime.fromisoformat(read(HOME/'SEARCH_START.json')['utc'])).total_seconds(),
        observed_envelope_scope='UTC observation includes human/model reasoning and network wait; not a compute or deployment latency estimate',
        inherited_formal_S3=dict(debits=2059,successful_results=2057,failed_slots=2,first_seed_budget_wall_s=2896.9277158999903,
            original_exact_wall_s=None,remaining_requests=[3233,5292,5292],remaining_wall_s=[11503.07228410001,14400,14400],
            new_formal_grid_requests=0,new_paid_refill_attempts=0,failure_refunds=0),
        source_binding_sha256=bind,independent_review_sha256=digest(HOME/'INDEPENDENT_REVIEW.json'))
    write_new(HOME/'RESOURCE_ACCOUNTING.json',resource)
    (HOME/'IMPLEMENTATION_REVIEW.md').write_text('''# 实施评审

**诊断与范围修订工程GO；恢复入口技术GO，正式剩余S3本轮未执行。** 仅S3局部目标范围契约修订，原S1/S2模型、环境动作/奖励、mask、端点、导航及H/原生求解器不变。唯一task221在窗口外，23目标任务均可达，符合用户条件A。

独立一次8项/6067断言通过，18旧保存局部输入语义相同、2515原冻结文件未变。新scope对所有案例/七方法共用，真实目标不可行仍硬拒绝；没有单案例算法补丁、全图裁剪或窗口外合法候选丢弃。

科学风险明确：S3可接纳状态范围扩大，不能称为纯工程无影响；新版本按v3+scope_r1记录。两个原失败缺失仍阻止完整网格结论。新/旧计时开销不假定相同。恢复默认只核来源/资源，实跑必须显式--execute；旧2059预扣和精确壁钟继承，不能更换目录/授权清账。原60仍封存。

依据：[诊断](S3_042_CONNECTIVITY_DIAGNOSIS.md)、[范围决定](INPUT_SCOPE_DECISION.md)、[独立审查](INDEPENDENT_REVIEW.md)、[恢复决定](RECOVERY_DECISION.md)。
''',encoding='utf8')
    (HOME/'EXPERIMENT_REPORT.md').write_text(f'''# 本轮实际执行报告

正式S3新增请求 **0**，训练/模型/native/H/教师/optimizer **0**。只对保存S3_042做了事前登记的 **1次冻结navigation**，完整{diag['all_diagnostic_wall_s']:.9f}秒，函数{resource['diagnostic']['navigation_function_wall_s']:.9f}秒；没有替换scene/F/A，也不伪称原D恢复。

实际结果：全父不可服务集合{{221}}（table94，窗口外），本次23目标不可服务集合为空。候选752/753位于障碍物4的0.4米膨胀区，往返depot均∞；D/前驱/坐标身份和映射通过。已实现统一S3目标范围修订，保持原full-map上下文和目标硬检查。

一次独立回归8/8、6067断言，外层{reg['outer_wall_s']:.9f}秒，科学函数均0。18旧S0局部输入新旧语义相同；2515旧文件字节未变；两失败排除、缺钟/错来源/目录重置拒绝及父收尾时钟通过。默认生产入口真实--check通过，没有启动--execute。

原账保持2059预扣/2057成功/2失败、2896.9277158999903秒预算账；剩余3233/5292/5292请求和11503.07228410001/14400/14400秒，不退款/补跑。完整S3性能门槛仍NOT_EVALUATED_INCOMPLETE。

新源码绑定 `{bind}`；科学影响是S3输入接纳范围修订，而非模型、奖励、H或训练变更。核心证据：[诊断](S3_042_CONNECTIVITY_DIAGNOSIS.md)、[数字证据](CONNECTIVITY_WITNESS.json)、[资源](RESOURCE_ACCOUNTING.json)、[回归](REGRESSION_RESULT.json)、[独立审查](INDEPENDENT_REVIEW.md)、[恢复](RECOVERY_DECISION.md)。
''',encoding='utf8')
    (HOME/'RESULT_TO_CLAIM.md').write_text('''# 结果到主张

独立reviewer总体partial/high：具体S3_042可服务性诊断、条件A统一输入契约实现和保留原账的工程命题yes；完整S3性能、独立DRL增量、泛化与部署优势仍no/incomplete，不能判为模型失败。无paper claim audit，论文写作判断provisional。

支持：同一保存F/A和标记的诊断D下，目标23任务均可服务，唯task221/table94在窗口外被安全导航过滤。新guard保留全部目标、完整地图/上下文/端点及边界服务候选；真目标不可行仍拒绝。18旧输入语义相同，2515旧文件保持，原账及两缺失不被清零。

不支持：原D逐字节恢复、其他未生成目标必然可行、神经规划质量/时间收益、总体S3 AND门槛通过或真实关节轨迹认证。接纳契约已改，不能称为未修订v3完整实验或完全无科学影响。8项软件回归不是模型性能实验。

路由：冻结scope_r1源码和一次恢复决定；实际续段按原序只执行未消费请求，保存版本标签和两个失败缺失，不训练/换权重/换场景/A/删任务。原研究目标仍为大规模父场景DRL训练、冻结模型局部完整约束规划。

依据：[独立审查](INDEPENDENT_REVIEW.md)及[审查trace](paper/review-traces/experiment-result-to-claim/20261109_run01/)。
''',encoding='utf8')
    (HOME/'findings.md').write_text('''# 2026-11-09 findings

Intended claim: identify S3_042 full-parent rejection, uniformly enforce actual S3 targets, retain original ledgers and failures. Independent verdict partial/high: narrow diagnosis/engineering yes, full S3 performance not evaluated. Failure is outside-window task221/table94; target23 serviceable. Input admission truly amended; no model/reward/H/training changes. Diagnosis used one paid frozen navigation call, not zero-call or original-D recovery. One8-test/6067-assertion regressionPASS, old2515 files and18local input semantics preserved. Route: freeze scope_r1/retained-ledger continuation, no reseeding/refill/training, retain original two missing slots and incomplete scientific gates.
''',encoding='utf8')
    (HOME/'NEXT_STEP_DECISION.md').write_text(f'''# 下一步决定

**GO：诊断和最小目标范围修订；GO：恢复入口技术验收。正式剩余S3未在本轮启动，性能结论仍INCOMPLETE。**

准确新源码绑定`{bind}`；条件A证据成立，窗口23目标全部可服务，阻断task221/table94在外。已保存一份scope_r1契约与恢复处置，不重批未变v3，不覆盖任何旧版本。

未来显式--execute沿原序执行剩余3233/5292/5292请求，使用原冻结模型和七方法，继承2896.9277158999903秒及两失败。新目录追加续段，旧结果不重算、旧失败不补跑。诊断D不作为免费正式导航缓存。新目标若真不可行，继续硬拒绝/收费停止，不能利用本次“窗口外”例外跳过它。

两个原槽缺失、输入契约变化及计时版本必须分别报告，不填值、不删格、不宣称完整原v3或总体AND通过。保持固定DRL研究目标、预算/门槛/Registry/A/权重；原60封存，不追加训练或合并main。
''',encoding='utf8')
    (HOME/'README.md').write_text(f'''# S3_042连接诊断与目标范围修订

**窗口可行：23目标均可服务；唯一阻断是窗口外task221/table94。工程与恢复入口技术GO，本轮没有执行剩余S3。** 实际诊断1次navigation/{diag['all_diagnostic_wall_s']:.6f}秒；独立8项6067断言通过，其他科学函数0。原2059/2057/2与精确预算账保留，两个失败不补跑。

- [具体诊断](S3_042_CONNECTIVITY_DIAGNOSIS.md)、[逐任务候选与导航证据](CONNECTIVITY_WITNESS.json)、[安全膨胀几何](CONNECTIVITY_GEOMETRY_EVIDENCE.json)
- [诊断D/坐标/服务节点/前驱](diagnosis/navigation.npz)及[导航身份](diagnosis/navigation.json)；原D哈希未知，明确为诊断重算
- [输入契约决定](INPUT_SCOPE_DECISION.md)与[独立修订](INPUT_SCOPE_AMENDMENT.json)
- [独立回归](REGRESSION_RESULT.json)、[独立审查](INDEPENDENT_REVIEW.md)
- [一次恢复处置](RECOVERY_DECISION.md)、[机器处置](RECOVERY_DECISION.json)、[新源码绑定](SOURCE_BINDING.json)
- [资源账](RESOURCE_ACCOUNTING.json)、[实际报告](EXPERIMENT_REPORT.md)、[结果主张](RESULT_TO_CLAIM.md)、[下一步](NEXT_STEP_DECISION.md)

源码SHA256：`{bind}`。默认`python -B -m rc3_s3_connectivity_20261109.resume`仅核验；显式`--execute`才运行网格。契约确有科学影响，后续标记v3+scope_r1；完整原网格及总体部署门槛不因工程通过而成立。原60与全部旧证据保持。
''',encoding='utf8')
    print(dict(source_binding=bind,resource_navigation=1,formal_grid_requests=0))


def metadata():
    bind=digest(HOME/'SOURCE_BINDING.json')
    p=ROOT/'rc3_global_local_execution_v3/ACTIVE_EXECUTION_STATUS.json';s=read(p)
    s.update(status='APPROVED_V3_S3_TARGET_SCOPE_R1_IMPLEMENTED_READY_NOT_EXECUTED',
        S3_runtime='PAUSED_RETAINED_LEDGER_TARGET_SCOPE_R1_READY',S3_execution_readiness='GO_SCOPE_R1_TECHNICAL_ENTRY_CHECK',
        S3_current_report=label(HOME/'README.md'),S3_input_scope_amendment=label(HOME/'INPUT_SCOPE_AMENDMENT.json'),
        S3_input_scope_amendment_sha256=digest(HOME/'INPUT_SCOPE_AMENDMENT.json'),
        S3_scope_source_binding=label(HOME/'SOURCE_BINDING.json'),S3_scope_source_binding_sha256=bind,
        S3_scope_recovery_decision=label(HOME/'RECOVERY_DECISION.json'),S3_scope_recovery_decision_sha256=digest(HOME/'RECOVERY_DECISION.json'),
        S3_scope_revision_basis='Actual2026-11-09 conditionalA repair task; no claim user pre-read generated SHA; original v3 protocol approval unchanged',
        S3_scope_scientific_admission_changed=True,S3_scope_formal_grid_executed_this_turn=0,
        S3_scope_default_entry_checked=True,S3_scope_diagnostic_navigation_calls=1,
        S3_scope_original_D_recovery_claim=False,S3_scope_target042_serviceable=True,
        S3_scope_failure_task_outside_target=221,S3_scope_remaining_requests=[3233,5292,5292],
        S3_scope_remaining_wall_s=[11503.07228410001,14400,14400],
        S3_scope_old_failed_slots_refilled=False,S3_scope_prior_failure_disposition='Old full-parent rejection retained missing; only other unexecuted requests may proceed under explicit scope_r1',
        S3_scope_models_training_H_changed=False,S3_scientific_decision='NOT_EVALUATED_INCOMPLETE')
    save(p,s)
    p=ROOT/'PROJECT_VERSION.json';v=read(p);v['version']=VERSION
    v['S3_connectivity_target_scope']=dict(report=label(HOME/'README.md'),source_binding_sha256=bind,
        target042_serviceable=True,full_parent_unservable_task_ids=[221],blocker_outside_window=True,
        navigation_calls=1,formal_S3_requests=0,independent_tests=8,assertions=6067,
        original_files_unchanged=2515,input_admission_amended=True,original60='SEALED',main_merged=False)
    save(p,v)
    p=ROOT/'README.md';txt=p.read_text(encoding='utf8')
    txt=txt.replace('2026.11.08-s3-persistence-recovery-input-stop',VERSION,1)
    marker='<!-- S3_CONNECTIVITY_CURRENT_BEGIN -->'
    if marker not in txt:
        paragraph='''<!-- S3_CONNECTIVITY_CURRENT_BEGIN -->
**S3_042目标窗口可行，统一目标范围修订和恢复入口技术GO；本轮未启动剩余S3。** 事前登记后实际1次冻结导航诊断，完整2.833479秒；全父唯一阻断task221/table94在窗口外，752/753候选落入0.4米膨胀障碍区，本次23目标均可服务。保留全父诊断、完整地图/上下文/F&A与全部合法边界候选，实际目标仍硬检查；这改变S3输入接纳契约，后续标记v3+scope_r1。独立一次8项/6067断言通过，18旧保存输入语义相同、2515旧文件原字节保持。原2059预扣/2057成功/2失败与2896.9277159秒继承，剩余3233/5292/5292请求，两失败不补跑，原60未读、训练0。见[具体诊断与全部交付](rc3_s3_connectivity_20261109/README.md)、[范围修订](rc3_s3_connectivity_20261109/INPUT_SCOPE_DECISION.md)、[恢复处置](rc3_s3_connectivity_20261109/RECOVERY_DECISION.md)和[资源账](rc3_s3_connectivity_20261109/RESOURCE_ACCOUNTING.json)。下面是原停止和恢复的历史记录。
<!-- S3_CONNECTIVITY_CURRENT_END -->

'''
        idx=txt.index('<!-- S3_RECOVERY_CURRENT_BEGIN -->');txt=txt[:idx]+paragraph+txt[idx:]
    p.write_text(txt,encoding='utf8')
    p=ROOT/'EXPERIMENT_RESULT_TO_CLAIM.md';txt=p.read_text(encoding='utf8')
    title='## 2026-11-09：S3_042诊断与目标范围修订'
    if title not in txt:
        entry='''## 2026-11-09：S3_042诊断与目标范围修订

- 独立reviewer判定partial/high：具体诊断/工程命题yes，完整S3质量/部署仍no/incomplete，不解释为模型性能失败。
- 收费1次冻结navigation，原保存F/A与诊断D下，目标23任务可服务，唯一全父阻断task221/table94位于窗口外。旧D哈希未知，明确是诊断重算；不修改场景/A/目标/安全间隙。
- 统一scope_r1保留全父诊断、实际目标硬检查、完整上下文和边界候选；确实改变输入接纳契约，不是完全无科学影响。原训练/网络/奖励/H/权重不变。
- 一次8测试/6067断言通过，18旧局部输入语义相同，2515冻结文件保持；实际S3请求0。两失败槽保留，原账与剩余预算不清零，无完整原网格或总体AND性能结论。
- 路由：按准确新绑定/一次恢复处置继续原顺序未执行请求，不重批未改v3、不补值/换seed/重训。依据：[诊断与报告](rc3_s3_connectivity_20261109/README.md)、[独立审查](rc3_s3_connectivity_20261109/INDEPENDENT_REVIEW.md)。

'''
        txt=txt.replace('# 实验结果到主张记录\n\n','# 实验结果到主张记录\n\n'+entry,1);p.write_text(txt,encoding='utf8')


def audit_and_manifest():
    changed=git('diff','--name-only',BASE,'--','rc3_s3_evaluator','rc3_s3_recovery_20261108','rc3_s1a_results_20261106',
        'rc3_s1b_results_20261106','rc3_s2_results_20261107','rc3_global_local_execution_v3/runs',
        'rc3_global_local_execution_v3/authorizations','rc3_global_local','rc3_global_local_trainer','rc3_navigation_compat_20261106')
    require(not changed,'Old frozen source/science changed')
    write_new(HOME/'GIT_VERSION_AUDIT.json',dict(utc=now(),baseline=BASE,branch=git('branch','--show-current'),
        prepublication_HEAD=git('rev-parse','HEAD'),fetched_remote_research=git('rev-parse','origin/codex/global-local-preflight'),
        fetched_main=git('rev-parse','origin/main'),main_merge_base=git('merge-base','HEAD','origin/main'),
        S0=git('rev-parse','codex/global-local-s0'),S0_merge_base=git('merge-base','HEAD','codex/global-local-s0'),
        old_frozen_science_changes=[],unrelated_Word_files='PRESERVED_NOT_STAGED',
        remote_fetch='Two direct attempts failed TCP connection; standard git fetch origin then succeeded with normal inherited transport environment. No Git global settings changed.',
        publication_status='PENDING_PAYLOAD_AT_AUDIT_CREATION',publication_receipt='project_management/s3_connectivity_delivery/PUBLICATION_RECEIPT.json',
        force_push=False,main_merged=False))
    manifest()


def manifest():
    p=ROOT/'project_management/GIT_MANIFEST.json';m=read(p)
    exclusion='project_management/s3_connectivity_delivery/PUBLICATION_RECEIPT.json'
    if exclusion not in m['excluded_self_referential_metadata']:m['excluded_self_referential_metadata'].append(exclusion)
    targets=[ROOT/n for n in ('README.md','PROJECT_VERSION.json','EXPERIMENT_RESULT_TO_CLAIM.md','rc3_global_local_execution_v3/ACTIVE_EXECUTION_STATUS.json')]
    for base in (HOME,ROOT/'project_management/s3_connectivity_delivery'):
        targets.extend(q for q in base.rglob('*') if q.is_file() and '__pycache__' not in q.parts)
    for q in targets:
        name=label(q)
        if name not in m['excluded_self_referential_metadata']:m['files'][name]=dict(sha256=digest(q),bytes=q.stat().st_size)
    m['files']=dict(sorted(m['files'].items()));m['file_count']=len(m['files']);m['total_bytes']=sum(x['bytes'] for x in m['files'].values());m['version']=VERSION
    m['S3_connectivity_scope_r1']=dict(baseline=BASE,report=label(HOME/'README.md'),source_binding_sha256=digest(HOME/'SOURCE_BINDING.json'),
        target042_serviceable=True,full_parent_unservable=[221],scope_changed=True,diagnostic_navigation=1,
        formal_S3_requests_this_turn=0,regression='PASS_8_TESTS_6067_ASSERTIONS',original_files_preserved=2515,
        scope='Only new diagnosis/amendment/delivery files and explicit mutable metadata refreshed; old science and unrelated live Word entries retained',main_merged=False)
    save(p,m)


if __name__=='__main__':
    import sys
    {'freeze':freeze,'check':entry_check,'reports':reports,'metadata':metadata,'audit':audit_and_manifest,'manifest':manifest}[sys.argv[1]]()
