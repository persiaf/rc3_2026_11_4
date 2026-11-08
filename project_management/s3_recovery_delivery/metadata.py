"""Scoped maintenance after actual S3 repair/continuation stop."""
import json
import subprocess
import sys
from rc3_s3_evaluator.common import ROOT, read, digest, label, now, write_new, require
from rc3_s3_recovery_20261108.authority import HOME, CONT, RUN

VERSION='2026.11.08-s3-persistence-recovery-input-stop'
BASE='596fbf3dc607ca14172b26ec11f813809c132f5a'


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT).decode('utf8').strip()


def current():
    r=read(HOME/'results/RESOURCE_ACCOUNTING.json')
    p=ROOT/'rc3_global_local_execution_v3/ACTIVE_EXECUTION_STATUS.json';s=read(p)
    s.update(status='APPROVED_V3_S3_RECOVERY_STOPPED_NEW_INPUT_FEASIBILITY_FAILURE',
        S3_runtime='RECOVERY_STOPPED_INPUT_FEASIBILITY_FAILURE',
        S3_execution_readiness='BLOCKED_NEW_INPUT_FAILURE_NOT_DISPOSED',
        S3_target_tasks='43_AUTHORIZED_CASES_MATERIALIZED_42_WITH_SUCCESSFUL_RESULTS_REMAINING65_NOT_GENERATED',
        S3_target_semantic_consumption=43,
        S3_target_semantic_consumption_unit='distinct S3 declarations whose target scene/F/A was materialized',
        S3_requests_debited=r['total_debits'],S3_requests_completed=r['complete_requests'],S3_requests_failed=r['failed_requests'],
        S3_requests_unexecuted=r['not_executed_requests'],S3_unique_materialized_target_cases=43,
        S3_report='rc3_s3_recovery_20261108/README.md',S3_scientific_decision='NOT_EVALUATED_INCOMPLETE',
        S3_recovery_remaining_requests=[3233,5292,5292],
        S3_recovery_remaining_wall_s=[r['actual_by_seed'][str(seed)]['remaining_wall_s'] for seed in (11622026,11622027,11622028)],
        S3_counter_snapshot_scope='Closed original plus continuation journals; original 163/162/1 remain frozen in rc3_s3_results_20261108 and BASELINE_IMPORT',
        S3_recovery_new_request_debits=1896,S3_recovery_new_success=1895,S3_recovery_new_failure=1,
        S3_recovery_charged_cumulative_wall_s=r['charged_all_stage_wall_s'],
        S3_recovery_exact_measured_new_session_wall_s=2295.1254143999977,
        S3_recovery_original_exact_stage_wall_s=None,
        S3_new_failed_request='88c202d54990cee9fe80d3b8a01fba6c7275a98069a822d3c25e9b701d7eff68',
        S3_new_failure_case='S3_042',S3_quality_gate=None,S3_deployment_gate=None,
        S3_persistence_engineering='GO_17_TESTS_AND_CLOSED_ACTUAL_FAILURE',
        S1_B_S2_S3_authorized_scope='S1-B/S2 complete; S3 original authorization remains valid; actual recovery stopped on a new undisposed input feasibility failure',
        working_tree_execution_sources='Approved training source 0f0ffd2 unchanged; original S3 b348 binding retained; scoped persistence source02 accepted and used; no new training',
        S3_recovery_source_hash='767f13bd4619bef37501db997296982baf6b1565cb25ea8cb964c2e39d381bea',
        S3_current_status_closed_utc=now())
    save(p,s)
    p=ROOT/'PROJECT_VERSION.json';v=read(p);v['version']=VERSION
    v['S3_persistence_recovery']=dict(status=s['S3_runtime'],report=s['S3_report'],
        baseline=BASE,source_binding_sha256=s['S3_recovery_source_hash'],
        old180_preserved=True,models_unchanged=12,new_debits=1896,new_success=1895,new_failure=1,
        total_debits=2059,total_success=2057,total_failure=2,unexecuted=13817,
        scientific_decision='NOT_EVALUATED_INCOMPLETE',optimizer_updates=0,original60='SEALED',main_merged=False)
    save(p,v)
    p=ROOT/'README.md';txt=p.read_text(encoding='utf8')
    txt=txt.replace('2026.11.08-s3-interrupted-io-failure',VERSION,1)
    marker='<!-- S3_RECOVERY_CURRENT_BEGIN -->'
    if marker not in txt:
        paragraph='''<!-- S3_RECOVERY_CURRENT_BEGIN -->
**S3持久化修复工程GO，已实际恢复1895个未执行请求；随后因新的S3_042输入不可行停止。** 累计2059预扣、2057有效结果、2失败，13817未执行。原180文件、旧HEAD/pending与12固定模型来源保持；原第163槽没有补跑。6次真实瞬态锚重试成功，新失败的Journal及父阶段时钟正常闭合。首种子预算账为592保守旧账＋9.8023启动失败＋2295.1254续段＝2896.9277秒，原准确墙钟仍null。后两种子未启动。质量294/2268、cold881/6804、hot882/6804；全部6质量/12部署格INCOMPLETE，不支持总体性能判断。新失败在全父场景F&A检查、模型/native/H之前，不改场景或免费重试；optimizer0，原60未读。见[真实恢复结果与全部交付](rc3_s3_recovery_20261108/README.md)、[累计资源](rc3_s3_recovery_20261108/results/RESOURCE_ACCOUNTING.json)、[请求状态](rc3_s3_recovery_20261108/results/REQUEST_STATE_INDEX.json)与[下一步决定](rc3_s3_recovery_20261108/NEXT_STEP_DECISION.md)。

以下原IO中断段落保留为历史前缀；当前状态以上述恢复记录和活动执行状态为准。
<!-- S3_RECOVERY_CURRENT_END -->

'''
        idx=txt.index('**正式 S3 已实际启动')
        txt=txt[:idx]+paragraph+txt[idx:]
    txt=txt.replace('正式S3现已授权且因IO中断，以上文本轮记录为准。','正式S3已授权且恢复后因新输入可行性失败停止，以上文本轮记录为准。')
    p.write_text(txt,encoding='utf8')


def audit():
    paths=['rc3_s1a_results_20261106','rc3_s1b_results_20261106','rc3_s2_results_20261107',
        'rc3_navigation_compat_20261106','rc3_global_local','rc3_global_local_trainer',
        'rc3_global_local_execution_v3/runs','rc3_global_local_execution_v3/authorizations',
        'rc3_s3_results_20261108']
    changed=git('diff','--name-only',BASE,'--',*paths)
    require(not changed,'Frozen scientific paths changed')
    original_py=git('diff','--name-only',BASE,'--','rc3_s3_evaluator/*.py','rc3_s3_evaluator/S3_*.json')
    require(not original_py,'Original evaluator binding/source changed')
    write_new(HOME/'GIT_VERSION_AUDIT.json',dict(schema='rc3-s3-recovery-git-audit-v1',utc=now(),
        baseline=BASE,branch=git('branch','--show-current'),local_prepublication_HEAD=git('rev-parse','HEAD'),
        fetched_remote_research=git('rev-parse','origin/codex/global-local-preflight'),
        fetched_main=git('rev-parse','origin/main'),main_merge_base=git('merge-base','HEAD','origin/main'),
        S0=git('rev-parse','codex/global-local-s0'),S0_merge_base=git('merge-base','HEAD','codex/global-local-s0'),
        ahead_behind_main=git('rev-list','--left-right','--count','origin/main...HEAD'),
        frozen_paths_changed=[],original_evaluator_changed=[],
        old180_and_12models='results/RECOVERY_AUDIT.json',
        allowed_changes=['rc3_s3_recovery_20261108','continuations/persistence_r1',
            'PERSISTENCE_RECOVERY_ACTIVATION.json','project_management/s3_recovery_delivery',
            'README.md','PROJECT_VERSION.json','EXPERIMENT_RESULT_TO_CLAIM.md',
            'rc3_global_local_execution_v3/ACTIVE_EXECUTION_STATUS.json','project_management/GIT_MANIFEST.json'],
        unrelated_Word_files='PRESERVED_NOT_STAGED',
        lfs_unicode_status_initial_failure='Git quoted a live unrelated Chinese Word path; core.quotepath=false status succeeded without modifying that file',
        fetch_transient_failure='First postrun direct fetch received connection reset; same command retry succeeded',
        payload_publication='PENDING_AT_AUDIT_CREATION',
        final_publication_receipt='project_management/s3_recovery_delivery/PUBLICATION_RECEIPT.json',
        force_push=False,main_merged=False))


def manifest():
    p=ROOT/'project_management/GIT_MANIFEST.json';m=read(p)
    exclude='project_management/s3_recovery_delivery/PUBLICATION_RECEIPT.json'
    if exclude not in m['excluded_self_referential_metadata']:m['excluded_self_referential_metadata'].append(exclude)
    targets=[ROOT/x for x in ('README.md','PROJECT_VERSION.json','EXPERIMENT_RESULT_TO_CLAIM.md',
        'rc3_global_local_execution_v3/ACTIVE_EXECUTION_STATUS.json')]
    for folder in (HOME,CONT,ROOT/'project_management/s3_recovery_delivery'):
        targets.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    targets.append(RUN/'PERSISTENCE_RECOVERY_ACTIVATION.json')
    for q in targets:
        name=label(q)
        if name in m['excluded_self_referential_metadata']:continue
        m['files'][name]=dict(sha256=digest(q),bytes=q.stat().st_size)
    m['version']=VERSION;m['files']=dict(sorted(m['files'].items()))
    m['scope']='Original LFS working-tree bytes retained. Current repair/paid-recovery status is S3_persistence_recovery; older phase objects retain their creation-time statuses. Unrelated live Word release entries are not refreshed.'
    m['file_count']=len(m['files']);m['total_bytes']=sum(x['bytes'] for x in m['files'].values())
    m['S3_persistence_recovery']=dict(technical='GO',scientific='NOT_EVALUATED_INCOMPLETE',
        runtime='STOPPED_NEW_INPUT_FEASIBILITY_FAILURE',baseline=BASE,
        source_binding_sha256='767f13bd4619bef37501db997296982baf6b1565cb25ea8cb964c2e39d381bea',
        new_request_debits=1896,new_success=1895,new_failure=1,cumulative_debits=2059,cumulative_success=2057,
        original180_preserved=True,original60='SEALED',optimizer_updates=0,main_merged=False,
        scope='Only repair/continuation/delivery files and explicit mutable current metadata refreshed; unrelated Word release entries and old science retained',
        report='rc3_s3_recovery_20261108/README.md')
    save(p,m)


if __name__=='__main__':
    {'current':current,'audit':audit,'manifest':manifest}[sys.argv[1]]()
