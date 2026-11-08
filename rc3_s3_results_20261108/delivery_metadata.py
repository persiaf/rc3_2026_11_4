"""Scoped S3 incident delivery metadata. Never edits approved evaluator/run evidence."""
import json
import subprocess
import sys
from .audit_saved import OUT,RUN
from rc3_s3_evaluator.common import *


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT).decode('utf8').strip()


def save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def status():
    a=read(OUT/'SAVED_RUN_AUDIT.json');r=read(OUT/'RESOURCE_ACCOUNTING.json')
    r['postmortem_engineering']=dict(
        read_only_audit_attempts=2,first_audit_error='datetime name collision in unbound postprocessing helper, fixed without run-file changes',
        synthetic_fault_registration='rc3_s3_results_20261108/IO_FAULT_REPRO_REGISTRATION.json',
        synthetic_fault_result='rc3_s3_results_20261108/IO_FAULT_REPRO_RESULT.json',
        synthetic_fault_elapsed_s=read(OUT/'IO_FAULT_REPRO_RESULT.json')['elapsed_s'],synthetic_fault_checks=5,
        independent_review='rc3_s3_results_20261108/INDEPENDENT_REVIEW.json',
        new_actual_requests=0,new_target_generation=0,new_navigation=0,new_native=0,new_H=0,new_inference=0,new_optimizer=0)
    save(OUT/'RESOURCE_ACCOUNTING.json',r)
    p=ROOT/'rc3_global_local_execution_v3/ACTIVE_EXECUTION_STATUS.json';v=read(p)
    v.update(status='APPROVED_V3_S3_STOPPED_JOURNAL_IO_FAULT',S3_runtime='STOPPED_IO_FAILURE',
        S3_authorized=True,S1_B_S2_S3_authorized=True,S1_B_S2_S3_authorized_scope='S1-B/S2 completed; S3 explicitly authorized but stopped with dirty IO state',
        S3_execution_readiness='REVISE_JOURNAL_IO_AND_EXPLICIT_RECOVERY',S3_scientific_decision='NOT_EVALUATED_INCOMPLETE',
        S3_requests_debited=163,S3_requests_completed=162,S3_requests_failed=1,
        S3_seeds_completed=[],S3_seeds_not_started=list(SEEDS[1:]),S3_unique_materialized_target_cases=4,
        S3_target_tasks='FOUR_AUTHORIZED_CASES_MATERIALIZED_REMAINING_NOT_GENERATED',S3_recovery_authorized=False,
        S3_original_head_pending_preserved=True,S3_exact_stage_wall_s=None,S3_conservative_resource_reservation_s=592,
        S3_report='rc3_s3_results_20261108/README.md',original60='SEALED')
    save(p,v)
    p=ROOT/'PROJECT_VERSION.json';v=read(p)
    v['version']='2026.11.08-s3-interrupted-io-failure'
    v['S3_scientific_execution']=dict(status='STOPPED_IO_FAILURE',decision='REVISE',scientific_decision='NOT_EVALUATED_INCOMPLETE',
        experiment_id=EXPERIMENT,report='rc3_s3_results_20261108/README.md',source_binding_sha256=digest(HERE/'S3_SOURCE_BINDING.json'),
        requests_debited=163,complete=162,failed=1,optimizer_updates=0,original60='SEALED',main_merged=False)
    # Preserve earlier metadata byte formatting except the current version and new phase block.
    raw=p.read_text(encoding='utf8');old_version=read(p)['version']
    raw=raw.replace(old_version,v['version'],1)
    pos=raw.rfind('}');block=json.dumps({'S3_scientific_execution':v['S3_scientific_execution']},ensure_ascii=False,indent=2)[1:-1].strip('\n')
    raw=raw[:pos].rstrip()+',\n'+block+'\n}'+raw[pos+1:]
    require(json.loads(raw)==v,'Version metadata append changed earlier fields')
    p.write_text(raw,encoding='utf8',newline='')


def audit():
    changed=git('diff','--name-only','b9f2fe2cd891b0e6582fcd441b879e7e7d4c6a02','--',
        'rc3_global_local_trainer','rc3_global_local','rc3_navigation_compat_20261106','rc3_s1a_results_20261106',
        'rc3_s1b_results_20261106','rc3_s2_results_20261107','rc3_global_local_execution_v3/runs',
        'rc3_global_local_execution_v3/authorizations')
    require(changed=='','Old scientific evidence modified')
    # Approved evaluator files (including the two older source bindings) remain original.
    tracked=git('ls-files','rc3_s3_evaluator').splitlines()
    for n in tracked:
        expected=subprocess.check_output(['git','show','b9f2fe2cd891b0e6582fcd441b879e7e7d4c6a02:'+n],cwd=ROOT)
        require((ROOT/n).read_bytes()==expected,'Approved evaluator evidence changed: '+n)
    original=read(OUT/'ARTIFACT_INDEX.json')['artifacts']
    require(all(digest(project_file(e['path']))==e['sha256'] for e in original),'Original failed run changed')
    write_new(OUT/'GIT_VERSION_AUDIT.json',dict(schema='rc3-s3-formal-git-audit-v1',utc=now(),
        baseline_commit='b9f2fe2cd891b0e6582fcd441b879e7e7d4c6a02',authorization_commit=git('rev-parse','74add854'),
        branch=git('branch','--show-current'),remote_research_before_publication=git('rev-parse','origin/codex/global-local-preflight'),
        remote_main_before_publication=git('rev-parse','origin/main'),main_merge_base=git('merge-base','HEAD','origin/main'),
        S0_merge_base=git('merge-base','HEAD','codex/global-local-s0'),approved_evaluator_files_byte_equal=len(tracked),
        original_failed_run_files_byte_equal=len(original),approved_source_binding_sha256=digest(HERE/'S3_SOURCE_BINDING.json'),
        old_scientific_paths_changed=[],user_Word_files='PRESERVED_NOT_STAGED',
        LFS_new_saved_input_NPZ_files=len(list((RUN/'inputs').rglob('*.npz'))),
        LFS_status='VERIFY_AFTER_PAYLOAD_COMMIT',push_status='PENDING',main_merged=False,force_push=False,
        publication_receipt='rc3_s3_results_20261108/PUBLICATION_RECEIPT.json'))


def manifest():
    p=ROOT/'project_management/GIT_MANIFEST.json';v=read(p)
    files=[ROOT/n for n in ['README.md','PROJECT_VERSION.json','EXPERIMENT_RESULT_TO_CLAIM.md','rc3_global_local_execution_v3/ACTIVE_EXECUTION_STATUS.json']]
    for folder in (OUT,RUN):files.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for f in files:v['files'][label(f)]=dict(sha256=digest(f),bytes=f.stat().st_size)
    v['files']=dict(sorted(v['files'].items()));v['file_count']=len(v['files']);v['total_bytes']=sum(e['bytes'] for e in v['files'].values())
    v['version']='2026.11.08-s3-interrupted-io-failure'
    v['S3_scientific_execution']=dict(status='STOPPED_IO_FAILURE',scientific_decision='NOT_EVALUATED_INCOMPLETE',
        report='rc3_s3_results_20261108/README.md',requests_debited=163,complete_requests=162,failed_requests=1,
        source_binding_sha256=digest(HERE/'S3_SOURCE_BINDING.json'),original_run_files_preserved=180,
        optimizer_updates=0,original60='SEALED',main_merged=False,
        scope='Only new S3 receipts/results/postmortem and explicit mutable status metadata refreshed; original approved evaluator and old science remain byte unchanged; unrelated Word release entries retained')
    save(p,v)


if __name__=='__main__':
    {'status':status,'audit':audit,'manifest':manifest}[sys.argv[1]]()
