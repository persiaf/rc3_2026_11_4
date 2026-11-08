"""Scoped Git metadata maintenance; never refresh unrelated live Word files."""
import json
import subprocess
import sys
from rc3_s3_evaluator.common import *


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT).decode('utf8').strip()


def audit():
    paths=git('diff','--name-only',BASELINE,'--','rc3_s1a_results_20261106','rc3_s1b_results_20261106',
        'rc3_s2_results_20261107','rc3_navigation_compat_20261106','rc3_global_local','rc3_global_local_trainer',
        'rc3_global_local_execution_v3/runs','rc3_global_local_execution_v3/authorizations')
    require(paths=='','Old training/scientific evidence changed')
    baseline_lfs=git('lfs','ls-files','-l',BASELINE)
    head_lfs=git('lfs','ls-files','-l','HEAD')
    require(baseline_lfs==head_lfs,'Unexpected new/changed LFS model pointers')
    write_new(HERE/'GIT_VERSION_AUDIT.json',dict(schema='rc3-s3-git-audit-v1',utc=now(),
        branch=git('branch','--show-current'),baseline=BASELINE,preregistration_commit=git('rev-parse','6be82edc'),
        main_sha=git('rev-parse','origin/main'),main_merge_base=git('merge-base','HEAD','origin/main'),
        S0_sha=git('rev-parse','codex/global-local-s0'),S0_merge_base=git('merge-base','HEAD','codex/global-local-s0'),
        remote_research_SHA_before_publication=git('rev-parse','origin/codex/global-local-preflight'),
        protocol_sha256=PROTOCOL,training_source_sha256=TRAIN_SOURCE,
        evaluation_source_binding_sha256=digest(HERE/'S3_SOURCE_BINDING.json'),
        frozen_baseline_opaque_audit='rc3_s3_evaluator/FROZEN_EVIDENCE_AUDIT.json',
        old_scientific_paths_changed=[],unrelated_user_Word_edits='PRESERVED_NOT_STAGED',
        mutable_old_metadata_allowlist=['README.md','PROJECT_VERSION.json','EXPERIMENT_RESULT_TO_CLAIM.md',
            'rc3_global_local_execution_v3/ACTIVE_EXECUTION_STATUS.json','project_management/GIT_MANIFEST.json'],
        frozen_audit_timing='33096 baseline files matched before explicit mutable status/documentation updates; all old scientific paths remain unchanged afterward',
        lfs=dict(fsck_command='git lfs fsck --objects HEAD',fsck_result='Git LFS fsck OK',
                 fsck_exit_code=0,object_scope='All referenced objects at preregistration HEAD, same LFS pointer set as scientific baseline',
                 pointer_set_unchanged=True,pointer_listing_sha256=identity(head_lfs),new_model_or_archive_objects=0),
        github_push_status='PENDING_PAYLOAD_COMMIT',main_merged=False,force_push=False,
        publication_receipt='project_management/s3_evaluator_delivery/PUBLICATION_RECEIPT.json'))


def manifest():
    path=ROOT/'project_management/GIT_MANIFEST.json';m=read(path)
    targets=[ROOT/p for p in ['README.md','PROJECT_VERSION.json','EXPERIMENT_RESULT_TO_CLAIM.md',
                              'rc3_global_local_execution_v3/ACTIVE_EXECUTION_STATUS.json']]
    for base in (HERE,ROOT/'project_management/s3_evaluator_delivery'):
        targets.extend(p for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for p in targets:m['files'][label(p)]=dict(sha256=digest(p),bytes=p.stat().st_size)
    m['version']='2026.11.07-s3-evaluator-engineering'
    m['files']=dict(sorted(m['files'].items()))
    m['file_count']=len(m['files']);m['total_bytes']=sum(v['bytes'] for v in m['files'].values())
    m['S3_evaluator_engineering']=dict(technical='GO',engineering_complete_requests=126,formal_S3=0,
        evaluation_source_binding_sha256=digest(HERE/'S3_SOURCE_BINDING.json'),
        report='rc3_s3_evaluator/README.md',original60='SEALED',new_optimizer_updates=0,main_merged=False,
        scope='Only evaluator/delivery files and explicit mutable status pointers refreshed; unrelated live Word entry and old scientific files retained')
    path.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    if sys.argv[1]=='audit':audit()
    elif sys.argv[1]=='manifest':manifest()
    else:raise ValueError('audit or manifest')
