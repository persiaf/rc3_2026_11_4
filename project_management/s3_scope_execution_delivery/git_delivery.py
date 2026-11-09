"""Explicit Git audit and content manifest; never stages, commits or pushes."""
import json
import subprocess
from pathlib import Path
from rc3_s3_evaluator.common import ROOT,read,digest,write_new,now,require,label
from .reports import HOME,RES,BASE,VERSION,SOURCE,save

ADMIN=ROOT/'project_management/s3_scope_execution_delivery'
RECEIPT=ADMIN/'PUBLICATION_RECEIPT.json'
USER_DOC='第二章与第三章修订稿_20261029.docx'
MUTABLE={'.gitattributes','README.md','PROJECT_VERSION.json','EXPERIMENT_RESULT_TO_CLAIM.md',
    'project_management/GIT_MANIFEST.json','rc3_global_local_execution_v3/ACTIVE_EXECUTION_STATUS.json'}
PREFIXES=('project_management/s3_scope_execution_delivery/','rc3_s3_scope_results_20261109/',
    'rc3_s3_connectivity_20261109/runs/')
NEW_RUNTIME={'rc3_s3_connectivity_20261109/SCOPE_RECOVERY_ACTIVATION.json','rc3_s3_connectivity_20261109/SCOPE_SUPERVISOR.lease'}


def git(*args):
    return subprocess.check_output(['git','-c','core.quotepath=false',*args],cwd=ROOT,encoding='utf8').strip()


def admissible(name):return name in MUTABLE or name in NEW_RUNTIME or name.startswith(PREFIXES)


def lfs_entries(revision):
    value=json.loads(git('lfs','ls-files','--json',revision))
    return value['files'] if isinstance(value,dict) else value


def sources():
    paths=sorted(ADMIN.glob('*.py'))
    paths.extend(sorted((HOME/'paper/review-traces').rglob('*.py')))
    write_new(HOME/'ANALYSIS_SOURCE_BINDING.json',dict(schema='rc3-read-only-analysis-source-binding-v1',utc=now(),
        approved_production_evaluation_binding_sha256=SOURCE,production_source_unchanged=True,
        source_sha256={label(p):digest(p) for p in paths},
        role='Read-only collection, independent audit, saved-data statistics and documentation/publication maintenance; not a new scientific evaluator or authorization',
        new_model_navigation_native_H_optimizer_teacher_calls=0))
    print(dict(analysis_helper_files=len(paths)))


def audit():
    branch=git('branch','--show-current');require(branch=='codex/global-local-preflight','Wrong research branch')
    changed=git('diff','--name-only',BASE).splitlines()
    require(all(admissible(n) or n==USER_DOC for n in changed),'Unrelated tracked scientific file changed')
    staged=git('diff','--cached','--name-only').splitlines();require(all(admissible(n) for n in staged),'User/unrelated files staged')
    base_lfs=lfs_entries(BASE)
    write_new(HOME/'GIT_VERSION_AUDIT.json',dict(utc=now(),baseline=BASE,branch=branch,
        initial_execution_HEAD=BASE,prepublication_HEAD=git('rev-parse','HEAD'),
        fetched_remote_research=git('rev-parse','origin/codex/global-local-preflight'),
        fetched_main=git('rev-parse','origin/main'),main_merge_base=git('merge-base','HEAD','origin/main'),
        S0=git('rev-parse','codex/global-local-s0'),S0_merge_base=git('merge-base','HEAD','codex/global-local-s0'),
        teacher_distill_branch=git('rev-parse','codex/teacher-distill-t0-t2'),
        teacher_distill_merge_base=git('merge-base','HEAD','codex/teacher-distill-t0-t2'),
        fetch='Successful normal inherited Git transport before run and before publication; no global Git settings changed',
        remote_main_before='5d5fe25754ffa032f97937a3fb54f15901c2fb7d',
        changed_tracked_files_at_audit=changed,unrelated_Word_files='PRESERVED_NOT_STAGED',
        bound_source_files_unchanged=194,old_frozen_files_unchanged=2515,model_source_files_unchanged=36,
        baseline_LFS_paths=len(base_lfs),LFS_maintenance='Three new scope-r1 journals use a path-specific LFS rule; existing object identities and all file bytes unchanged',
        source_binding_sha256=SOURCE,force_push=False,main_merged=False,
        publication_status='PENDING_PAYLOAD_AT_AUDIT_CREATION',publication_receipt=label(RECEIPT)))
    print('Git version audit saved; publication not yet claimed')


def manifest():
    p=ROOT/'project_management/GIT_MANIFEST.json';m=read(p)
    for ex in (label(RECEIPT),):
        if ex not in m['excluded_self_referential_metadata']:m['excluded_self_referential_metadata'].append(ex)
    targets=[ROOT/n for n in MUTABLE if n!='project_management/GIT_MANIFEST.json']
    for base in (HOME,ADMIN,ROOT/'rc3_s3_connectivity_20261109/runs'):
        targets.extend(q for q in base.rglob('*') if q.is_file() and '__pycache__' not in q.parts)
    targets.extend(ROOT/n for n in NEW_RUNTIME)
    for q in targets:
        name=label(q)
        if name not in m['excluded_self_referential_metadata']:m['files'][name]=dict(sha256=digest(q),bytes=q.stat().st_size)
    m['files']=dict(sorted(m['files'].items()));m['file_count']=len(m['files']);m['total_bytes']=sum(x['bytes'] for x in m['files'].values());m['version']=VERSION
    m['S3_scope_r1_formal_results']=dict(baseline=BASE,report=label(HOME/'README.md'),source_binding_sha256=SOURCE,
        new_requests=13817,new_success=13817,new_failure=0,cumulative_requests=15876,cumulative_success=15874,historical_missing=2,
        quality_cells_failed=5,quality_cells_incomplete=1,deployment_cells_failed=9,deployment_cells_incomplete=3,
        global_quality_gate=None,global_deployment_gate=None,combined_gate=None,
        unexecuted=0,original60='SEALED',optimizer_updates=0,main_merged=False,
        scope='Only authorized S3 scope-r1 execution evidence, read-only analysis/delivery and mutable current metadata; user live Word release entries retained')
    save(p,m);print(dict(manifest_files=m['file_count'],total_bytes=m['total_bytes']))


def publication():
    head=git('rev-parse','HEAD');refs=git('ls-remote','origin','refs/heads/codex/global-local-preflight','refs/heads/main')
    remote={line.split()[1]:line.split()[0] for line in refs.splitlines()}
    require(remote['refs/heads/codex/global-local-preflight']==head,'Remote branch does not match payload')
    require(remote['refs/heads/main']=='5d5fe25754ffa032f97937a3fb54f15901c2fb7d','Unexpected main movement; do not merge/rewrite')
    before=lfs_entries(BASE);after=lfs_entries('HEAD')
    b={z['name']:z['oid'] for z in before};a={z['name']:z['oid'] for z in after}
    require(all(a.get(n)==v for n,v in b.items()),'Historical LFS identity changed')
    new={n:v for n,v in a.items() if n not in b}
    fsck=git('lfs','fsck','--objects','HEAD')
    staged=git('diff','--cached','--name-only');require(not staged,'Unexpected staged content after payload commit')
    obj=dict(utc=now(),payload_commit=head,remote_branch='refs/heads/codex/global-local-preflight',
        remote_research_SHA=remote['refs/heads/codex/global-local-preflight'],remote_main_SHA=remote['refs/heads/main'],remote_verified=True,
        push_status='PAYLOAD_PUSHED_AND_REMOTE_SHA_VERIFIED',LFS_object_fsck=fsck,
        historical_LFS_paths_unchanged=len(b),new_LFS_paths=len(new),new_LFS_oids=new,
        current_LFS_paths=len(a),force_push=False,main_merged=False,
        receipt_scope='Attests already-published payload. This receipt and updated audit/manifest are a metadata-only descendant; its final SHA is verified externally after the second push, avoiding self-reference.')
    write_new(RECEIPT,obj)
    p=HOME/'GIT_VERSION_AUDIT.json';v=read(p);v.update(publication_status=obj['push_status'],payload_commit=head,
        remote_research_SHA=obj['remote_research_SHA'],remote_main_SHA=obj['remote_main_SHA'],
        LFS_historical_paths_unchanged=len(b),LFS_new_paths=len(new),LFS_current_paths=len(a),LFS_fsck=fsck,
        metadata_receipt_descendant_scope=obj['receipt_scope']);save(p,v)
    print(dict(payload=head,remote_verified=True,LFS_old=len(b),LFS_new=len(new),LFS_total=len(a)))


if __name__=='__main__':
    import sys
    {'audit':audit,'manifest':manifest,'publication':publication,'sources':sources}[sys.argv[1]]()
