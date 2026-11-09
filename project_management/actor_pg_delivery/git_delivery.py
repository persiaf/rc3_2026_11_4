"""Allowlisted publication metadata. No scientific stages or automatic Git writes."""
import json
import re
import subprocess
import time
from rc3_actor_pg_20261109.common import *
from .reports import VERSION,save

ADMIN=ROOT/'project_management/actor_pg_delivery'
BRANCH='codex/actor-pg-screen-20261109'
MAIN='5d5fe25754ffa032f97937a3fb54f15901c2fb7d'
RECEIPT=ADMIN/'PUBLICATION_RECEIPT.json'
MUTABLE={'README.md','PROJECT_VERSION.json','EXPERIMENT_RESULT_TO_CLAIM.md','project_management/GIT_MANIFEST.json'}
USER='第二章与第三章修订稿_20261029.docx'


def git(*args):return subprocess.check_output(['git','-c','core.quotepath=false',*args],cwd=ROOT,encoding='utf8').strip()
def allowed(name):return name in MUTABLE or name.startswith(('rc3_actor_pg_20261109/','project_management/actor_pg_delivery/'))
def lfs(rev):
    value=json.loads(git('lfs','ls-files','--json',rev));return value['files'] if isinstance(value,dict) else value


def audit():
    require(git('branch','--show-current')==BRANCH,'Wrong independent branch')
    changes=git('diff','--name-only',BASE).splitlines();require(all(allowed(n) or n==USER for n in changes),'Old frozen evidence changed')
    write_new(HOME/'GIT_VERSION_AUDIT.json',dict(utc=now(),baseline=BASE,branch=BRANCH,
        prepublication_HEAD=git('rev-parse','HEAD'),fetched_origin_preflight=git('rev-parse','origin/codex/global-local-preflight'),
        fetched_main=git('rev-parse','origin/main'),merge_base_main=git('merge-base','HEAD','origin/main'),
        S0_branch=git('rev-parse','codex/global-local-s0'),merge_base_S0=git('merge-base','HEAD','codex/global-local-s0'),
        teacher_branch=git('rev-parse','codex/teacher-distill-t0-t2'),merge_base_teacher=git('merge-base','HEAD','codex/teacher-distill-t0-t2'),
        initial_and_prepublication_fetch='Successful normal inherited Git transport; no global settings changed',
        old_LFS_paths=len(lfs(BASE)),old_scientific_byte_audit='INDEPENDENT_REVIEW.json and full trace:49638 files unchanged',
        tracked_changes_before_new_commits=changes,unrelated_user_Word='PRESERVED_NOT_STAGED',
        publication_status='PENDING_NEW_BRANCH_PUBLICATION',publication_receipt=label(RECEIPT),
        protocol_registration_scope='Files+source/request hashes registered before first science; Git archiving occurs after execution; no false pre-run commit claim',
        old_S3_active_status_unchanged=True,old_budget_reused=False,main_merged=False,force_push=False))
    print('Independent branch Git audit recorded')


def bind_analysis():
    ps=list(ADMIN.glob('*.py'))+list((HOME/'paper/review-traces').rglob('*.py'))
    write_new(HOME/'READ_ONLY_ANALYSIS_BINDING.json',dict(utc=now(),production_source_binding_sha256=digest(HOME/'SOURCE_BINDING.json'),
        source_sha256={label(p):digest(p) for p in sorted(ps)},role='Static saved-data audits/statistics/report/publication only, not new scientific evaluator',
        new_scientific_calls=0))
    print(dict(analysis_files=len(ps)))


def manifest():
    p=ROOT/'project_management/GIT_MANIFEST.json';m=read(p)
    exclusion=label(RECEIPT)
    if exclusion not in m['excluded_self_referential_metadata']:m['excluded_self_referential_metadata'].append(exclusion)
    targets=[ROOT/n for n in MUTABLE if n!='project_management/GIT_MANIFEST.json']
    for folder in (HOME,ADMIN):targets.extend(q for q in folder.rglob('*') if q.is_file() and '__pycache__' not in q.parts)
    for p in targets:
        name=label(p)
        if name not in m['excluded_self_referential_metadata']:m['files'][name]=dict(sha256=digest(p),bytes=p.stat().st_size)
    m['files']=dict(sorted(m['files'].items()));m['file_count']=len(m['files']);m['total_bytes']=sum(v['bytes'] for v in m['files'].values());m['version']=VERSION
    m['actor_deployment_matched_pg_screen']=dict(baseline=BASE,branch=BRANCH,report=label(HOME/'README.md'),
        source_binding_sha256=digest(HOME/'SOURCE_BINDING.json'),deployment='GO_EXACT_QUALITY_OBSERVED_SPEED',mechanism='REVISE',
        actor_exports=12,deployment_requests=756,A_updates=128,B_updates=128,mechanism_terminal=3126,
        old_scientific_files_preserved=49638,old_S3_failures_refilled=0,old60='SEALED',old_budgets_reused=False,main_merged=False,
        scope='Only new independent study, saved-data reports and mutable project metadata; old scientific entries and live user Word release entries retained')
    save(ROOT/'project_management/GIT_MANIFEST.json',m);print(dict(files=m['file_count'],bytes=m['total_bytes']))


def validate():
    start=time.perf_counter();b=read(HOME/'SOURCE_BINDING.json');a=read(HOME/'READ_ONLY_ANALYSIS_BINDING.json')
    require(all(digest(ROOT/n)==v for n,v in b['source_sha256'].items()),'Production source changed after run')
    require(all(digest(ROOT/n)==v for n,v in a['source_sha256'].items()),'Analysis source changed after binding')
    for key,p in (('protocol_sha256','PROTOCOL.json'),('input_registry_sha256','INPUT_REGISTRY.json'),('deployment_grid_sha256','DEPLOYMENT_GRID.json'),('authorization_sha256','AUTHORIZATION.json')):
        require(digest(HOME/p)==b[key],'Registered input/protocol changed: '+p)
    names=git('diff','--cached','--name-only').splitlines();require(all(allowed(n) for n in names),'Unrelated staged file')
    oldchanges=[n for n in git('diff','--name-only',BASE).splitlines() if not allowed(n) and n!=USER]
    require(not oldchanges,'Old science modified')
    links=[]
    for p in HOME.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf8')):
            if '://' in target or target.startswith('#'):continue
            require((p.parent/target.split('#')[0]).exists(),'Missing report link '+str((p,target)))
            links.append([p.name,target])
    m=read(ROOT/'project_management/GIT_MANIFEST.json');checked=0
    for name in [n for n in m['files'] if n.startswith(('rc3_actor_pg_20261109/','project_management/actor_pg_delivery/'))]+[n for n in MUTABLE if n!='project_management/GIT_MANIFEST.json']:
        require(digest(ROOT/name)==m['files'][name]['sha256'],'New delivery manifest mismatch '+name);checked+=1
    write_new(HOME/'DELIVERY_VALIDATION.json',dict(status='PASS',utc=now(),wall_s=time.perf_counter()-start,
        production_source_files=len(b['source_sha256']),analysis_source_files=len(a['source_sha256']),registered_input_hashes_match=True,
        new_delivery_manifest_hashes_checked=checked,report_links_checked=len(links),unrelated_Word_staged=False,
        old_scientific_changes=[],new_scientific_calls=0))
    print(dict(status='PASS',sources=len(b['source_sha256']),manifest_files=checked,links=len(links)))


def publication():
    head=git('rev-parse','HEAD');refs=git('ls-remote','origin','refs/heads/'+BRANCH,'refs/heads/main','refs/heads/codex/global-local-preflight')
    remote={s.split()[1]:s.split()[0] for s in refs.splitlines()}
    require(remote.get('refs/heads/'+BRANCH)==head,'Remote new branch does not match payload')
    require(remote['refs/heads/main']==MAIN and remote['refs/heads/codex/global-local-preflight']==BASE,'Existing branches unexpectedly moved')
    before={v['name']:v['oid'] for v in lfs(BASE)};after={v['name']:v['oid'] for v in lfs('HEAD')}
    require(all(after.get(n)==v for n,v in before.items()),'Old LFS path identity changed')
    added={n:v for n,v in after.items() if n not in before};fsck=git('lfs','fsck','--objects','HEAD')
    receipt=dict(utc=now(),payload_commit=head,remote_branch=BRANCH,remote_research_SHA=remote['refs/heads/'+BRANCH],
        remote_main_SHA=remote['refs/heads/main'],remote_old_preflight_SHA=remote['refs/heads/codex/global-local-preflight'],
        remote_verified=True,push_status='PAYLOAD_PUSHED_REMOTE_SHA_AND_LFS_VERIFIED',LFS_fsck=fsck,
        old_LFS_paths_unchanged=len(before),new_LFS_paths=len(added),all_LFS_paths=len(after),new_LFS_objects=added,
        force_push=False,main_merged=False,
        scope='Attests already-published code/protocol and actual-result payload. This receipt/audit/manifest are a metadata-only descendant; final descendant HEAD is independently verified after push without a self-referential claim.')
    write_new(RECEIPT,receipt)
    p=HOME/'GIT_VERSION_AUDIT.json';x=read(p);x.update(publication_status=receipt['push_status'],payload_commit=head,
        remote_research_SHA=receipt['remote_research_SHA'],remote_main_SHA=receipt['remote_main_SHA'],
        old_preflight_SHA=BASE,LFS_old_paths_unchanged=len(before),LFS_new_paths=len(added),LFS_total_paths=len(after),LFS_fsck=fsck,
        metadata_receipt_scope=receipt['scope']);save(p,x)
    p=ROOT/'PROJECT_VERSION.json';x=read(p);x['actor_deployment_matched_pg_screen'].update(publication_receipt=label(RECEIPT),payload_commit=head,payload_remote_verified=True);save(p,x)
    print(dict(payload=head,remote_verified=True,LFS_old=len(before),LFS_new=len(added),LFS_total=len(after)))


if __name__=='__main__':
    import sys
    {'audit':audit,'bind':bind_analysis,'manifest':manifest,'validate':validate,'publication':publication}[sys.argv[1]]()
