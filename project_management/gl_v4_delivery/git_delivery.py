"""Explicitly scoped publication bookkeeping; no staging/commit/push hidden."""
import json
import subprocess
from pathlib import Path
from rc3_gl_v4_20261109.common import ROOT,HOME,read,write_new,digest,label,now
from .reports import VERSION,save

BASE='925a0a1a8957299e3e849b29ac501a99212aa902'
BRANCH='codex/gl-v4-natural-start-20261109'
ADMIN=ROOT/'project_management/gl_v4_delivery'
RECEIPT=ADMIN/'PUBLICATION_RECEIPT.json'
MUTABLE={'.gitattributes','README.md','PROJECT_VERSION.json','EXPERIMENT_RESULT_TO_CLAIM.md','project_management/GIT_MANIFEST.json'}


def git(*args):return subprocess.check_output(['git','-c','core.quotepath=false',*args],cwd=ROOT,encoding='utf8').strip()
def allowed(n):return n in MUTABLE or n.startswith(('rc3_gl_v4_20261109/','project_management/gl_v4_delivery/'))
def lfs(ref):
    x=json.loads(git('lfs','ls-files','--json',ref));return x['files'] if isinstance(x,dict) else x


def audit():
    if git('branch','--show-current')!=BRANCH:raise ValueError('Wrong branch')
    changed=git('diff','--name-only',BASE).splitlines()
    if not all(allowed(n) or n=='第二章与第三章修订稿_20261029.docx' for n in changed):raise ValueError('Unrelated tracked changes')
    write_new(HOME/'GIT_VERSION_AUDIT.json',dict(utc=now(),baseline=BASE,branch=BRANCH,
        code_protocol_commit='35eac416789a3f50ffd407f183ae676f11bb3d8f',prepublication_HEAD=git('rev-parse','HEAD'),
        fetched_main=git('rev-parse','origin/main'),main_merge_base=git('merge-base','HEAD','origin/main'),
        old_S3_branch=git('rev-parse','codex/global-local-preflight'),old_S3_merge_base=git('merge-base','HEAD','codex/global-local-preflight'),
        preserved_actor_branch=git('rev-parse','codex/actor-pg-screen-20261109'),
        preserved_actor_merge_base=git('merge-base','HEAD','codex/actor-pg-screen-20261109'),
        original_S0=git('rev-parse','codex/global-local-s0'),S0_merge_base=git('merge-base','HEAD','codex/global-local-s0'),
        branch_creation='Created directly from exact925a0a1a, retaining committed c5f47ec actor branch and user Word edits',
        source_binding_sha256=digest(HOME/'SOURCE_BINDING.json'),baseline_LFS_paths=len(lfs(BASE)),
        unrelated_Word_files='PRESERVED_NOT_STAGED',old_S1_S2_S3_files_unchanged=read(HOME/'INDEPENDENT_AUDIT.json')['old_frozen_files_unchanged'],
        publication_status='PENDING_RESULT_PAYLOAD',publication_receipt=label(RECEIPT),main_merged=False,force_push=False))
    print('Git audit saved')


def manifest():
    p=ROOT/'project_management/GIT_MANIFEST.json';m=read(p)
    if label(RECEIPT) not in m['excluded_self_referential_metadata']:m['excluded_self_referential_metadata'].append(label(RECEIPT))
    targets=[ROOT/n for n in MUTABLE if n!='project_management/GIT_MANIFEST.json']
    for folder in (HOME,ADMIN):targets.extend(q for q in folder.rglob('*') if q.is_file() and '__pycache__' not in q.parts)
    for q in targets:
        n=label(q)
        if n not in m['excluded_self_referential_metadata']:m['files'][n]=dict(sha256=digest(q),bytes=q.stat().st_size)
    m['files']=dict(sorted(m['files'].items()));m['file_count']=len(m['files']);m['total_bytes']=sum(v['bytes'] for v in m['files'].values());m['version']=VERSION
    m['GL_v4']=dict(baseline=BASE,report=label(HOME/'README.md'),source_binding_sha256=digest(HOME/'SOURCE_BINDING.json'),
        deployment_requests=396,optimizer_updates=384,training_terminal_requests=3072,evaluation_requests=216,
        branch=BRANCH,old_S3_preserved=True,old60='SEALED',main_merged=False,
        scope='Only new independent GL-v4 and mutable root/current metadata. Existing science and live user Word release entries retained.')
    save(p,m);print(dict(manifest_files=m['file_count'],bytes=m['total_bytes']))


def sources():
    write_new(HOME/'READ_ONLY_ANALYSIS_BINDING.json',dict(utc=now(),scientific_source_binding_sha256=digest(HOME/'SOURCE_BINDING.json'),
        source_sha256={label(q):digest(q) for q in sorted(ADMIN.glob('*.py'))},
        scope='Read-only analysis/report/publication helpers, not new scientific execution source',new_scientific_calls=0))


def publication():
    head=git('rev-parse','HEAD');remote={x.split()[1]:x.split()[0] for x in git('ls-remote','origin','refs/heads/'+BRANCH,'refs/heads/main').splitlines()}
    if remote['refs/heads/'+BRANCH]!=head:raise ValueError('Remote research SHA mismatch')
    before={v['name']:v['oid'] for v in lfs(BASE)};after={v['name']:v['oid'] for v in lfs('HEAD')}
    if not all(after.get(n)==h for n,h in before.items()):raise ValueError('Old LFS identity changed')
    fsck=git('lfs','fsck','--objects','HEAD');new={n:h for n,h in after.items() if n not in before}
    data=dict(utc=now(),payload_commit=head,remote_research_SHA=remote['refs/heads/'+BRANCH],remote_main_SHA=remote['refs/heads/main'],
        remote_verified=True,publication_status='PAYLOAD_PUSHED_VERIFIED',historical_LFS_paths_unchanged=len(before),new_LFS_paths=len(new),
        current_LFS_paths=len(after),LFS_fsck=fsck,new_LFS_oids=new,force_push=False,main_merged=False,
        scope='Attests published payload. This receipt/audit/manifest are a metadata descendant; final descendant SHA verified externally after push to avoid self-reference.')
    write_new(RECEIPT,data);p=HOME/'GIT_VERSION_AUDIT.json';a=read(p)
    a.update(publication_status=data['publication_status'],payload_commit=head,remote_research_SHA=data['remote_research_SHA'],
        remote_main_SHA=data['remote_main_SHA'],LFS_historical_paths_unchanged=len(before),LFS_new_paths=len(new),LFS_current_paths=len(after),LFS_fsck=fsck)
    save(p,a);print(dict(payload=head,remote_verified=True,LFS_old=len(before),LFS_new=len(new)))


if __name__=='__main__':
    import sys
    {'audit':audit,'manifest':manifest,'sources':sources,'publication':publication}[sys.argv[1]]()
