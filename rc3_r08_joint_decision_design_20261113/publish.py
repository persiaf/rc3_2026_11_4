"""Scoped maintenance metadata and authorized independent branch publication."""
import datetime,hashlib,json,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent;HOME=Path(__file__).resolve().parent
BASE='d0b021db8a7f74f50622c0feb68d5dda23ed6738'
BRANCH='refs/heads/codex/rc3-r08-joint-decision-design-20261113'
TRACE='paper/review-traces/experiment-result-to-claim/20261113_r08_run01'
CFG=['-c','http.proxy=http://127.0.0.1:7897','-c','http.sslBackend=schannel','-c','http.version=HTTP/1.1']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def git(args,timeout=90):return subprocess.check_output(['git',*args],cwd=ROOT,stderr=subprocess.STDOUT,timeout=timeout).decode('utf-8').strip()
def wall():return time.time()-datetime.datetime(2026,11,13,4,40,tzinfo=datetime.timezone.utc).timestamp()

def metadata(final=False):
    r=read(HOME/'RESOURCE_ACCOUNTING.json');r.update(all_inclusive_wall_through_checkpoint_s=wall(),checkpoint_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if final:r.update(publication_tail_charged_reserve_s=180.,conservative_all_inclusive_wall_upper_s=wall()+180.,reserve_is_not_measured_scientific_time=True)
    save(HOME/'RESOURCE_ACCOUNTING.json',r)
    sources=read(HOME/'SOURCE_BINDING.json');sources['new_source_sha256']={p.name:sha(p) for p in HOME.glob('*.py')};save(HOME/'SOURCE_BINDING.json',sources)
    rows=['# R08-A输出清单\n','独立新目录、Git提交版本化；依用户要求不覆盖旧paper计划，不生成重复时间戳报告副本。SHA按本轮最终内容，不包含本清单自身或根Git清单以避免自引用。\n','|文件|字节|SHA256|\n|---|---:|---|']
    for p in sorted(HOME.iterdir()):
        if p.is_file() and p.name!='MANIFEST.md':rows.append(f'|[{p.name}]({p.name})|{p.stat().st_size}|{sha(p)}|')
    (HOME/'MANIFEST.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    p=ROOT/'project_management/GIT_MANIFEST.json';raw=p.read_bytes();m=json.loads(raw)
    # Do not read or refresh old user-file entries. Only this scope is added.
    names=git(['ls-files','-z','-c','-o','--exclude-standard','--',HOME.name,TRACE,'README.md','.gitattributes','EXPERIMENT_RESULT_TO_CLAIM.md']).split('\0')
    for name in sorted(set(n for n in names if n)):
        f=ROOT/name
        if f.is_file():m['files'][name]=dict(sha256=sha(f),bytes=f.stat().st_size)
    m['version']='2026.11.13-R08-A-joint-decision-design'
    m['R08_A']=dict(base_commit=BASE,branch=BRANCH.removeprefix('refs/heads/'),stage='A',status='COMPLETE_SAVED_ANALYSIS_AND_DESIGN',
        comparison_rows=3550,training_pairs=352,training_physical_parents=96,R07_saved_states=144,
        new_Actor_Critic_forward_H_native_navigation_teacher_scene_complete_requests=0,
        interface_tests=12,independent_claim_verdict='partial',future_B_E='NOT_RUN_NOT_AUTHORIZED',
        active_legacy_protocol_pointer_changed=False,old_user_file_entries_not_refreshed=True,
        report=HOME.name+'/EXPERIMENT_REPORT.md',plan=HOME.name+'/R08_MASTER_PLAN.md',main_merged=False)
    if (HOME/'PUBLICATION_RECEIPT.json').exists():m['R08_A'].update(read(HOME/'PUBLICATION_RECEIPT.json'))
    out=json.dumps(m,ensure_ascii=False,indent=2)+'\n'
    if b'\r\n' in raw:out=out.replace('\n','\r\n')
    p.write_bytes(out.encode('utf-8'))

def publish(final=False):
    start=time.time();metadata(final)
    print(git(['add','--',HOME.name,TRACE,'README.md','.gitattributes','EXPERIMENT_RESULT_TO_CLAIM.md','project_management/GIT_MANIFEST.json']))
    names=git(['diff','--cached','--name-only']).splitlines()
    allowed=lambda n:n.startswith((HOME.name+'/',TRACE+'/')) or n in ['README.md','.gitattributes','EXPERIMENT_RESULT_TO_CLAIM.md','project_management/GIT_MANIFEST.json']
    assert all(allowed(n) for n in names),names
    print(git(['-c','core.whitespace=cr-at-eol','diff','--cached','--check']))
    message='Record R08-A remote SHA/LFS verification and final resources' if final else 'Complete R08-A saved-route analysis, joint coverage-routing model and bounded hierarchical DRL proposal'
    print(git(['commit','-m',message]))
    print(git([*CFG,'-c','lfs.locksverify=false','push','--porcelain','origin','HEAD:'+BRANCH],timeout=120))
    head=git(['rev-parse','HEAD']);refs=dict((x.split()[1],x.split()[0]) for x in git([*CFG,'ls-remote','origin',BRANCH,'refs/heads/main']).splitlines())
    assert refs[BRANCH]==head
    if final:
        receipt=read(HOME/'PUBLICATION_RECEIPT.json');assert refs['refs/heads/main']==receipt['remote_main_sha']
        assert not git(['diff','--name-only',receipt['payload_commit']+'..HEAD','--',HOME.name+'/ROUTE_STRUCTURE_EVIDENCE.json.gz'])
        assert time.time()-start<=180
    print(json.dumps(dict(status='FINAL_REMOTE_VERIFIED' if final else 'PAYLOAD_PUSHED',sha=head,remote_sha=refs[BRANCH],actual_all_inclusive_wall_s=wall(),new_training_planning_calls=0,stage_B_E='NOT_RUN_NOT_AUTHORIZED',remaining_status=git(['status','--short'])),ensure_ascii=False),flush=True)

if __name__=='__main__':
    import sys
    publish('--final' in sys.argv)
