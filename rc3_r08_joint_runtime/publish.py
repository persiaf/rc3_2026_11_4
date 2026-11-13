"""Scoped B delivery, no scientific calls. Preserve original 1800s wall."""
import datetime,hashlib,json,subprocess,time
from .common import ROOT,HOME,RUN,BASE,read,put,sha

BRANCH='refs/heads/codex/rc3-r08-b-joint-opportunity-20261113'
TRACE='paper/review-traces/experiment-result-to-claim/20261113_r08_b_run01'
CFG=['-c','http.proxy=http://127.0.0.1:7897','-c','http.sslBackend=schannel','-c','http.version=HTTP/1.1']
def git(args,timeout=90):return subprocess.check_output(['git',*args],cwd=ROOT,stderr=subprocess.STDOUT,timeout=timeout).decode('utf-8').strip()

def metadata(final):
    p=read(HOME/'PROTOCOL.json');elapsed=time.time()-p['origin_unix_s'];assert elapsed<1800
    r=read(HOME/'RESOURCE_ACCOUNTING.json');r['wall_s']=elapsed;r['accounting_checkpoint_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    if final:
        assert elapsed+180<=1800
        r.update(actual_wall_checkpoint_s=elapsed,final_metadata_tail_charged_reserve_s=180,wall_s=elapsed+180,
            wall_s_is_conservative_charged_upper_bound=True,tail_scope='Actual all-inclusive through final remote check asserted <=charged upper; unused tail reserve remains charged, not called measured science time')
    put(HOME/'RESOURCE_ACCOUNTING.json',r,True)
    put(HOME/'DELIVERY_SOURCE_BINDING.json',dict(scope='Postprocessing and publication only; no frozen execution source modified',
        source_sha256={f.name:sha(f) for f in HOME.glob('*.py')},preregistered_execution_commit='e7a2f3150',utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),(HOME/'DELIVERY_SOURCE_BINDING.json').exists())
    mpath=ROOT/'project_management/GIT_MANIFEST.json';raw=mpath.read_bytes();m=json.loads(raw)
    scopes=[HOME.name,TRACE,'rc3_r04_m1_recovery_20261111/run_claims/'+RUN+'*','README.md','.gitattributes','EXPERIMENT_RESULT_TO_CLAIM.md']
    names=git(['ls-files','-z','-c','-o','--exclude-standard','--',*scopes]).split('\0')
    for name in sorted(set(n for n in names if n)):
        f=ROOT/name
        if f.is_file():m['files'][name]=dict(sha256=sha(f),bytes=f.stat().st_size)
    m['version']='2026.11.13-R08-B-fixed-neighborhood-closed'
    m['R08_B']=dict(base_commit=BASE,branch=BRANCH.removeprefix('refs/heads/'),status='OPPORTUNITY_INSUFFICIENT',
        requests=719,anchors=48,edits=671,Actor=0,Critic=0,H=719,new_teacher_scene_navigation_native=0,
        physical_training_parents=48,oracle_gain=.011954778056381211,material_net_joint_fraction=11/74,
        C_D_E='NOT_RUN_NOT_AUTHORIZED',old_evidence_and_user_files_preserved=True,main_merged=False,
        report=HOME.name+'/EXPERIMENT_REPORT.md',resource=HOME.name+'/RESOURCE_ACCOUNTING.json')
    if (HOME/'PUBLICATION_RECEIPT.json').exists():m['R08_B'].update(read(HOME/'PUBLICATION_RECEIPT.json'))
    text=json.dumps(m,ensure_ascii=False,indent=2)+'\n'
    if b'\r\n' in raw:text=text.replace('\n','\r\n')
    mpath.write_bytes(text.encode('utf-8'))

def main(final=False):
    start=time.time();metadata(final)
    scopes=[HOME.name,TRACE,'rc3_r04_m1_recovery_20261111/run_claims/'+RUN+'.json','rc3_r04_m1_recovery_20261111/run_claims/'+RUN+'.terminal.json','README.md','.gitattributes','EXPERIMENT_RESULT_TO_CLAIM.md','project_management/GIT_MANIFEST.json']
    print(git(['add','--',*scopes]));print(git(['-c','core.whitespace=cr-at-eol','diff','--cached','--check']))
    names=git(['diff','--cached','--name-only']).splitlines()
    allowed=lambda n:n.startswith((HOME.name+'/',TRACE+'/','rc3_r04_m1_recovery_20261111/run_claims/'+RUN)) or n in ['README.md','.gitattributes','EXPERIMENT_RESULT_TO_CLAIM.md','project_management/GIT_MANIFEST.json']
    assert all(allowed(n) for n in names),names
    print(git(['commit','-m','Close R08-B remote verification and resource accounting' if final else 'Complete R08-B719 real requests: insufficient joint-edit opportunity, close fixed neighborhood']))
    print(git([*CFG,'-c','lfs.locksverify=false','push','--porcelain','origin','HEAD:'+BRANCH],timeout=100))
    head=git(['rev-parse','HEAD']);refs=dict((line.split()[1],line.split()[0]) for line in git([*CFG,'ls-remote','origin',BRANCH,'refs/heads/main'],timeout=35).splitlines());assert refs[BRANCH]==head
    total=time.time()-read(HOME/'PROTOCOL.json')['origin_unix_s']
    if final:
        assert refs['refs/heads/main']==read(HOME/'PUBLICATION_RECEIPT.json')['remote_main_sha']
        assert not git(['diff','--name-only',read(HOME/'PUBLICATION_RECEIPT.json')['payload_commit']+'..HEAD','--',HOME.name+'/rollouts'])
        assert time.time()-start<=180 and total<=read(HOME/'RESOURCE_ACCOUNTING.json')['wall_s']<=1800
    print(json.dumps(dict(status='FINAL_REMOTE_VERIFIED' if final else 'PAYLOAD_PUSHED',sha=head,remote_sha=refs[BRANCH],actual_total_wall_s=total,
        charged_wall_s=read(HOME/'RESOURCE_ACCOUNTING.json')['wall_s'],C_D_E='NOT_RUN_NOT_AUTHORIZED',remaining_status=git(['status','--short'])),ensure_ascii=False),flush=True)

if __name__=='__main__':
    import sys
    main('--final' in sys.argv)
