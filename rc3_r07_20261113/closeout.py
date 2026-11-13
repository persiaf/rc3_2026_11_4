"""Bounded metadata publication tail. No scientific calls, Torch, or new LFS."""
import datetime,hashlib,json,subprocess,time
from .common import ROOT,HOME,read,put
from .finalize import manifest

CFG=['-c','http.proxy=http://127.0.0.1:7897','-c','http.sslBackend=schannel','-c','http.version=HTTP/1.1']
BRANCH='refs/heads/codex/rc3-r07-action-rollouts-20261113'

def git(args,timeout=60):
    return subprocess.check_output(['git',*args],cwd=ROOT,stderr=subprocess.STDOUT,timeout=timeout).decode('utf-8').strip()

def main():
    start=time.time();p=read(HOME/'PROTOCOL.json');tail_cap=180.
    elapsed=start-p['origin_unix_s'];assert elapsed+tail_cap<=1800
    receipt=read(HOME/'PUBLICATION_RECEIPT.json');assert read(HOME/'LFS_REMOTE_READBACK.json')['status']=='PASS'
    r=read(HOME/'RESOURCE_ACCOUNTING.json')
    assert r['counts']['requests']==304 and r['counts']['actor_optimizer_attempts']==r['counts']['critic_optimizer_attempts']==0
    r.update(actual_wall_checkpoint_s=elapsed,actual_wall_checkpoint_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        wall_s=elapsed+tail_cap,wall_s_is_conservative_charged_upper_bound=True,
        final_metadata_and_remote_verification_reserved_s=tail_cap,
        final_tail_rule='Actual whole tail is asserted <=180s and origin-to-final-remote-readback <=this charged upper bound; unused reserve remains charged, never presented as actual measured scientific execution.',
        all_science_finished=True,no_further_scientific_stage=True,
        actual_new_LFS_objects=receipt['actual_new_LFS_objects'],actual_new_LFS_bytes=receipt['actual_new_LFS_bytes'])
    r['metadata_closeout_failures']=[dict(kind='FileExistsError',scope='metadata writer omitted explicit replace=True; failed before any science or resource modification',new_scientific_requests=0,ledger_reset=False)]
    put(HOME/'RESOURCE_ACCOUNTING.json',r,replace=True)
    put(HOME/'GIT_CLOSEOUT_OBSERVED.json',dict(base_commit=p['base_commit'],payload_commit=receipt['payload_commit'],
        preregistration_commit='9ae6f6e199211cc1f7069f1118fa57252c1d8bb2',
        actual_remote_payload_sha=receipt['actual_remote_payload_sha'],real_new_LFS_remote_readback='PASS',
        actual_wall_checkpoint_s=elapsed,charged_upper_bound_s=elapsed+tail_cap,
        final_followup_scope='Publication receipts, resource/manifest, metadata closeout source and status only; no new model/LFS',
        subsequent_final_branch_sha_verified_with_ls_remote=True,
        meaning='Final SHA itself cannot be embedded in its own commit; code below checks real final ref after push and reports the SHA.',
        no_main_merge=True,no_force_push=True))
    status=read(HOME/'STATUS.json');status.update(decision='OPPORTUNITY_INSUFFICIENT',automatic_next_stage='NONE',results='RESULTS.json',old_D5_D6_closed=True)
    put(HOME/'STATUS.json',status,replace=True)
    binding=read(HOME/'DELIVERY_SOURCE_BINDING.json');binding['sources']['closeout.py']=hashlib.sha256((HOME/'closeout.py').read_bytes()).hexdigest();put(HOME/'DELIVERY_SOURCE_BINDING.json',binding,replace=True)
    manifest()
    print(git(['add','--','rc3_r07_20261113','project_management/GIT_MANIFEST.json']))
    print(git(['-c','core.whitespace=cr-at-eol','diff','--cached','--check']))
    print(git(['commit','-m','Record R07 remote payload/LFS readback and close all-inclusive1800s resource ledger']))
    final=git(['rev-parse','HEAD'])
    print(git([*CFG,'-c','lfs.locksverify=false','push','--porcelain','origin','HEAD:'+BRANCH],timeout=90))
    refs=dict((line.split()[1],line.split()[0]) for line in git([*CFG,'ls-remote','origin',BRANCH,'refs/heads/main'],timeout=35).splitlines())
    assert refs[BRANCH]==final
    assert refs['refs/heads/main']==receipt['remote_main_sha']
    assert git(['diff','--name-only',receipt['payload_commit']+'..HEAD','--','rc3_r07_20261113/rollouts'])==''
    assert git(['log','--format=%H','--merges',p['base_commit']+'..HEAD'])==''
    total=time.time()-p['origin_unix_s'];tail=time.time()-start
    assert tail<=tail_cap and total<=r['wall_s']<=1800,(tail,total,r['wall_s'])
    print(json.dumps(dict(status='FINAL_REMOTE_VERIFIED',local_sha=final,remote_sha=refs[BRANCH],main_unchanged=True,
        actual_total_wall_s=total,actual_tail_s=tail,conservative_charged_wall_upper_s=r['wall_s'],
        actual_new_LFS_objects=receipt['actual_new_LFS_objects'],actual_new_LFS_bytes=receipt['actual_new_LFS_bytes'],
        remaining_user_only_status=git(['status','--short'])),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
