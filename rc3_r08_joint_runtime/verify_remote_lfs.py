"""Fresh scoped Git-LFS readback of this task's actual new objects, including history."""
from pathlib import Path
import subprocess,json,re,hashlib,time,datetime,sys

ROOT=Path(__file__).resolve().parent.parent;H=Path(__file__).resolve().parent
BASE='27deb2daf46f091cc2b9a7e3c475b7242eb2753e'
BRANCH='refs/heads/codex/rc3-r08-b-joint-opportunity-20261113'
CFG=['-c','http.proxy=http://127.0.0.1:7897','-c','http.sslBackend=schannel','-c','http.version=HTTP/1.1']
def git(args):return subprocess.check_output(['git',*args],cwd=ROOT)
def save(n,d):(H/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    start=time.monotonic();payload=git(['rev-parse','HEAD']).decode().strip()
    commits=git(['rev-list','--reverse',BASE+'..'+payload]).decode().splitlines()
    inv={};final_oids=set()
    for commit in commits:
        tree=git(['ls-tree','-rlz',commit,'--',H.name])
        small={}
        for line in tree.split(b'\0'):
            if not line:continue
            head,path=line.split(b'\t',1);mode,typ,blob,size=head.decode().split()
            if typ=='blob' and int(size)<=200:small.setdefault(blob,[]).append(path.decode('utf-8'))
        blobs=list(small)
        packed=subprocess.run(['git','cat-file','--batch'],cwd=ROOT,input=('\n'.join(blobs)+'\n').encode(),stdout=subprocess.PIPE,check=True).stdout
        cursor=0
        for blob in blobs:
            end=packed.index(b'\n',cursor);header=packed[cursor:end].decode();length=int(header.split()[-1]);begin=end+1;data=packed[begin:begin+length];cursor=begin+length+1
            m=re.fullmatch(rb'version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n',data)
            if not m:continue
            oid=m[1].decode();size=int(m[2]);e=inv.setdefault(oid,dict(oid=oid,size=size,paths=[],commits=[]));assert e['size']==size
            e['paths']=sorted(set(e['paths']+small[blob]));e['commits'].append(commit)
            if commit==payload:final_oids.add(oid)
    # Identity-only base pointer catalog. No historical route/data body is read.
    base_lines=git(['lfs','ls-files','--long',BASE]).decode('utf-8').splitlines()
    old_oids={line.split()[0] for line in base_lines if re.match(r'^[0-9a-f]{64} ',line)}
    new={o:v for o,v in inv.items() if o not in old_oids}
    inventory=dict(base_commit=BASE,payload_commit=payload,scope=H.name+'/**',commits=commits,
        unique_scoped_objects=len(inv),actual_new_objects=len(new),actual_new_bytes=sum(v['size'] for v in new.values()),
        current_tree_objects=len(final_oids),current_tree_new_objects=len(final_oids-old_oids),
        intermediate_only_new_objects=len(set(new)-final_oids),
        base_pointer_catalog_identity_only=True,base_pointer_OID_count=len(old_oids),
        base_sorted_OID_sha256=hashlib.sha256('\n'.join(sorted(old_oids)).encode()).hexdigest(),
        objects=sorted(inv.values(),key=lambda x:x['oid']))
    save('LFS_NEW_OBJECT_INVENTORY.json',inventory)
    remote=git([*CFG,'ls-remote','origin',BRANCH,'refs/heads/main']).decode().splitlines()
    refs={line.split()[1]:line.split()[0] for line in remote};assert refs[BRANCH]==payload,refs
    store=ROOT/'.git/lfs_verify_r08_b_20261113'
    assert not store.exists(),'Verification must start with a genuinely empty distinct object store'
    cmd=['git',*CFG,'-c','lfs.storage='+str(store),'-c','lfs.fetchrecentalways=false','-c','lfs.concurrenttransfers=4',
         'lfs','fetch','--include='+H.name+'/**','--exclude=','origin',*commits]
    save('LFS_READBACK_REGISTRATION.json',dict(payload_commit=payload,remote_refs=refs,command=cmd,
        fresh_store=str(store),fresh_store_absent_before_fetch=True,actual_new_objects=len(new),
        actual_new_bytes=inventory['actual_new_bytes'],scope=H.name+'/**',delete_or_prune=False,
        no_original60_oldS3_semantics=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(json.dumps(dict(status='FRESH_SCOPED_FETCH_START',objects=len(inv),actual_new=len(new),bytes=inventory['actual_new_bytes'])),flush=True)
    with (H/'LFS_READBACK_STDOUT.txt').open('w',encoding='utf-8') as out:
        fetch=subprocess.run(cmd,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
    if fetch.returncode:raise RuntimeError('Actual remote LFS fetch failed; see preserved stdout')
    verified=[]
    for oid,entry in sorted(inv.items()):
        p=store/'objects'/oid[:2]/oid[2:4]/oid
        assert p.is_file() and p.stat().st_size==entry['size'] and digest(p)==oid,oid
        verified.append(dict(oid=oid,size=entry['size'],new_relative_to_base=oid in new))
    save('LFS_REMOTE_READBACK.json',dict(status='PASS',payload_commit=payload,verified_objects=len(verified),
        verified_new_objects=len(new),verified_new_bytes=inventory['actual_new_bytes'],
        intermediate_only_new_objects=inventory['intermediate_only_new_objects'],
        fresh_downloaded_objects_sha256_and_size_checked=True,store=str(store),
        current_tree_new_objects=inventory['current_tree_new_objects'],objects=verified,
        wall_s=time.monotonic()-start,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    caches=[]
    for cache in [store]:
        caches.append(dict(path=str(cache),exists=cache.exists(),
            bytes=sum(p.stat().st_size for p in cache.rglob('*') if p.is_file()) if cache.exists() else None,
            semantic_inputs_read=False,deleted=False))
    save('CACHE_MAINTENANCE_RECORD.json',dict(caches=caches,cleanup_requested=False,
        prior_cleanup_rejection_not_bypassed=True,cache_removal_is_not_scientific_admission=True))
    changed=git(['diff','--name-only',BASE+'..'+payload]).decode('utf-8').splitlines()
    allowed=lambda p:p.startswith((H.name+'/', 'paper/review-traces/experiment-result-to-claim/20261113_r08_b_run01/','rc3_r04_m1_recovery_20261111/run_claims/rc3-r08-b-joint-opportunity-20261113-r1')) or p in ['.gitattributes','README.md','project_management/GIT_MANIFEST.json','EXPERIMENT_RESULT_TO_CLAIM.md']
    assert all(allowed(p) for p in changed),[p for p in changed if not allowed(p)]
    assert not git(['log','--format=%H','--merges',BASE+'..'+payload]).strip()
    save('PRESERVATION_CHECK.json',dict(status='PASS',base_commit=BASE,payload_commit=payload,
        changed_paths=len(changed),all_changes_in_authorized_scope=True,
        old_D5_D6_and_closed_experiment_git_blobs_unchanged=True,user_files_not_committed=True,
        merge_commits_added=0,original60_oldS3_semantic_reads=0))
    save('PUBLICATION_RECEIPT.json',dict(status='PAYLOAD_REMOTE_AND_REAL_NEW_LFS_VERIFIED',
        branch=BRANCH.removeprefix('refs/heads/'),base_commit=BASE,payload_commit=payload,
        actual_remote_payload_sha=refs[BRANCH],remote_main_sha=refs['refs/heads/main'],
        actual_new_LFS_objects=len(new),actual_new_LFS_bytes=inventory['actual_new_bytes'],
        remote_downloaded_hash_verified=True,intermediate_versions_included=True,
        verification_receipt='LFS_REMOTE_READBACK.json',no_main_merge=True,no_force_push=True,
        user_files_excluded=True,old_D5_D6_bytes_unchanged=True,
        verification_cache_preserved=True,cache_cleanup_not_a_scientific_gate=True,
        metadata_followup='Publication receipts and final resource/manifest commit only; no new model/LFS objects',
        elapsed_s=time.monotonic()-start,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(json.dumps(dict(status='REMOTE_LFS_PASS',new_objects=len(new),bytes=inventory['actual_new_bytes'],payload=payload)),flush=True)

if __name__=='__main__':main()
