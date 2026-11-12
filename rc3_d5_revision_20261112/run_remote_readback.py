"""Bound publication readback by the original remaining closeout pool."""
from pathlib import Path
import json,datetime,time,os,sys,hashlib

ROOT=Path(__file__).resolve().parent.parent;H=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    start=time.monotonic();wall=time.time();raw=read(H/'RESOURCE_SCIENTIFIC_FINAL_RAW.json')
    current=read(H/'RESOURCE_ACCOUNTING.json');assert read(H/'STATUS.json')['closed'] and current['counts']==raw['counts']
    end=datetime.datetime.fromisoformat(read(H/'controls/science-03/SUPERVISOR_RECEIPT.json')['finish_utc']).timestamp()
    review=current['delivery_accounting']['parallel_reviewer_window_s']
    actual_closeout=time.time()-end+review
    remaining=min(7200-actual_closeout,345600-raw['wall']['total_s']-actual_closeout)-150
    assert remaining>60,'Not enough original closeout budget for another bounded readback'
    sources={n:hashlib.sha256((H/n).read_bytes()).hexdigest() for n in ['verify_remote_lfs.py','run_remote_readback.py']}
    save(H/'LFS_READBACK_SOURCE_BINDING.json',dict(source_sha256=sources,remaining_original_reproduction_s=remaining,
        reserve_final_metadata_s=150,scientific_counts_unchanged=True,no_model_or_planning=True,
        before_remote_fetch=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    from rc3_r04_m1_recovery_20261111.supervisor import launch_supervised
    cmd=[sys.executable,'-B','-X','utf8',str(H/'verify_remote_lfs.py')]
    receipt=launch_supervised(cmd,H/'publication_controls/remote-lfs-01','d5-revision-remote-lfs-01',
        wall_limit_s=remaining,closeout_reserve_s=20,termination_tolerance_s=2,cwd=ROOT,env=os.environ.copy(),
        entry_monotonic=start,entry_wall=wall)
    assert read(H/'RESOURCE_ACCOUNTING.json')['counts']==raw['counts']
    print(json.dumps(dict(status=receipt['status'],wall_s=receipt['elapsed_s']),ensure_ascii=False),flush=True)
    if receipt['status']!='COMPLETE':raise RuntimeError('Publication readback failed; actual supervisor receipts retained')
if __name__=='__main__':main()
