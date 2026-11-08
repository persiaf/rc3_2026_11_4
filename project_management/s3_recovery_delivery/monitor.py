"""Small read-only log snapshots; never opens locked leases or computes tasks."""
import json
from pathlib import Path
import re
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2]
CONT=ROOT/'rc3_s3_evaluator/runs/rc3_v3_global_local_20261106_01/continuations/persistence_r1'


def snapshot():
    out=dict(utc=datetime.now(timezone.utc).isoformat(),seeds=[])
    for seed in (11622026,11622027,11622028):
        folder=CONT/str(seed);r=dict(seed=seed,status='NOT_STARTED')
        if folder.exists():
            r['status']='STARTED'
            r['anchor_retry_receipts']=sum(1 for _ in (folder/'io_receipts').glob('*.json'))
            logs=list(folder.glob('session_*.log'))
            if logs:
                p=max(logs,key=lambda p:p.stat().st_mtime_ns)
                with p.open('rb') as f:f.seek(max(0,p.stat().st_size-4096));tail=f.read().decode('utf8',errors='replace')
                lines=tail.splitlines();r['last_line']=lines[-1] if lines else ''
                for line in reversed(lines):
                    m=re.search(r'(\d+) scheduled=(\d+)/(\d+) success=(\d+) historical_failure=(\d+)',line)
                    if m:
                        r.update(scheduled=int(m[2]),success=int(m[4]),historical_failure=int(m[5]));break
            done=folder/'SCHEDULE_SEAL.json'
            if done.exists():r['status']='SCHEDULE_DISPOSED_SEALED'
        out['seeds'].append(r)
    out['analysis_present']=(CONT/'SCIENTIFIC_ANALYSIS.json').exists()
    closeout=ROOT/'rc3_s3_recovery_20261108/results/RESOURCE_ACCOUNTING.json'
    if closeout.exists():
        account=json.loads(closeout.read_text(encoding='utf8'))
        for row in out['seeds']:
            actual=account['actual_by_seed'][str(row['seed'])]
            row.update(status=actual['status'],scheduled=actual['total_debits'],
                       success=actual['total_success'],new_failures=actual['new_failures'])
        out['partial_closeout_present']=True
    return out


if __name__=='__main__':print(json.dumps(snapshot(),ensure_ascii=False))
