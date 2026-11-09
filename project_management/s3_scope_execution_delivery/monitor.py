"""Small read-only progress snapshots; no task/model/navigation work."""
import json
import re
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'rc3_s3_connectivity_20261109/runs/rc3_v3_global_local_20261106_01/scope_r1'


def snapshot():
    out=dict(utc=datetime.now(timezone.utc).isoformat(),seeds=[])
    for seed,olddebits,oldsuccess in ((11622026,2059,2057),(11622027,0,0),(11622028,0,0)):
        folder=RUN/str(seed)
        row=dict(seed=seed,status='NOT_STARTED',inherited_debits=olddebits,inherited_success=oldsuccess)
        if folder.exists():
            row.update(status='STARTED',anchor_retry_receipts=sum(1 for _ in (folder/'io_receipts').glob('*.json')))
            logs=list(folder.glob('session_*.log'))
            if logs:
                p=max(logs,key=lambda x:x.stat().st_mtime_ns)
                with p.open('rb') as f:
                    f.seek(max(0,p.stat().st_size-4096));tail=f.read().decode('utf8',errors='replace')
                lines=tail.splitlines();row['last_line']=lines[-1] if lines else ''
                for line in reversed(lines):
                    match=re.search(r'(\d+) scheduled=(\d+)/5292 successful=(\d+)',line)
                    if match:
                        row.update(last_successful_schedule=int(match[2]),total_success=int(match[3]),
                            new_success=int(match[3])-oldsuccess);break
            finals=list((folder/'sessions').glob('*/FINAL.json'))
            if finals:
                p=max(finals,key=lambda x:x.stat().st_mtime_ns);final=json.loads(p.read_text(encoding='utf8'))
                row.update(status='SESSION_'+final['status'],sealed_session_wall_s=final['elapsed_s'],error=final.get('error'))
            if (folder/'SCHEDULE_SEAL.json').exists():row['status']='SCHEDULE_DISPOSED_SEALED'
        out['seeds'].append(row)
    out['analysis_present']=(RUN/'SCIENTIFIC_ANALYSIS_SCOPE_R1.json').exists()
    return out


if __name__=='__main__':print(json.dumps(snapshot(),ensure_ascii=False))
