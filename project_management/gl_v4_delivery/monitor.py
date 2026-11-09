"""Read only tiny process-log tails; never inspect model/task outcomes."""
import json
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
HOME=ROOT/'rc3_gl_v4_20261109'


def monitor():
    rows=[]
    for phase in ('DEPLOYMENT','SCIENCE'):
        folder=HOME/'runs'/phase;final=folder/'PHASE_FINAL.json';log=folder/'process.log';row=dict(phase=phase,status='NOT_STARTED')
        if folder.exists():row['status']='RUNNING'
        if final.exists():
            f=json.loads(final.read_text(encoding='utf8'));row.update(status=f['status'],wall_s=f['wall_s'],error=f['error'])
        if log.exists():
            with log.open('rb') as f:f.seek(max(0,log.stat().st_size-4096));tail=f.read().decode('utf8','replace').splitlines()
            row['last_line']=tail[-1] if tail else ''
            if any('Traceback' in l for l in tail):row['error_tail']=tail[-8:]
        rows.append(row)
    return dict(utc=datetime.now(timezone.utc).isoformat(),phases=rows,
        new_task_bank_frozen=(HOME/'runs/SCIENCE/NEW_TASK_BANK_FROZEN.json').exists(),
        completed_arms=[a for a in ('A128','B128','C128') if (HOME/'runs/SCIENCE'/a/'COMPLETE.json').exists()])


if __name__=='__main__':print(json.dumps(monitor(),ensure_ascii=False))
