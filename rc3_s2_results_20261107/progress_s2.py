"""Compact read-only progress from complete saved JSON lines, no model calls."""
from pathlib import Path
import json
import time

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01'

def last(p):
    if not p.exists():return None
    with p.open('rb') as f:
        # A complete full-scene batch contains all legal masks and can exceed
        # half a megabyte. This is a monitor buffer, not a scientific log cap.
        size=p.stat().st_size;f.seek(max(0,size-16*1024*1024));data=f.read()
    for line in reversed(data.splitlines()):
        try:return json.loads(line)
        except (ValueError,UnicodeError):pass
    return None

def snapshot():
    rows=[]
    for seed in [11622026,11622027,11622028]:
        adm=RUN/f'stage_admission/S2/{seed}'
        row=dict(seed=seed,state='NOT_STARTED',RL=0,continuedSL=0)
        if (adm/'START.json').exists():row['state']='RUNNING'
        for phase,key in [('RL','RL'),('continued_SL','continuedSL')]:
            p=RUN/f'seed_{seed}/{phase}/updates.jsonl';d=last(p)
            if d:
                row[key]=d['cursor_after'];row['stage_elapsed_s']=d['stage_elapsed_s']
                row['last_update_age_s']=round(time.time()-p.stat().st_mtime,1)
        for n,state in [('COMPLETE.json','COMPLETE'),('FAILED.json','FAILED')]:
            p=adm/n
            if p.exists():
                c=json.loads(p.read_text(encoding='utf8'));row['state']=state
                row['error']=c.get('error')
                for v in c.get('cumulative_stage_resources',{}).values():row['stage_elapsed_s']=v['elapsed_wall_s']
        rows.append(row)
    return rows

if __name__=='__main__':print(json.dumps(snapshot(),ensure_ascii=False))
