"""Additional saved-data arithmetic: raw/raw attribution and old timing ledgers.

No project imports and no scientific calls. This does not fill either old slot.
"""
import csv
import hashlib
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
RES = ROOT / 'rc3_s3_scope_results_20261109/results'
START=time.perf_counter()
START_UTC=datetime.now(timezone.utc).isoformat()
SEEDS=(11622026,11622027,11622028)

def identity(v):
    return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def write(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

rows=list(csv.DictReader((RES/'REQUEST_METRICS.csv').open(encoding='utf-8-sig',newline='')))
index={r['request_id']:r for r in rows}
old_roots=sorted({str((ROOT/r['artifact_path']).parent.parent) for r in rows
                  if r['persistence_revision']!='311a58bd867e95c852ec3246c54a398468eb526db391fbb6d12239c6a3379949'})
old=[]
old_ids=set()
for folder in map(Path,old_roots):
    prev='0'*64
    counts=Counter()
    for i,line in enumerate((folder/'journal.jsonl').open(encoding='utf-8')):
        e=json.loads(line)
        h=e.pop('sha256')
        assert identity(e)==h and e['previous']==prev and e['sequence']==i
        prev=h
        counts[e['event']]+=1
        if e['event']=='request_complete':
            rid=e['id']
            assert rid not in old_ids
            old_ids.add(rid)
            assert e['artifact_sha256']==index[rid]['artifact_sha256']
            assert math.isclose(e['complete_time_s'],float(index[rid]['complete_time_s']),rel_tol=1e-12,abs_tol=1e-10)
    old.append(dict(journal=str((folder/'journal.jsonl').relative_to(ROOT)),event_counts=dict(counts),tail=prev))
assert len(old_ids)==2057
write('independent_old_timing_ledger.json',dict(status='PASS',old_successful_complete_clocks_checked=2057,
      old_journals=old, original_dirty_head_preserved=True,
      scope='Byte-hash frozen original journal plus full append chain; do not rewrite historical original head.',scientific_calls=0))

raw=[]
for group,n in [('seen',6),('unseen',12)]:
    for size in ['4x3','6x6','8x8']:
        q=[r for r in rows if r['group']==group and r['size']==size and r['kind']=='quality']
        parents=sorted({r['physical_key'] for r in q})
        assert len(parents)==n
        ix=np.random.Generator(np.random.PCG64(11632026)).integers(n,size=(20000,n))
        for control in ['frozen_supervised','continued_SL','untrained']:
            costs={}
            for method in ['RL_raw',control]:
                costs[method]=np.empty((n,3))
                for pi,p in enumerate(parents):
                    for si,seed in enumerate(SEEDS):
                        rr=[r for r in q if r['physical_key']==p and int(r['seed'])==seed and r['method']==method]
                        assert len(rr)==2 and {r['endpoint'] for r in rr}=={'closed_depot','open_first_last_connected'}
                        costs[method][pi,si]=np.mean([float(r['raw_J']) for r in rr])
            pooled={m:c.mean(1) for m,c in costs.items()}
            v=(pooled[control]-pooled['RL_raw'])/pooled[control]
            bs=v[ix].mean(1)
            sg=[float(((costs[control][:,i]-costs['RL_raw'][:,i])/costs[control][:,i]).mean()) for i in range(3)]
            raw.append(dict(group=group,size=size,actor='RL_raw_saved_raw_J',control=control+'_saved_raw_J',
                       nphysical_parents=n,mean=float(v.mean()),lower95=float(np.quantile(bs,.025)),
                       upper95=float(np.quantile(bs,.975)),seed_gains=sg,
                       scope='Saved quality raw/raw arithmetic only; auxiliary attribution, does not replace any registered H quality cell'))
write('independent_raw_vs_raw.json',raw)
result=dict(status='PASS',started_utc=START_UTC,finished_utc=datetime.now(timezone.utc).isoformat(),
            wall_s=time.perf_counter()-START,scientific_calls=0,old_complete_clocks_checked=len(old_ids),
            raw_raw_comparisons=len(raw),project_module_imports=0)
write('independent_supplemental_audit.json',result)
print(json.dumps(result,indent=2))
for r in raw:
    print(r['group'],r['size'],r['control'],r['mean'],r['lower95'],r['upper95'])
