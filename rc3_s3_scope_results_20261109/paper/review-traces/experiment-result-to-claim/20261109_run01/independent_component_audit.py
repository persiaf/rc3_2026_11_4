"""Compare the newly provided component profile with saved CSV, arithmetic only."""
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parent
RES=ROOT/'rc3_s3_scope_results_20261109/results'
tick=time.perf_counter()
start=datetime.now(timezone.utc).isoformat()
rows=list(csv.DictReader((RES/'REQUEST_METRICS.csv').open(encoding='utf-8-sig',newline='')))
profile=json.loads((RES/'SAVED_COMPONENT_PROFILE.json').read_text(encoding='utf-8'))
checks=0
for cell in profile['components']:
    rr=[r for r in rows if r['method']==cell['method'] and r['kind']==cell['kind']]
    assert len(rr)==cell['requests']
    for component,reported in cell['components'].items():
        a=[float(r[component]) for r in rr if r[component]!='']
        assert len(a)==reported['observed']
        if a:
            assert math.isclose(float(np.median(a)),reported['median'],rel_tol=1e-12,abs_tol=1e-10)
            assert math.isclose(float(np.quantile(a,.95)),reported['p95'],rel_tol=1e-12,abs_tol=1e-10)
        else:
            assert reported['median'] is None and reported['p95'] is None
        checks+=1
selected=[]
for method in ['RL_plus_H','frozen_supervised','continued_SL']:
    rr=[r for r in rows if r['method']==method and r['kind']=='hot']
    selected.append(dict(method=method,n=len(rr),**{c:float(np.median([float(r[c]) for r in rr])) for c in
        ['complete_time_s','model_load_and_verify_s','features_and_global_local_encoding_s','greedy_construction_s','common_H_pipeline_s']}))
result=dict(status='PASS',started_utc=start,finished_utc=datetime.now(timezone.utc).isoformat(),
            wall_s=time.perf_counter()-tick,component_cells_compared=checks,hot_selected=selected,
            scientific_calls=0,scope='Saved inclusive component observations only. No summing medians or subtractive counterfactual speed claim.')
(OUT/'independent_component_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
