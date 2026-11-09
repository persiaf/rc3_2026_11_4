"""Supplementary static checks; no policy/optimizer/planner is instantiated."""
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch

TRACE = Path(__file__).resolve().parent
HOME = TRACE.parents[3]
ROOT = HOME.parent
begin = time.perf_counter()
torch.set_grad_enabled(False)


def denied(*args, **kwargs):
    raise AssertionError('Scientific execution forbidden')


torch.nn.Module.__call__ = denied
torch.optim.Optimizer.__init__ = denied
torch.autograd.backward = denied
torch.Tensor.backward = denied


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for v in iter(lambda:f.read(4*1024*1024),b''):h.update(v)
    return h.hexdigest()


def exact(a,b):
    return set(a)==set(b) and all(a[k].dtype==b[k].dtype and a[k].shape==b[k].shape and torch.equal(a[k],b[k]) for k in a)


def check(ok,msg):
    if not ok:raise AssertionError(msg)


out={}
dev=[read(p) for p in (HOME/'runs/DEV54/results').glob('*.json')]
absolute=[]
for arm in ['SL256','A128','B128']:
    rows=[r for r in dev if r['study_method']==arm]
    absolute.append(dict(model=arm,n=len(rows),**{k:float(np.mean([r['metrics'][k] for r in rows])) for k in ['J','L','K','raw_J','raw_L','raw_K']},
                         H_gain_J=float(np.mean([r['H_gain_J'] for r in rows]))))
published=list(csv.DictReader((HOME/'results/DEV_MODEL_SUMMARY.csv').open(encoding='utf8')))
for a,b in zip(absolute,published):
    check(a['model']==b['model'] and a['n']==int(b['n']),'Absolute model summary identity')
    check(all(math.isclose(a[k],float(b[k]),rel_tol=1e-12,abs_tol=1e-10) for k in a if k not in ['model','n']),'Absolute model summary arithmetic')
by={d['model']:d for d in absolute}
out['absolute_case_means']=absolute
out['absolute_mean_relative_gains']=[dict(control=c,metric=m,relative_gain=(by[c][m]-by['B128'][m])/by[c][m]) for c in ['A128','SL256'] for m in ['raw_J','J']]
out['aggregation_explanation']='The ratio of grand absolute means weights relative changes by control costs; preregistered mean of parent relative gains weights the three physical parents equally. Both must be reported; the candidate gate uses only the preregistered statistic.'
out['case_wins']=[]
index={(r['case_id'],r['study_method']):r for r in dev}
for c in ['A128','SL256']:
    for metric in ['raw_J','J']:
        delta=np.array([index[cid,c]['metrics'][metric]-index[cid,'B128']['metrics'][metric] for cid in sorted({r['case_id'] for r in dev})])
        out['case_wins'].append(dict(control=c,metric=metric,better=int(np.sum(delta>1e-9)),equal=int(np.sum(np.abs(delta)<=1e-9)),worse=int(np.sum(delta < -1e-9))))

checkpoints=[]
for arm in ['A128','B128']:
    cp=HOME/'runs'/arm/'checkpoints/update_0000.pt'
    side,commit=read(cp.with_suffix('.pt.json')),read(cp.with_suffix('.pt.commit.json'))
    check(sha(cp)==side['sha256']==commit['checkpoint_sha256'] and sha(cp.with_suffix('.pt.json'))==commit['sidecar_sha256'],'Initial checkpoint hashes')
    payload=torch.load(cp,map_location='cpu',weights_only=False)
    check(not payload['optimizer']['state'] and payload['terminal_state']['request_count']==0 and not payload['streams']['torch'] and not payload['streams']['numpy'],'Initial optimizer/RNG/query cursor not fresh')
    check(payload['extra']['new_experiment']=='rc3_actor_pg_screen_20261109_01','Initial experiment source')
    actor=read(HOME/'screen_actors'/f'11622026_{arm}.pt.json')
    saved_actor=torch.load(ROOT/actor['path'],map_location='cpu',weights_only=True)
    endpoint=torch.load(HOME/'runs'/arm/'checkpoints/update_0128.pt',map_location='cpu',weights_only=False)
    check(exact(saved_actor['model'],endpoint['model']),'Screen actor endpoint equality')
    check(actor['provenance']['training_checkpoint_sha256']==sha(HOME/'runs'/arm/'checkpoints/update_0128.pt') and sha(ROOT/actor['path'])==actor['checkpoint_sha256'],'Screen actor source mapping')
    check(all(r['weight']['model_tensor_sha256']==actor['model_tensor_sha256'] and r['weight']['checkpoint_sha256']==actor['checkpoint_sha256'] for r in dev if r['study_method']==arm),'DEV used wrong endpoint weights')
    checkpoints.append(dict(arm=arm,initial_checkpoint_sha256=sha(cp),screen_actor_exact=True,screen_actor_sha256=actor['checkpoint_sha256'],endpoint_tensor_sha256=endpoint['model_tensor_sha256']))
out['additional_checkpoint_checks']=checkpoints

timing=list(csv.DictReader((HOME/'results/DEPLOYMENT_TIMING.csv').open(encoding='utf8')))
by_timing={(r['method'],r['mode'],r['navigation']):r for r in timing}
speedup=[]
for row in csv.DictReader((HOME/'results/DEPLOYMENT_SPEEDUP.csv').open(encoding='utf8')):
    baseline,new=by_timing[row['method'],'D0',row['navigation']],by_timing[row['method'],row['mode'],row['navigation']]
    values=dict(D0_median_s=float(baseline['median_s']),new_median_s=float(new['median_s']),
                median_speedup=float(baseline['median_s'])/float(new['median_s']),median_reduction=1-float(new['median_s'])/float(baseline['median_s']),
                D0_p95_s=float(baseline['p95_s']),new_p95_s=float(new['p95_s']),p95_speedup=float(baseline['p95_s'])/float(new['p95_s']))
    check(all(math.isclose(v,float(row[k]),rel_tol=1e-12,abs_tol=1e-12) for k,v in values.items()),'Published speedup arithmetic')
    speedup.append(dict(method=row['method'],navigation=row['navigation'],mode=row['mode'],**values))
out['speedup_arithmetic_checked']=len(speedup)
out['speedup']=speedup

first=[]
new_navigation=0
inference_H=0
native=0
for phase,mode in [('DEPLOYMENT','D0'),('DEPLOYMENT','D1'),('DEPLOYMENT','D2'),('DEV54',None)]:
    folder=HOME/'runs'/phase
    if mode:folder=folder/mode
    events=[json.loads(l) for l in (folder/'events.jsonl').read_text(encoding='utf8').splitlines()]
    completes=[e for e in events if e['event']=='request_complete']
    new_navigation+=sum(e['counts'].get('navigation_solves',0) for e in completes)
    new_navigation+=sum(e['counts'].get('navigation_solves',0) for e in events if e['event']=='hot_prime')
    inference_H+=sum(e['counts'].get('common_H_invocations',0) for e in completes)
    native+=sum(e['counts'].get('native_pipeline_completed',0) for e in completes)
    if mode:
        s=read(folder/'STARTUP.json')
        request=next(e['request'] for e in events if e['event']=='request_start')
        e=completes[0]
        first.append(dict(mode=mode,first_request=request,process_to_ready_s=s['process_to_ready_s'],first_complete_request_s=e['complete_time_s'],
                          startup_plus_first_request_spans_s=s['process_to_ready_s']+e['complete_time_s']))
supplied=read(HOME/'results/STARTUP_AND_FIRST_RESPONSE.json')
for d in first:
    p=supplied[d['mode']]
    check(p['first_request']==d['first_request'] and all(math.isclose(d[k],p[k],rel_tol=1e-12,abs_tol=1e-12) for k in ['process_to_ready_s','first_complete_request_s','startup_plus_first_request_spans_s']),'Startup span sum mismatch')
out['startup_and_first_spans']=first
out['startup_latency_limit']='These are two measured spans, excluding intervening STARTUP receipt/request predebit IO; they are not exact process-to-first-response latency. First request method also differs across modes.'
out['complete_resource_reconciliation']=dict(actual_new_navigation_solves=new_navigation,inference_H_invocations=inference_H,
    training_H_invocations=3072,total_H_invocations=inference_H+3072,native_complete_requests=native,
    all_scientific_complete_requests=756+3072+54)
check(new_navigation==435 and inference_H==486 and native==216,'Supplementary resource counts')
out['new_scientific_calls']=0
out['audit_wall_s']=time.perf_counter()-begin
out['status']='PASS'
(TRACE/'additional_saved_data_checks.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in out.items() if k in ['status','audit_wall_s','absolute_case_means','absolute_mean_relative_gains','complete_resource_reconciliation']},ensure_ascii=False))
