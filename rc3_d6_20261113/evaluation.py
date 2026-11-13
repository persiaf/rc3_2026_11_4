import collections,statistics,time
import numpy as np
from .common import HOME,read,put,state,unarchive,sha
from .runtime import policy_request,native_request,frozen_actor,deployed_actor

def result_rows(path):
    return read(path)['rows'] if path.exists() else []

def evaluate(budget,pool,method,seed,endpoint,actor=None,entry=None,*,tier='development',cases=None,purpose='eval',tag=None):
    cases=pool.cases if cases is None else cases;name=tag or f'{seed}_{method}_{endpoint}'
    path=HOME/'evaluations'/tier/(name+'.json');report=read(path) if path.exists() else dict(method=method,seed=seed,endpoint=endpoint,tier=tier,rows=[])
    existing={row['case_id'] for row in report['rows']}
    if report.get('complete') and len(existing)==len(cases):return report
    if actor is None and method not in ('Fixed3','Detour16'):actor=frozen_actor(budget) if method=='FrozenSL' else deployed_actor(entry,budget)
    for case in cases:
        if case['id'] in existing:continue
        rid=f'{tier}:{name}:{case["id"]}';folder='evaluations/'+tier+'/records'
        try:
            out=native_request(budget,pool,case['id'],rid,method,folder,purpose=purpose) if method in ('Fixed3','Detour16') else policy_request(
                budget,pool,case['id'],rid,actor,purpose=purpose,folder=folder,greedy=True,seed=seed,method=method)
            r=out['record'];row=dict(case_id=case['id'],physical_key=case['physical_key'],group=case.get('group',tier),mode=case['spec']['mode'],
                width=case['spec']['width'],height=case['spec']['height'],method=method,seed=seed,endpoint=endpoint,
                valid=True,metrics=r['metrics'],full_request_s=out['full_request_s'],record=out['ref'],
                actor_tensor_sha256=r.get('actor_tensor_sha256'),raw_semantics=r['raw_semantics'])
        except Exception as exc:
            row=dict(case_id=case['id'],physical_key=case['physical_key'],group=case.get('group',tier),mode=case['spec']['mode'],
                width=case['spec']['width'],height=case['spec']['height'],method=method,seed=seed,endpoint=endpoint,valid=False,metrics=None,full_request_s=None,error=repr(exc))
            report['rows'].append(row);put(path,report,path.exists());raise
        report['rows'].append(row);put(path,report,path.exists())
    report['complete']=len(report['rows'])==len(cases);report['summary']=summarize(report['rows']);put(path,report,True)
    return report

def summarize(rows):
    groups={'overall':rows}
    for group in sorted({r['group'] for r in rows}):groups[group]=[r for r in rows if r['group']==group]
    for size in ((4,3),(6,6),(8,8)):groups[f'{size[0]}x{size[1]}']=[r for r in rows if (r['width'],r['height'])==size]
    output={}
    for key,rr in groups.items():
        good=[r for r in rr if r['valid']];parent=collections.defaultdict(list)
        for row in good:parent[row['physical_key']].append(row)
        output[key]=dict(planned=len(rr),completed_valid=len(good),success_rate=len(good)/len(rr) if rr else None,
            physical_parents=len(parent),request_mean={m:float(np.mean([r['metrics'][m] for r in good])) if good else None for m in ('J','L','K','raw_J','raw_L','raw_K')},
            parent_equal_mean={m:float(np.mean([np.mean([r['metrics'][m] for r in pr]) for pr in parent.values()])) if parent else None for m in ('J','L','K','raw_J','raw_L','raw_K')},
            time_p50_s=float(np.median([r['full_request_s'] for r in good])) if good else None,time_p95_s=float(np.quantile([r['full_request_s'] for r in good],.95)) if good else None)
    return output

def gains(candidate,control,metric='J',group=None):
    c={r['case_id']:r for r in candidate if group is None or r['group']==group};b={r['case_id']:r for r in control if group is None or r['group']==group}
    if set(c)!=set(b) or not c or any(not r['valid'] for r in list(c.values())+list(b.values())):return dict(complete=False,mean=None,parents={})
    parents=collections.defaultdict(list)
    for key,row in c.items():parents[row['physical_key']].append(key)
    gp={}
    for parent,ids in parents.items():
        base=float(np.mean([b[i]['metrics'][metric] for i in ids]));cand=float(np.mean([c[i]['metrics'][metric] for i in ids]))
        if base<=0:return dict(complete=False,mean=None,parents={})
        gp[parent]=(base-cand)/base
    return dict(complete=True,mean=float(np.mean(list(gp.values()))),parents=gp)

def route_changes(pg,base):
    b={r['case_id']:r for r in base['rows']};rows=[]
    for row in pg['rows']:
        if not row['valid'] or not b[row['case_id']]['valid']:continue
        a=unarchive(row['record']);z=unarchive(b[row['case_id']]['record']);out=dict(case_id=row['case_id'],physical_key=row['physical_key'])
        for label,key in (('raw','raw_plan'),('H','plan')):
            aa=a[key]['visit_order_candidate_ids'];bb=z[key]['visit_order_candidate_ids']
            out[label+'_stop_set_changed']=set(aa)!=set(bb);out[label+'_visit_order_changed']=aa!=bb;out[label+'_same_set_order_changed']=set(aa)==set(bb) and aa!=bb
        rows.append(out)
    return rows

