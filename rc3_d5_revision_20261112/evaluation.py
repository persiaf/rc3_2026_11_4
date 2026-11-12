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
            if method not in ('Fixed3','Detour16'):report['rows'].append(row);put(path,report,path.exists());raise
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

def baselines(budget,pool):
    # Fixed models are computed once in this campaign, shared for deterministic
    # quality only. These times are not reused as later deployment timing.
    outputs={}
    for method in ('FrozenSL','Fixed3','Detour16'):outputs[method]=evaluate(budget,pool,method,0,0,tag='baseline_'+method)
    return outputs

def p1_gate(pg,sl,base):
    comparisons={}
    for name,control in [('FrozenSL',base),('SL-U',sl)]:
        comparisons[name]={m:gains(pg['rows'],control['rows'],m) for m in ('J','raw_J')}
        comparisons[name]['groups']={g:gains(pg['rows'],control['rows'],'J',g) for g in ('trained_parent_new_tasks_exposed_selection','independent_parent_exposed_development')}
    complete=all(r['valid'] for r in pg['rows']) and all(comparisons[n]['J']['complete'] for n in comparisons)
    passed=complete and comparisons['FrozenSL']['J']['mean']>=.01 and comparisons['SL-U']['J']['mean']>=.005
    passed=passed and all(v['complete'] and v['mean']>=0 for n in comparisons.values() for v in n['groups'].values())
    return dict(status='PROMISING' if passed else 'NO_DEVELOPMENT_SIGNAL' if complete else 'INCOMPLETE',passed=bool(passed),comparisons=comparisons,
        all_66_candidate_requests_valid=all(r['valid'] for r in pg['rows']),endpoints_used=[4096],development_exposed=True)

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

def frozen_source_id():
    return dict(scientific_identity=read(HOME/'SCIENTIFIC_IDENTITY.json')['identity'],source_sha256=sha(HOME/'SCIENTIFIC_SOURCE_BINDING.json'))

def average_seed_rows(reports):
    indexed=[{r['case_id']:r for r in rep['rows']} for rep in reports];out=[]
    if any(set(x)!=set(indexed[0]) for x in indexed):raise ValueError('Seed case grids differ')
    for key,row in indexed[0].items():
        rr=[x[key] for x in indexed];valid=all(x['valid'] for x in rr);copy=dict(row,valid=valid)
        copy['metrics']={m:float(np.mean([x['metrics'][m] for x in rr])) for m in ('J','L','K','raw_J','raw_L','raw_K')} if valid else None;out.append(copy)
    return out

def independent_gate(candidate_reports,control_reports,*,statistics_seed=2026111199):
    # Three seed rows aggregate within parent before inference. Deployment seed
    # is separately tested; no seed/model selection by held-out performance.
    deploy=candidate_reports[0];comparisons={};all_pass=all(all(r['valid'] for r in x['rows']) for x in candidate_reports)
    for name,ctrls in control_reports.items():
        per=[gains(a['rows'],b['rows']) for a,b in zip(candidate_reports,ctrls)]
        if not all(x['complete'] for x in per):comparisons[name]=dict(complete=False);all_pass=False;continue
        dg=per[0];rows={r['physical_key']:r for r in deploy['rows']};mixed=[p for p in dg['parents'] if rows[p]['mode']=='mixed'];blocked=[p for p in dg['parents'] if rows[p]['mode']=='blocked']
        rng=np.random.default_rng(statistics_seed);draw=np.empty(20000)
        for i in range(20000):draw[i]=np.mean([dg['parents'][p] for ids in (mixed,blocked) for p in rng.choice(ids,len(ids),replace=True)])
        lo=float(np.quantile(draw,.05));seed_means=[x['mean'] for x in per]
        mean_across_seeds=gains(average_seed_rows(candidate_reports),average_seed_rows(ctrls))['mean']
        ok=dg['mean']>=.005 and lo>0 and mean_across_seeds>0
        comparisons[name]=dict(complete=True,deployment_parent_mean=dg['mean'],lower_one_sided95=lo,seed_means=seed_means,three_seed_parent_mean=mean_across_seeds,passed=bool(ok));all_pass=all_pass and ok
    simultaneous=sum(all(comparisons[n].get('seed_means',[-1]*3)[i]>0 for n in comparisons) for i in range(3))
    return dict(passed=bool(all_pass and simultaneous>=2),comparisons=comparisons,seeds_simultaneously_positive=simultaneous,bootstrap=20000)
