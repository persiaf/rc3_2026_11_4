"""Conditional T2 only: frozen students, no online terminal search for decisions."""
import csv
import time
import numpy as np
from rc3_constructive.problem import Cache
from rc3_constructive.evaluate import load_policy,greedy_routes
from rc3_nav_value.r1.policy import State
from rc3_nav_value.r1.oracle import _append
from .register import ROOT,OUT,read,write,digest
from .rank_model import predict,load
from .evaluate_actions import bootstrap,chosen_index


def construct(x,item,B,method):
    s=State(item)
    while not s.complete:
        x.check();menu=x.menu(item,B,s.prefix)
        if method=='menu_rule':
            from .candidate_menu import rule_action
            a=rule_action(item,menu,s.prefix)
        else:
            pred,_=predict(x.models[method],x.features(item,B,s.prefix,menu),x.stats)
            a=menu[chosen_index(menu,pred)]
        s.step(a);x.stats['student_decisions']+=1
    x.stats['raw_student_trajectories']+=1
    return s.route()


def request(x,scene,case,method,cachemode,rep,phase,old):
    cache=Cache(2)
    if cachemode=='hot':
        x.prepare(scene,cache);x.stats['hot_priming']+=1
    tick=time.perf_counter();pt=time.perf_counter();item=x.prepare(scene,cache)
    prep=time.perf_counter()-pt;Bcost=0.;decision=0.
    meta=dict(case=case,method=method,cache_mode=cachemode,rep=rep,phase=phase)
    if method in ('fixed3','detour16'):
        row=x.oracle.native(item,method,'T2_routes',meta)
        if row['fallback']:raise RuntimeError('T2 native fallback')
    else:
        B=None
        if method in ('gBH','menu_rule','linear','neural'):
            b=x.oracle.native(item,'fixed3','T2_routes',dict(**meta,purpose='B_generation_inside_request'))
            if b['fallback']:raise RuntimeError('T2 B fallback')
            B=b['route'][1:-1];Bcost=b['oracle_s']
        dt=time.perf_counter()
        if method=='gBH':raw=x.gb(item,B)
        elif method=='old128':
            raw=greedy_routes(old,[item],'cpu')[0][0]
            x.stats['old_model_trajectories']+=1;x.stats['old_model_decisions']+=len(raw)-2
        else:raw=construct(x,item,B,method)
        decision=time.perf_counter()-dt
        row=x.oracle.full(item,raw,'T2_routes',dict(**meta,completion_s=decision))
    wall=time.perf_counter()-tick
    total=prep+Bcost+decision+row['oracle_s']
    plan=row.pop('plan',None)
    result={**row,**meta, 'preparation_s':prep,'B_generation_s':Bcost,'decision_s':decision,
        'total_s':total,'measured_request_wall_s':wall,'bookkeeping_residual_s':wall-total}
    if phase=='quality' and plan is not None:
        write(OUT/'plans'/f'{case:02}_{method}.json',plan)
    _append(OUT/'T2_requests.jsonl',result)
    return result


def teacher(x,scene,sha,case):
    tick=time.perf_counter();item=x.prepare(scene,Cache(0))
    b=x.oracle.native(item,'fixed3','T2_teacher',dict(case=case,purpose='offline_paired_teacher_B'))
    if b['fallback']:raise RuntimeError('Teacher B fallback')
    B=b['route'][1:-1];s=State(item);checkpoint=0;selected=None;root=None;trace=[]
    thresholds=[0,.25,.5,.75]
    while not s.complete:
        if checkpoint<len(thresholds) and 1-s.remaining.mean()+1e-12>=thresholds[checkpoint]:
            menu=x.menu(item,B,s.prefix)
            values=[x.label(item,B,sha,s.prefix,a,'T2_teacher',dict(case=case,checkpoint=checkpoint,purpose='offline_teacher')) for a in menu]
            if root is None:root=values[0]
            best=min(range(len(menu)),key=lambda i:(values[i]['J'],menu[i]))
            selected=values[best]
            trace.append(dict(prefix=list(s.prefix),menu=menu,Y=[r['J'] for r in values],chosen=menu[best]))
            s.step(menu[best]);checkpoint+=1
        else:s.step(next(a for a in B if s.legal_mask()[a-1]))
    assert selected is not None and root is not None
    assert s.route()==selected['raw_route']
    assert selected['J']<=root['J']+1e-7
    result=dict(case=case,gBH_J=root['J'],teacher_J=selected['J'],raw_route=s.route(),
        gBH_raw_route=root['raw_route'],gBH_final_route=root['route'],
        final_route=selected['route'],trace=trace,elapsed_s=time.perf_counter()-tick,
        root_call_id=root['call_id'],final_call_id=selected['call_id'],online_student_queries=0)
    _append(OUT/'T2_teacher.jsonl',result)
    return result


def summarize(x,quality,timing,teachers):
    methods=x.p['T2']['methods'];summary=[]
    by={(r['case'],r['method']):r for r in quality}
    for t in teachers:
        g=by[t['case'],'gBH']
        assert abs(g['J']-t['gBH_J'])<=1e-7
        assert g['raw_route']==t['gBH_raw_route'] and g['route']==t['gBH_final_route']
    for method in methods:
        q=[r for r in quality if r['method']==method]
        row=dict(method=method,cases=len(q),mean_J=float(np.mean([r['J'] for r in q])),
            mean_L=float(np.mean([r['L'] for r in q])),mean_K=float(np.mean([r['K'] for r in q])),
            feasible_percent=100*np.mean([r['valid'] for r in q]))
        for mode in ('cold','hot'):
            v=[r['total_s'] for r in timing if r['method']==method and r['cache_mode']==mode]
            row[mode+'_median_s']=float(np.median(v));row[mode+'_p95_s']=float(np.quantile(v,.95))
        summary.append(row)
    comparisons={}
    for baseline in ('gBH','menu_rule','linear','fixed3','detour16','old128'):
        values=[100*(by[c,baseline]['J']-by[c,'neural']['J'])/by[c,baseline]['J'] for c in range(36)]
        comparisons[baseline]=dict(mean_gain_percent=float(np.mean(values)),CI95=bootstrap(values))
    qmap={r['method']:r for r in summary};nn=qmap['neural'];det=qmap['detour16'];fixed=qmap['fixed3']
    gains=comparisons['gBH'];rule=comparisons['menu_rule']
    denominator=sum(t['gBH_J']-t['teacher_J'] for t in teachers)
    numerator=sum(by[t['case'],'gBH']['J']-by[t['case'],'neural']['J'] for t in teachers)
    compression=numerator/denominator if denominator>1e-8 else None
    mechanism=bool(gains['mean_gain_percent']>=.5 and gains['CI95'][0]>0 and rule['CI95'][0]>0 and nn['feasible_percent']==100)
    degradation=np.array([100*(by[c,'neural']['J']-by[c,'detour16']['J'])/by[c,'detour16']['J'] for c in range(36)])
    rng=np.random.default_rng(11522026)
    upper=float(np.quantile(degradation[rng.integers(36,size=(20000,36))].mean(1),.95))
    gates={}
    for mode in ('cold','hot'):
        dominated=fixed['mean_J']<=nn['mean_J'] and fixed[mode+'_median_s']<=nn[mode+'_median_s'] and fixed[mode+'_p95_s']<=nn[mode+'_p95_s']
        gates[mode]=bool(upper<=.5 and nn[mode+'_median_s']/det[mode+'_median_s']<=.7 and nn[mode+'_p95_s']/det[mode+'_p95_s']<=1 and not dominated)
    result=dict(summary=summary,comparisons=comparisons,mechanism_passed=mechanism,
        paired_teacher_total_savings=denominator,student_total_savings=numerator,
        teacher_compression_fraction=compression,compression_passed=bool(mechanism and compression is not None and compression>=1/3),
        deployment_passed=gates,detour_degradation_one_sided95_upper_percent=upper,
        T3_authorized=False)
    write(OUT/'T2_RESULTS.json',result)
    with (OUT/'quality_time.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    x.check()
    write(OUT/'STATUS.json',dict(T0='complete',T1='passed',T2='complete',
        T2_mechanism=mechanism,T2_compression=result['compression_passed'],T2_deployment=gates,
        T3='not_authorized',retained60='sealed',default_methods=['Fixed3','Detour']))


def routes(x):
    assert read(OUT/'T1_GATE.json')['passed']
    frozen=read(OUT/'T1_GATE.json')['frozen_model_sha256']
    for kind,h in frozen.items():assert digest(OUT/'weights'/f'{kind}.pt')==h
    load_times={}
    for kind in frozen:
        x.models[kind],load_times[kind]=load(OUT/'weights'/f'{kind}.pt',digest(OUT/'protocol.json'))
    tick=time.perf_counter();old,load_s=load_policy(ROOT/'rc3_constructive/outputs/checkpoints/update0128.pt','cpu')
    write(OUT/'T2_model_load.json',dict(old_s=load_s,student_load_s=load_times,scope='saved frozen checkpoints reloaded; model load outside deployment request and separately timed'))
    methods=x.p['T2']['methods'];quality=[];timing=[];teachers=[];rng=np.random.default_rng(11522026)
    for case in range(36):
        scene,sha=x.scene('T2',case)
        for method in rng.permutation(methods):
            quality.append(request(x,scene,case,str(method),'cold',-1,'quality',old))
        teachers.append(teacher(x,scene,sha,case))
        jobs=[(m,mode,rep) for m in methods for mode in ('cold','hot') for rep in range(3)]
        for idx in rng.permutation(len(jobs)):
            m,mode,rep=jobs[idx]
            row=request(x,scene,case,m,mode,rep,'timing',old);timing.append(row)
            reference=next(r for r in quality if r['case']==case and r['method']==m)
            assert row['raw_route']==reference['raw_route'] and row['route']==reference['route'] and abs(row['J']-reference['J'])<1e-7
        x.resources()
        print({'stage':'T2','cases':case+1,'calls':x.ledger.total,'elapsed_T2_s':time.perf_counter()-tick},flush=True)
    summarize(x,quality,timing,teachers)
