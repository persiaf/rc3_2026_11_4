"""Registered disjoint-distribution quality/time AND tests, real request times."""
from .common import *
from .statistics import paired_effects


def tables(quality,timings,ready):
    rows=[];totals=[];contrasts=[]
    for group in ('all','final_seen','final_unseen'):
        for size in ('all',(4,3),(6,6),(8,8)):
            take=lambda r:(group=='all' or r['spec']['role']==group) and (size=='all' or (r['spec']['width'],r['spec']['height'])==size)
            q=[r for r in quality if take(r)];tt=[r for r in timings if take(r)]
            if not q:continue
            effects={c:paired_effects(q,c) for c in ('SL256','residual_SL','Fixed3','Detour16')}
            for c,e in effects.items():contrasts.append(dict(group=group,size=size,reference=c,**e))
            bound=-effects['Detour16']['one_sided95_lower']
            fixed_quality_dominates=effects['Fixed3']['mean']<=0
            for cache in ('cold','hot'):
                sample=[r for r in tt if r['request']['cache']==cache]
                for boundary in ('service_time_before_IO_s','_audit_time_s'):
                    by={}
                    for role in ('RL','Fixed3','Detour16'):
                        by[role]=[r[boundary] for r in sample if r['study_method']==role]
                    require(all(by.values()) and all(all(v is not None and np.isfinite(v) for v in a) for a in by.values()),'Missing/nonfinite timing cell')
                    med={k:float(np.median(v)) for k,v in by.items()};p95={k:float(np.quantile(v,.95)) for k,v in by.items()}
                    median_ratio=med['RL']/med['Detour16'];p95_ratio=p95['RL']/p95['Detour16']
                    dominated=fixed_quality_dominates and med['Fixed3']<=med['RL'] and (effects['Fixed3']['mean']<0 or med['Fixed3']<med['RL'])
                    go=bound<=.005 and median_ratio<=.7 and p95_ratio<=1. and not dominated
                    rows.append(dict(group=group,size=size,cache=cache,boundary=boundary,RL_requests=len(by['RL']),native_requests=len(by['Detour16']),
                        Detour_cost_degradation_one_sided95_upper=bound,median=med,p95=p95,median_ratio=median_ratio,p95_ratio=p95_ratio,
                        Fixed3_dominates=dominated,strict_deployment_PASS=go))
            # Totals use the first actual N completed measured service durations,
            # not N times a mean or a subtraction from old audited times.
            for cache in ('cold','hot'):
                for seed,role in [(s,r) for s in SEEDS for r in ('SL256','residual_SL','RL_raw','RL')]+[(None,'Fixed3'),(None,'Detour16')]:
                    stream=[r for r in tt if r['request']['cache']==cache and r['seed']==seed and r['study_method']==role]
                    for N in (1,10,100):
                        ok=len(stream)>=N
                        totals.append(dict(group=group,size=size,cache=cache,seed=seed,role=role,N=N,ready_s=ready['ready_s'],
                            status='MEASURED_SEQUENCE' if ok else 'INSUFFICIENT_MEASURED_REQUESTS',
                            ready_plus_actual_service_s=ready['ready_s']+sum(r['service_time_before_IO_s'] for r in stream[:N]) if ok else None,
                            measured_requests=len(stream)))
    return rows,totals,contrasts
