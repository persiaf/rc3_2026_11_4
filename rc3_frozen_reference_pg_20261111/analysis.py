"""Predeclared complete-route quality comparison; no diagnostic surrogate gate."""
from collections import defaultdict
import csv
import numpy as np
from .common import put,csvrow

PAIRS=(('PG-cond','FrozenSL'),('PG-cond','continued_SL_A1'),('PG-cond','PG-global'),
       ('PG-global','FrozenSL'),('PG-global','continued_SL_A1'),('PG-cond','Fixed3'),('PG-cond','Detour16'))
def evaluate(path,endpoint):
    with (path/'REQUEST_INDEX.csv').open(encoding='utf-8',newline='') as f:
        rows=[r for r in csv.DictReader(f) if r['kind']=='evaluation' and int(r['opportunity'])==endpoint]
    with (path/'REQUEST_TIMING.csv').open(encoding='utf-8',newline='') as f:
        timing={r['request_id']:float(r['full_request_s']) for r in csv.DictReader(f)}
    for r in rows:r['request_wall_s']=timing[r['request_id']]
    if len(rows)!=396:raise ValueError('Endpoint incomplete; no success-only analysis')
    results=[]
    cohorts=sorted({r['cohort'] for r in rows})
    for cohort in ['ALL',*cohorts]:
        for size in ('ALL','4x3','6x6','8x8'):
            selected=[r for r in rows if (cohort=='ALL' or r['cohort']==cohort) and (size=='ALL' or r['width']+'x'+r['height']==size)]
            if not selected:continue
            for a,b in PAIRS:
                cells={}
                for r in selected:
                    if r['role'] in (a,b):cells.setdefault((r['physical_key'],r['case_id']),{})[r['role']]=r
                if any(set(v)!={a,b} for v in cells.values()):raise ValueError('Unpaired complete-route data')
                groups=defaultdict(list)
                for (parent,_),v in cells.items():groups[parent].append(v)
                parents=sorted(groups)
                draws=np.random.Generator(np.random.PCG64(11652026)).integers(0,len(parents),size=(20000,len(parents)))
                for metric in ('J','raw_J','L','K','request_wall_s'):
                    means=[]
                    for parent in parents:
                        va=np.mean([float(v[a][metric]) for v in groups[parent]]);vb=np.mean([float(v[b][metric]) for v in groups[parent]])
                        means.append((vb-va)/vb)
                    means=np.array(means);lo,hi=np.quantile(means[draws].mean(1),[.025,.975],method='linear')
                    results.append(dict(endpoint=endpoint,cohort=cohort,size=size,method=a,control=b,metric=metric,
                        physical_parents=len(parents),relative_saving=float(means.mean()),ci95_low=float(lo),ci95_high=float(hi)))
    for r in results:csvrow(path/'QUALITY_CONTRASTS.csv',r)
    checks=[]
    for b in ('FrozenSL','continued_SL_A1','PG-global'):
        for cohort in cohorts:
            for metric in ('J','raw_J'):
                r=next(r for r in results if (r['method'],r['control'],r['cohort'],r['size'],r['metric'])==('PG-cond',b,cohort,'ALL',metric))
                checks.append(dict(**r,passed=r['relative_saving']>=.005 and r['ci95_low']>0))
    size_checks=[]
    for b in ('FrozenSL','continued_SL_A1'):
        for size in ('4x3','6x6','8x8'):
            for metric in ('J','raw_J'):
                r=next(r for r in results if (r['control'],r['cohort'],r['size'],r['metric'])==(b,'ALL',size,metric) and r['method']=='PG-cond')
                size_checks.append(dict(**r,passed=r['relative_saving']>=0))
    success=all(r['passed'] for r in checks+size_checks)
    result=dict(endpoint=endpoint,status='QUALITY_SCREEN_PASS' if success else 'QUALITY_SCREEN_NOT_MET',
        all_routes_valid=all(r['valid']=='True' for r in rows),checks=checks,size_checks=size_checks,
        scope='single-seed exposed-development screen; descriptive physical-parent intervals; no independent generalization',
        automatic_followup=False)
    put(path/f'ENDPOINT_{endpoint:04d}_DECISION.json',result)
    return result
