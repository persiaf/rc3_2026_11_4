"""Preregistered physical-parent aggregation and six/twelve AND decisions."""
import numpy as np
from .common import SEEDS, SIZES, METHODS, require, finite_number
from .journal import grid
from .pipelines import verify_determinism


def bootstrap(values, indices=None):
    v = np.asarray(values, dtype=float)
    require(v.ndim == 1 and len(v) > 0 and np.isfinite(v).all(), 'Invalid parent bootstrap sample')
    if indices is None:
        indices = np.random.Generator(np.random.PCG64(11632026)).integers(len(v), size=(20000,len(v)))
    require(np.asarray(indices).ndim == 2 and np.asarray(indices).shape[1] == len(v) and
            np.issubdtype(np.asarray(indices).dtype, np.integer) and np.min(indices) >= 0 and np.max(indices) < len(v), 'Invalid bootstrap indices')
    means = v[indices].mean(1)
    return dict(mean=float(v.mean()), lower95=float(np.quantile(means,.025)),
                upper95=float(np.quantile(means,.975)), upper_one_sided95=float(np.quantile(means,.95)))


def gains(control, actor):
    c, a = np.asarray(control), np.asarray(actor)
    require(c.shape == a.shape and np.isfinite(c).all() and np.isfinite(a).all() and (c > 0).all(), 'Invalid matched costs')
    return (c-a)/c


def quality_pass(b, seed_gains):
    return bool(b['mean'] >= .005 and b['lower95'] > 0 and all(v > 0 for v in seed_gains))


def dominated(fixed, rl):
    return bool(all(a <= b for a,b in zip(fixed,rl)) and any(a < b for a,b in zip(fixed,rl)))


def analyze(rows, cases):
    require(len(cases) == 108 and len({c['id'] for c in cases}) == 108, 'Formal analysis requires exact case grid')
    expected = {r['id']:r for s in SEEDS for r in grid(cases,s)}
    index = {}
    bycase = {c['id']:c for c in cases}
    for row in rows:
        q = row['request']; rid=q['id']
        require(expected.get(rid) == q and rid not in index and row['valid'] is True, 'Missing/duplicate/invalid request')
        case = bycase[q['case_id']]
        require(all(row[k] == case[k] for k in ('physical_key','group','size','endpoint')) and row['case_id'] == case['id'], 'Result/physical parent declaration mismatch')
        for key in ('J','L','K','raw_J','raw_L','raw_K'): finite_number(row['metrics'][key])
        finite_number(row['complete_time_s'],positive=True)
        index[rid] = row
    require(set(index) == set(expected), 'Incomplete full grid; no dropped/replaced cases')
    deterministic = verify_determinism(rows)
    lookup = {(r['case_id'],r['request']['seed'],r['request']['method'],r['request']['kind'],r['request']['repeat']):r for r in rows}
    quality, deployment, endpoints = [], [], []
    for group,nparents in (('seen',6),('unseen',12)):
        for size in SIZES:
            subset=[c for c in cases if c['group']==group and tuple(c['size'])==size]
            parents=sorted({c['physical_key'] for c in subset})
            require(len(parents)==nparents, 'Physical cluster count changed')
            costs={}
            for m in METHODS:
                a=np.empty((nparents,3),dtype=float)
                for pi,p in enumerate(parents):
                    cc=[c for c in subset if c['physical_key']==p]
                    require(len(cc)==2 and {c['endpoint'] for c in cc}=={'closed_depot','open_first_last_connected'}, 'Parent endpoint grid invalid')
                    for si,s in enumerate(SEEDS):
                        a[pi,si]=np.mean([lookup[c['id'],s,m,'quality',0]['metrics']['J'] for c in cc])
                costs[m]=a
            pooled={m:a.mean(1) for m,a in costs.items()}
            indices=np.random.Generator(np.random.PCG64(11632026)).integers(nparents,size=(20000,nparents))
            controls=[]
            for m in ('frozen_supervised','continued_SL'):
                b=bootstrap(gains(pooled[m],pooled['RL_plus_H']),indices)
                seed_g=[float(gains(costs[m][:,i],costs['RL_plus_H'][:,i]).mean()) for i in range(3)]
                controls.append(dict(control=m,**b,seed_gains=seed_g,passed=quality_pass(b,seed_g)))
            quality.append(dict(group=group,size=list(size),parents=parents,controls=controls,passed=all(c['passed'] for c in controls)))
            deg=bootstrap(-gains(pooled['Detour16_endpoint_adapter'],pooled['RL_plus_H']),indices)
            for cache in ('cold','hot'):
                times={m:np.asarray([lookup[c['id'],s,m,cache,rep]['complete_time_s'] for c in subset for s in SEEDS for rep in range(3)]) for m in METHODS}
                timing={m:dict(median_s=float(np.median(t)),p95_s=float(np.quantile(t,.95)),n=len(t)) for m,t in times.items()}
                med=timing['RL_plus_H']['median_s']/timing['Detour16_endpoint_adapter']['median_s']
                p95=timing['RL_plus_H']['p95_s']/timing['Detour16_endpoint_adapter']['p95_s']
                vec=lambda m:(float(pooled[m].mean()),timing[m]['median_s'],timing[m]['p95_s'])
                dom=dominated(vec('Fixed3_endpoint_adapter'),vec('RL_plus_H'))
                passed=deg['upper_one_sided95']<=.005 and med<=.7 and p95<=1 and not dom
                deployment.append(dict(group=group,size=list(size),cache=cache,degradation=deg,median_ratio=med,p95_ratio=p95,Fixed3_dominates=dom,timing=timing,passed=bool(passed)))
            for ep in ('closed_depot','open_first_last_connected'):
                for m in METHODS:
                    rr=[r for r in rows if r['group']==group and tuple(r['size'])==size and r['endpoint']==ep and r['request']['method']==m and r['request']['kind']=='quality']
                    endpoints.append(dict(group=group,size=list(size),endpoint=ep,method=m,n=len(rr),**{k:float(np.mean([r['metrics'][k] for r in rr])) for k in ('J','L','K','raw_J','raw_L','raw_K')}))
    return dict(scope='FORMAL_S3_ONLY',quality_cells=quality,deployment_cells=deployment,endpoint_tables=endpoints,
        independent_RL_pass=all(c['passed'] for c in quality),deployment_pass=all(c['passed'] for c in deployment),
        combined_pass=all(c['passed'] for c in quality+deployment),determinism=deterministic,
        uncertainty='Physical-parent percentile intervals; not simultaneous familywise coverage; optimizer seeds share one data bank.')
