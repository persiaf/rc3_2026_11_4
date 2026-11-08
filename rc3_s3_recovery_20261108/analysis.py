"""Separate quality/cold/hot completeness; never pad a failed timing slot."""
from collections import Counter
import time
import numpy as np
from rc3_s3_evaluator.common import (SEEDS,SIZES,METHODS,require,finite_number,read,write_new,identity,digest)
from rc3_s3_evaluator.journal import grid
from rc3_s3_evaluator.statistics import bootstrap,gains,quality_pass,dominated
from .authority import HOME,RUN,CONT,FAILED_ID,admit


def analyze(rows,cases):
    require(len(cases)==108 and len({c['id'] for c in cases})==108,'Wrong declared grid')
    expected={q['id']:q for s in SEEDS for q in grid(cases,s)};bycase={c['id']:c for c in cases}
    lookup={};seen=set();stable={};raw={};versions=Counter()
    for r in rows:
        q=r['request'];rid=q['id'];c=bycase[q['case_id']]
        require(expected.get(rid)==q and rid not in seen and r['valid'],'Wrong/duplicate result')
        require(all(r[k]==c[k] for k in ('physical_key','group','size','endpoint')),'Wrong physical parent/stratum')
        for k in ('J','L','K','raw_J','raw_L','raw_K'):finite_number(r['metrics'][k])
        finite_number(r['complete_time_s'],positive=True)
        seen.add(rid);lookup[c['id'],q['seed'],q['method'],q['kind'],q['repeat']]=r
        key=(c['id'],q['seed'],q['method']);v=(r['raw_route_sha256'],r['route_sha256'],r['metrics'])
        require(key not in stable or stable[key]==v,'Frozen route/cost changed between revisions/repeats');stable[key]=v
        if q['method'] in ('RL_raw','RL_plus_H'):
            key=(c['id'],q['seed']);require(key not in raw or raw[key]==r['raw_route_sha256'],'RL raw pair differs');raw[key]=r['raw_route_sha256']
        versions[r.get('persistence_revision','legacy_b348')]+=1
    missing=sorted(set(expected)-seen)
    coverage={kind:dict(expected=sum(q['kind']==kind for q in expected.values()),
        observed=sum(r['request']['kind']==kind for r in rows),missing=[rid for rid in missing if expected[rid]['kind']==kind])
        for kind in ('quality','cold','hot')}
    quality=[];deploy=[];endpoints=[]
    def timing(rr):
        values=[r['complete_time_s'] for r in rr]
        return dict(n=len(values),median_s=float(np.median(values)) if values else None,
                    p95_s=float(np.quantile(values,.95)) if values else None)
    for group,nparents in (('seen',6),('unseen',12)):
        for size in SIZES:
            subset=[c for c in cases if c['group']==group and tuple(c['size'])==size];parents=sorted({c['physical_key'] for c in subset})
            require(len(parents)==nparents,'Wrong physical cluster count')
            ids={c['id'] for c in subset}
            qmissing=[rid for rid in missing if expected[rid]['case_id'] in ids and expected[rid]['kind']=='quality']
            qcell=dict(group=group,size=list(size),parents=parents,status='INCOMPLETE' if qmissing else 'COMPLETE',missing_ids=qmissing,passed=None,controls=[])
            pooled=None;indices=np.random.Generator(np.random.PCG64(11632026)).integers(nparents,size=(20000,nparents))
            if not qmissing:
                costs={}
                for m in METHODS:
                    a=np.empty((nparents,3))
                    for pi,p in enumerate(parents):
                        cc=[c for c in subset if c['physical_key']==p]
                        require(len(cc)==2 and {c['endpoint'] for c in cc}=={'closed_depot','open_first_last_connected'},'Wrong endpoint grid')
                        for si,s in enumerate(SEEDS):a[pi,si]=np.mean([lookup[c['id'],s,m,'quality',0]['metrics']['J'] for c in cc])
                    costs[m]=a
                pooled={m:a.mean(1) for m,a in costs.items()}
                for m in ('frozen_supervised','continued_SL'):
                    b=bootstrap(gains(pooled[m],pooled['RL_plus_H']),indices)
                    sg=[float(gains(costs[m][:,i],costs['RL_plus_H'][:,i]).mean()) for i in range(3)]
                    qcell['controls'].append(dict(control=m,**b,seed_gains=sg,passed=quality_pass(b,sg)))
                qcell['passed']=all(v['passed'] for v in qcell['controls'])
                for ep in ('closed_depot','open_first_last_connected'):
                    for m in METHODS:
                        rr=[lookup[c['id'],s,m,'quality',0] for c in subset if c['endpoint']==ep for s in SEEDS]
                        endpoints.append(dict(group=group,size=list(size),endpoint=ep,method=m,n=len(rr),
                            **{k:float(np.mean([r['metrics'][k] for r in rr])) for k in ('J','L','K','raw_J','raw_L','raw_K')}))
            quality.append(qcell)
            for cache in ('cold','hot'):
                tmissing=[rid for rid in missing if expected[rid]['case_id'] in ids and expected[rid]['kind']==cache]
                times={m:[r for r in rows if r['case_id'] in ids and r['request']['method']==m and r['request']['kind']==cache] for m in METHODS}
                summaries={m:timing(rr) for m,rr in times.items()}
                version_stats={m:{v:timing([r for r in rr if r.get('persistence_revision','legacy_b348')==v])
                    for v in sorted({r.get('persistence_revision','legacy_b348') for r in rr})} for m,rr in times.items()}
                cell=dict(group=group,size=list(size),cache=cache,status='INCOMPLETE' if qmissing or tmissing else 'COMPLETE',
                    missing_ids=tmissing,quality_missing_ids=qmissing,passed=None,timing=summaries,timing_by_persistence_revision=version_stats)
                if not tmissing and not qmissing:
                    deg=bootstrap(-gains(pooled['Detour16_endpoint_adapter'],pooled['RL_plus_H']),indices)
                    med=summaries['RL_plus_H']['median_s']/summaries['Detour16_endpoint_adapter']['median_s']
                    p95=summaries['RL_plus_H']['p95_s']/summaries['Detour16_endpoint_adapter']['p95_s']
                    vec=lambda m:(float(pooled[m].mean()),summaries[m]['median_s'],summaries[m]['p95_s'])
                    dom=dominated(vec('Fixed3_endpoint_adapter'),vec('RL_plus_H'))
                    cell.update(degradation=deg,median_ratio=med,p95_ratio=p95,Fixed3_dominates=dom,
                        passed=bool(deg['upper_one_sided95']<=.005 and med<=.7 and p95<=1 and not dom))
                deploy.append(cell)
    qfull=all(c['status']=='COMPLETE' for c in quality);tfull=all(c['status']=='COMPLETE' for c in deploy)
    qpass=all(c['passed'] for c in quality) if qfull else None
    dpass=all(c['passed'] for c in deploy) if tfull else None
    return dict(scope='ORIGINAL_FIXED_S3_GRID_WITH_EXPLICIT_MISSINGNESS',expected=len(expected),observed=len(rows),missing_ids=missing,
        missing_requests=[expected[rid] for rid in missing],coverage=coverage,quality_cells=quality,deployment_cells=deploy,
        endpoint_tables=endpoints,independent_RL_pass=qpass,deployment_pass=dpass,combined_pass=(qpass and dpass) if qfull and tfull else None,
        quality_status='EVALUABLE' if qfull else 'INCOMPLETE',deployment_status='EVALUABLE' if tfull else 'INCOMPLETE_TIMING_GRID',
        timing_revision_counts=dict(versions),timing_comparability='Same CPU/thread policy, model files, method order and complete-request boundaries. Repaired anchor checks/retries are measured IO and may add overhead; no exact equality of persistence overhead is claimed. Version-stratified summaries retained. Observed pooled times descriptive; missing cell never filled or globally passed.',
        uncertainty='20000 PCG64 physical-parent resamples, seed11632026; original matched aggregation and AND thresholds; no familywise simultaneous guarantee.')


def saved_rows(legacy):
    """Stream plans off disk into small metric records, without new evaluations."""
    keep=('request','case_id','group','size','endpoint','physical_key','valid','metrics','raw_route_sha256','route_sha256','persistence_revision')
    rows=[]
    for e in legacy['results'].values():
        from rc3_s3_evaluator.common import project_file
        r=read(project_file(e['artifact_path']));require(digest(project_file(e['artifact_path']))==e['artifact_sha256'],'Legacy result changed')
        thin={k:r[k] for k in keep if k in r};thin['complete_time_s']=e['complete_time_s'];rows.append(thin)
    for seed in SEEDS:
        folder=CONT/str(seed)
        from .storage import read_chain
        events,_=read_chain(folder/'journal.jsonl')
        for e in events:
            if e['event']!='request_complete':continue
            p=folder/'results'/(e['id']+'.json');require(digest(p)==e['artifact_sha256'],'New result changed')
            r=read(p);thin={k:r[k] for k in keep if k in r};thin['complete_time_s']=e['complete_time_s'];rows.append(thin)
    return rows


def analyze_saved():
    tick=time.perf_counter();permit,legacy,binding=admit()
    from .runner import Inputs
    rows=saved_rows(legacy);input_identity=identity(rows)
    path=CONT/'SCIENTIFIC_ANALYSIS.json'
    if path.exists():
        existing=read(path)
        require(existing['binding']==binding and existing['input_result_identity']==input_identity and
            existing['missing_ids']==[FAILED_ID],'Existing analysis/source/results changed; no implicit overwrite')
        return existing
    result=analyze(rows,Inputs(permit).cases)
    require(result['missing_ids']==[FAILED_ID],'Unexpected missing slots after scheduling')
    write_new(path,dict(result,analysis_wall_s=time.perf_counter()-tick,binding=binding,input_result_identity=input_identity,
        actual_new_analysis_generation_inference_native_H_optimizer=0))
