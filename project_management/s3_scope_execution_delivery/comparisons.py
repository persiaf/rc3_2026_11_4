"""Saved-data comparisons; formal cells retain the frozen completeness rule."""
import csv
from collections import defaultdict
import time
import numpy as np
from rc3_s3_evaluator.common import ROOT,SEEDS,METHODS,read,write_new,digest,require
from rc3_s3_evaluator.statistics import gains,bootstrap

OUT=ROOT/'rc3_s3_scope_results_20261109/results'


def dump_csv(path,rows):
    with path.open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def compare():
    start=time.perf_counter()
    with (OUT/'REQUEST_METRICS.csv').open(encoding='utf8',newline='') as f:rows=list(csv.DictReader(f))
    analysis=read(OUT/'SCIENTIFIC_ANALYSIS.json')
    qrows=[r for r in rows if r['kind']=='quality'];lookup={(r['case_id'],int(r['seed']),r['method']):r for r in qrows}
    groups=defaultdict(list)
    for r in rows:groups[r['group'],r['size'],r['method']].append(r)
    summaries=[];versions=[];seed_rows=[];comparisons=[]
    for cell in analysis['quality_cells']:
        group=cell['group'];size='x'.join(map(str,cell['size']));parents=cell['parents'];nparents=len(parents)
        eligible=[r for r in qrows if r['group']==group and r['size']==size]
        case_parent={r['case_id']:r['physical_key'] for r in eligible}
        expected_quality=nparents*2*3
        for m in METHODS:
            allrows=groups[group,size,m];qq=[r for r in allrows if r['kind']=='quality']
            summary=dict(group=group,size=size,method=m,quality_observed=len(qq),quality_expected=expected_quality,
                quality_cell_status=cell['status'])
            for k in ('J','L','K','raw_J','raw_L','raw_K','H_gain_J'):
                summary[k+'_mean']=float(np.mean([float(r[k]) for r in qq])) if qq else None
            for cache in ('cold','hot'):
                rr=[r for r in allrows if r['kind']==cache];t=[float(r['complete_time_s']) for r in rr]
                summary[cache+'_n']=len(t);summary[cache+'_expected']=nparents*2*3*3
                summary[cache+'_median_s']=float(np.median(t)) if t else None;summary[cache+'_p95_s']=float(np.quantile(t,.95)) if t else None
                for version in sorted({(r['scope_revision'],r['persistence_revision']) for r in rr}):
                    vv=[float(r['complete_time_s']) for r in rr if (r['scope_revision'],r['persistence_revision'])==version]
                    versions.append(dict(group=group,size=size,method=m,cache=cache,scope_revision=version[0],persistence_revision=version[1],
                        n=len(vv),median_s=float(np.median(vv)),p95_s=float(np.quantile(vv,.95))))
            summaries.append(summary)
            for seed in SEEDS:
                ss=[r for r in qq if int(r['seed'])==seed]
                seed_rows.append(dict(group=group,size=size,seed=seed,method=m,n=len(ss),expected=nparents*2,
                    J_mean=float(np.mean([float(r['J']) for r in ss])) if ss else None,
                    raw_J_mean=float(np.mean([float(r['raw_J']) for r in ss])) if ss else None))
        for control in ('RL_raw','frozen_supervised','continued_SL','Fixed3_endpoint_adapter','Detour16_endpoint_adapter','untrained'):
            pairs=[]
            for case_id in case_parent:
                for seed in SEEDS:
                    a=lookup.get((case_id,seed,'RL_plus_H'));c=lookup.get((case_id,seed,control))
                    if a is not None and c is not None:pairs.append((c,a))
            row=dict(group=group,size=size,control=control,actor='RL_plus_H',formal_cell_status=cell['status'],
                matched_quality_pairs=len(pairs),expected_quality_pairs=expected_quality,
                observed_pairwise_gain_mean=float(np.mean([(float(c['J'])-float(a['J']))/float(c['J']) for c,a in pairs])) if pairs else None,
                registered_parent_gain_mean=None,lower95=None,upper95=None,seed_gains=None,
                original_quality_gate_pass=None,attribution='H-only postprocessing effect' if control=='RL_raw' else 'Frozen matched-seed pipeline comparison')
            if cell['status']=='COMPLETE':
                require(len(pairs)==expected_quality,'Complete quality cell has missing matched pair')
                cost={m:np.empty((nparents,3)) for m in (control,'RL_plus_H')}
                for pi,p in enumerate(parents):
                    ids=[c for c,par in case_parent.items() if par==p];require(len(ids)==2,'Wrong endpoint pairs')
                    for si,seed in enumerate(SEEDS):
                        for m in cost:cost[m][pi,si]=np.mean([float(lookup[c,seed,m]['J']) for c in ids])
                b=bootstrap(gains(cost[control].mean(1),cost['RL_plus_H'].mean(1)))
                row.update(registered_parent_gain_mean=b['mean'],lower95=b['lower95'],upper95=b['upper95'],
                    seed_gains=[float(gains(cost[control][:,i],cost['RL_plus_H'][:,i]).mean()) for i in range(3)])
                if control in ('frozen_supervised','continued_SL'):
                    original=next(x for x in cell['controls'] if x['control']==control)
                    require(all(abs(row[k]-original[{'registered_parent_gain_mean':'mean'}.get(k,k)])<1e-12 for k in ('registered_parent_gain_mean','lower95','upper95')),'Mismatch original aggregation')
                    row['original_quality_gate_pass']=original['passed']
            comparisons.append(row)
    dump_csv(OUT/'METHOD_QUALITY_TIME.csv',summaries)
    dump_csv(OUT/'TIMING_BY_REVISION.csv',versions)
    dump_csv(OUT/'QUALITY_BY_MODEL_SEED.csv',seed_rows)
    dump_csv(OUT/'MATCHED_COMPARISONS.csv',comparisons)
    write_new(OUT/'MATCHED_COMPARISONS.json',dict(comparisons=comparisons,
        scope='All six declared quality cells; incomplete cell only labelled observed matched-pair description, no formal interval/gate or imputation',
        reference_rule='Parent endpoints averaged, then matched optimizer seeds averaged, relative gains computed per physical parent,20000 PCG64 parent resamples seed11632026',
        no_simultaneous_familywise_CI_claim=True,no_new_scientific_calls=True,read_only_wall_s=time.perf_counter()-start,
        source_helper_sha256=digest(__file__)))
    print(dict(method_rows=len(summaries),comparisons=len(comparisons),formal_complete_cells=sum(c['status']=='COMPLETE' for c in analysis['quality_cells'])))


if __name__=='__main__':compare()
