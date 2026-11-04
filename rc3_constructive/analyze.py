"""Saved-record arithmetic only; no training, model forward or planner calls."""
from __future__ import annotations
import json
import numpy as np
from .problem import ROOT,write,protocol


def bootstrap(values):
    a=np.asarray(values,dtype=float);rng=np.random.default_rng(114)
    samples=a[rng.integers(len(a),size=(10000,len(a)))].mean(1)
    return dict(mean=float(a.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),upper_one_sided95=float(np.quantile(samples,.95)))


def main():
    out=ROOT/'outputs';records=json.loads((out/'final_evaluation.json').read_text())
    assert len(records)==24*2*2*7
    assert len({(r['case'],r['rep'],r['cache_mode'],r['method']) for r in records})==len(records)
    methods=protocol()['evaluation']['methods'];by={}
    for mode in ('cold','hot'):
        by[mode]={}
        for method in methods:
            rows=[r for r in records if r['method']==method and r['cache_mode']==mode]
            assert len(rows)==48 and all(r['valid'] and r['raw_valid'] for r in rows)
            observed_times=np.array([r['total_s'] for r in rows])*1000
            case_groups=[[r for r in rows if r['case']==case] for case in range(24)]
            times=np.array([np.mean([r['total_s'] for r in group]) for group in case_groups])*1000
            components={k:float(np.median([np.mean([r[k]*1000 for r in group]) for group in case_groups])) for k in ('preparation_s','decision_s','postprocess_s','export_validation_s')}
            by[mode][method]=dict(mean_J=float(np.mean([r['J'] for r in rows])),mean_raw_J=float(np.mean([r['raw_J'] for r in rows])),
                median_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),
                measurement_median_ms=float(np.median(observed_times)),measurement_p95_ms=float(np.quantile(observed_times,.95)),
                timing_unit='24 scene means after averaging two repetitions; observed 48-run summaries separately retained',
                mean_L=float(np.mean([r['L'] for r in rows])),mean_K=float(np.mean([r['K'] for r in rows])),
                fallback_rate=float(np.mean([r['fallback'] for r in rows])),raw_valid_rate=1.,final_valid_rate=1.,
                postprocess_mean_gain_percent=float(np.mean([100*(r['raw_J']-r['J'])/r['raw_J'] for r in rows])),
                median_components_ms=components,common_fraction_median=float(np.median([np.mean([r['preparation_s']+r['export_validation_s'] for r in group])/np.mean([r['total_s'] for r in group]) for group in case_groups])))
    paired={}
    for comparator in ('fixed3','detour8','detour16','insertion','prior','initial'):
        values=[]
        for case in range(24):
            ours=np.mean([r['J'] for r in records if r['case']==case and r['method']=='model'])
            base=np.mean([r['J'] for r in records if r['case']==case and r['method']==comparator])
            values.append(100*(ours-base)/base)
        paired[comparator]=dict(bootstrap(values),scene_degradation_percent=values)
    primary={}
    for mode in ('cold','hot'):
        m=by[mode]['model'];d=by[mode]['detour16']
        median_ratio=m['median_ms']/d['median_ms'];p95_ratio=m['p95_ms']/d['p95_ms']
        primary[mode]=dict(quality_upper95=paired['detour16']['upper_one_sided95'],quality_pass=paired['detour16']['upper_one_sided95']<=.5,
            median_time_ratio=median_ratio,median_time_pass=median_ratio<=.7,p95_ratio=p95_ratio,p95_pass=p95_ratio<=1.)
        primary[mode]['passed']=all(primary[mode][k] for k in ('quality_pass','median_time_pass','p95_pass'))
        primary[mode]['summary_frontier_dominators']=[name for name,row in by[mode].items()
            if name!='model' and row['mean_J']<=m['mean_J'] and row['median_ms']<=m['median_ms']
            and (row['mean_J']<m['mean_J'] or row['median_ms']<m['median_ms'])]
    summary=json.loads((out/'training_summary.json').read_text())
    initial=json.loads((out/'selection/update0000.json').read_text());chosen=json.loads((out/'selection'/f"update{summary['selected_update']:04}.json").read_text())
    gains=[100*(a['post_J']-b['post_J'])/a['post_J'] for a,b in zip(initial,chosen)]
    a=np.array([r['post_J'] for r in initial]);b=np.array([r['post_J'] for r in chosen])
    rng=np.random.default_rng(114);idx=rng.integers(len(a),size=(10000,len(a)))
    ratios=100*(a[idx].mean(1)-b[idx].mean(1))/a[idx].mean(1)
    signal_stats=dict(mean=float(100*(a.mean()-b.mean())/a.mean()),ci95=np.quantile(ratios,[.025,.975]).tolist(),
                      estimator='ratio of mean postprocessed costs, as specified by selection_mean_post_J_reduction',
                      paired_mean_relative_percent=float(np.mean(gains)))
    continuation=dict(selection_gain=signal_stats,quality_signal=signal_stats['mean']>=1. and signal_stats['ci95'][0]>0,
                      speed_signal=primary['cold']['median_time_ratio']<=.9,all_valid=True)
    continuation['passed']=continuation['quality_signal'] and continuation['speed_signal']
    small=json.loads((out/'small_exact.json').read_text());small_summary={}
    for method in ('fixed3','detour8','detour16','insertion'):
        gap=[100*(r['baselines'][method]['J']/r['full_exact']['objective']-1) for r in small]
        small_summary[method]=dict(mean_gap_percent=float(np.mean(gap)),max_gap_percent=float(np.max(gap)))
    write(out/'analysis.json',dict(schema='rc3-constructive-c2-analysis-v1',summary=by,paired_degradation=paired,
        primary=primary,continuation_signal=continuation,small_summary=small_summary,
        small_full_optimal=sum(r['full_exact']['optimal'] for r in small),
        small_restricted_optimal=sum(r['baseline_candidate_restriction']['result']['optimal'] for r in small),
        strict_outside_baseline_set_cases=sum(r['strict_outside_set_value'] for r in small),
        training=summary,counts=dict(c0_development_measurements=480,c2_development_measurements=len(records),
                                    independent_development_scenes=24,selection_scenes=48,final_test_scenes_queried=0),
        scope='Development/selection screening with one training seed; bootstrap on scenes conditional on selected model. No final-test, stability or cross-scale claim.'))
    print(json.dumps(dict(primary=primary,continuation=continuation,selected_update=summary['selected_update'])),flush=True)


if __name__=='__main__':main()
