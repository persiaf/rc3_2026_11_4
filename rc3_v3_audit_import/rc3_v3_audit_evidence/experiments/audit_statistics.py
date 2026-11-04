"""Independent frozen-v3 statistics. No model training or source mutation."""
from __future__ import annotations
import csv
import argparse
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-root',type=Path,default=Path.cwd())
parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent)
args=parser.parse_args()
OUT=args.output_dir.resolve();OUT.mkdir(parents=True,exist_ok=True)
ROOT=args.source_root.resolve()
BASE=ROOT/'rc3_v3'
SEED=20261103
BOOTSTRAPS=20000


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def key(r):return r['seed'],r['nx'],r['ny'],r['mode']


def write_csv(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def paired_ci(delta):
    # Reset for each comparison: identical duplicated groups get identical CIs.
    rng=np.random.default_rng(SEED)
    delta=np.asarray(delta,dtype=float)
    draws=delta[rng.integers(len(delta),size=(BOOTSTRAPS,len(delta)))].mean(1)
    return np.quantile(draws,[.025,.975]).tolist()


def mapped_path(path):
    parts=path.replace('\\','/').split('/')
    index=next(i for i,p in enumerate(parts) if p in ('rc3_v3','rc3_improve','rc3_fast','rc3_baseline'))
    return ROOT.joinpath(*parts[index:])


def main():
    rows=read(BASE/'results/metrics.json')
    groups={'all_descriptive':rows,
            'primary_up_to_196':[r for r in rows if r['tables']<=196],
            'in_distribution_100':[r for r in rows if r['tables']==100],
            'smaller_transfer':[r for r in rows if r['tables']<100],
            'larger_transfer':[r for r in rows if 100<r['tables']<=196],
            'stress_over_196':[r for r in rows if r['tables']>196]}
    groups.update({f'tables_{n}':[r for r in rows if r['tables']==n] for n in sorted({r['tables'] for r in rows})})
    stats=[];comparisons=[]
    for group,subset in groups.items():
        configs=list(dict.fromkeys(r['config'] for r in subset))
        lookup={(key(r),r['config']):r for r in subset if r['status']=='complete'}
        for config in configs:
            selected=[r for r in subset if r['config']==config];good=[r for r in selected if r['status']=='complete']
            first=selected[0]
            stat=dict(group=group,config=config,method=first['method'],kind=first['kind'],budget=first['budget'],
                      scenes=len(selected),success=len(good),failures=len(selected)-len(good),
                      mean_objective_m=float(np.mean([r['objective_m'] for r in good])),
                      mean_improvement_pct=float(np.mean([r['improvement_pct'] for r in good])),
                      mean_dp_calls=float(np.mean([r['dp_calls'] for r in good])),
                      mean_accepted=float(np.mean([r['accepted'] for r in good])),
                      mean_overrun_rate=float(np.mean([r['overrun_rate'] for r in good])),
                      mean_overrun_ms=float(np.mean([r['mean_overrun_s'] for r in good]))*1000,
                      max_overrun_ms=float(max(r['max_overrun_s'] for r in good))*1000,
                      mean_fallback_rate=float(np.mean([r['fallback_rate'] for r in good])))
            for field in ('planning_s','total_s','preprocessing_s'):
                vals=np.array([r[field] for r in good])*1000
                for quantile,fun in [('median',np.median),('mean',np.mean),('p95',lambda a:np.quantile(a,.95)),('max',np.max)]:
                    stat[f'{field[:-2]}_{quantile}_ms']=float(fun(vals))
            stats.append(stat)
        for config in configs:
            if not config.startswith(('learned_','detour_')):continue
            references=['fixed3',config.replace(config.split('_')[0]+'_','random_',1)]
            if config.startswith('learned_'):references.append(config.replace('learned_','detour_',1))
            for reference in references:
                pairs=[(r,lookup[key(r),reference]) for r in subset
                       if r['config']==config and r['status']=='complete' and (key(r),reference) in lookup]
                if not pairs:continue
                delta=np.array([(b['objective_m']-a['objective_m'])/b['objective_m']*100 for a,b in pairs])
                ci=paired_ci(delta)
                comparisons.append(dict(group=group,config=config,reference=reference,n=len(pairs),
                                        mean_reduction_pct=float(delta.mean()),ci_low=ci[0],ci_high=ci[1],
                                        wins=int((delta>1e-7).sum()),ties=int((np.abs(delta)<=1e-7).sum()),losses=int((delta< -1e-7).sum()),
                                        median_planning_speedup=float(np.median([b['planning_s']/a['planning_s'] for a,b in pairs]))))
    frozen=read(BASE/'results/summary.json')
    disagreements=[]
    for s in frozen['summaries']:
        group='all_descriptive' if s['group']=='all' else s['group']
        local=next(r for r in stats if r['group']==group and r['config']==s['config'])
        mapping={'total':'scenes','success':'success','mean_objective_m':'mean_objective_m','mean_improvement_pct':'mean_improvement_pct',
                 'median_planning_ms':'planning_median_ms','median_total_ms':'total_median_ms','mean_dp_calls':'mean_dp_calls',
                 'mean_overrun_rate':'mean_overrun_rate','mean_fallback_rate':'mean_fallback_rate'}
        for field,target in mapping.items():
            if abs(s[field]-local[target])>1e-9:disagreements.append(dict(group=group,config=s['config'],field=field,frozen=s[field],computed=local[target]))
    for c in frozen['comparisons']:
        group='all_descriptive' if c['group']=='all' else c['group']
        local=next(r for r in comparisons if r['group']==group and r['config']==c['config'] and r['reference']==c['reference'])
        for field in ('mean_reduction_pct','wins','ties','losses','median_planning_speedup'):
            if abs(c[field]-local[field])>1e-9:disagreements.append(dict(group=group,config=c['config'],field=field,frozen=c[field],computed=local[field]))
    # Count/reconstruct repeats from serialized runs; this is not geometric validation.
    run_rows=[];rebuild=[];plan_count=0;run_count=0
    for index,path in enumerate(sorted((BASE/'results/plans').glob('*.json'))):
        payload=read(path);scene=payload['scene'];plan_count+=len(payload['plans']);run_count+=len(payload['runs'])
        bucket=defaultdict(list)
        for label,run in payload['runs'].items():
            config,repeat=label.rsplit('_rep',1);bucket[config].append(run)
            runtime=run.get('runtime',{})
            entry=dict(seed=scene['seed'],tables=len(scene['tables']),mode=scene['complexity'],config=config,repeat=int(repeat),
                       status=run['status'],objective_m=run.get('objective_m'),planning_s=run['planning_s'],
                       dp_calls=run.get('dp_calls'),accepted=run.get('accepted'),stop_reason=run.get('stop_reason'),
                       deadline_exceeded=run.get('deadline_exceeded'),overrun_s=run.get('overrun_s'),
                       saved_plan_present=label in payload['plans'])
            for name in ('reference_s','feature_s','model_s','repair_s','final_validation_s'):
                entry[name]=runtime.get(name)
            run_rows.append(entry)
        for config,samples in bucket.items():
            metric=next(r for r in rows if r['seed']==scene['seed'] and r['config']==config)
            cost=float(np.mean([r['objective_m'] for r in samples]))
            planning=float(np.median([r['planning_s'] for r in samples]))
            rebuild.append(dict(seed=scene['seed'],config=config,repeats=len(samples),
                                objective_matches=abs(cost-metric['objective_m'])<1e-8,
                                planning_matches=abs(planning-metric['planning_s'])<1e-9,
                                overrun_matches=abs(np.mean([r['deadline_exceeded'] for r in samples])-metric['overrun_rate'])<1e-9))
        if (index+1)%15==0:print(f'read {index+1}/60 repeat records',flush=True)
    wall=[]
    for config in sorted({r['config'] for r in run_rows if '_wall_' in r['config']}):
        selected=[r for r in run_rows if r['config']==config]
        budget=int(config.split('_')[-1][:-2])/1000
        durations=np.array([r['planning_s'] for r in selected])
        overruns=np.maximum(durations-budget,0)
        wall.append(dict(config=config,independent_scenes=len({r['seed'] for r in selected}),repeated_runs=len(selected),
                         deadline_ms=budget*1000,overrun_runs=int((overruns>0).sum()),overrun_rate=float((overruns>0).mean()),
                         repeated_run_median_planning_ms=float(np.median(durations))*1000,
                         repeated_run_p95_ms=float(np.quantile(durations,.95))*1000,max_planning_ms=float(durations.max())*1000,
                         mean_overrun_ms=float(overruns.mean())*1000,max_overrun_ms=float(overruns.max())*1000,
                         stop_reasons=dict(Counter(r['stop_reason'] for r in selected))))
    component_stats=[]
    for group,scales in [('primary_up_to_196',set(range(197))),('in_distribution_100',{100}),('stress_over_196',{256,400})]:
        for config in ['fixed3','random_calls_8','detour_calls_8','learned_calls_8','learned_wall_200ms','learned_wall_400ms','learned_wall_800ms']:
            selected=[r for r in run_rows if r['tables'] in scales and r['config']==config]
            if not selected:continue
            seeds=sorted({r['seed'] for r in selected})
            summary=dict(group=group,config=config,n=len(seeds))
            for field in ('planning_s','reference_s','feature_s','model_s','repair_s','final_validation_s'):
                scene_medians=[np.median([r[field] for r in selected if r['seed']==s]) for s in seeds]
                summary[field[:-2]+'_median_ms']=float(np.median(scene_medians))*1000
                summary[field[:-2]+'_mean_ms']=float(np.mean(scene_medians))*1000
            component_stats.append(summary)
    # Source/data hashes: no unpickling required.
    manifest=read(BASE/'results/manifest.json');checks=[]
    for section in ('source_sha256','data_sha256'):
        for source,expected in manifest[section].items():
            path=mapped_path(source);actual=sha(path) if path.exists() else None
            checks.append(dict(section=section,path=str(path),expected=expected,actual=actual,matches=expected==actual))
    for pathkey,hashkey in [('model_path','model_sha256'),('wall_config_path','wall_config_sha256')]:
        path=mapped_path(manifest[pathkey]);actual=sha(path)
        checks.append(dict(section=hashkey,path=str(path),expected=manifest[hashkey],actual=actual,matches=actual==manifest[hashkey]))
    cases={split:read(BASE/f'data_v3_20261102/{split}_cases.json') for split in ('train','validation','test')}
    summary=dict(rows=len(rows),unique_scenes=len({key(r) for r in rows}),
                 scale_scene_counts=dict(Counter(r['tables'] for r in rows if r['config']=='fixed3')),
                 primary_scenes=sum(r['config']=='fixed3' and r['tables']<=196 for r in rows),
                 stress_scenes=sum(r['config']=='fixed3' and r['tables']>196 for r in rows),
                 row_counts_by_kind=dict(Counter(r['kind'] for r in rows)),
                 expected_row_formula='60 * (1 anchor + 3 methods * 4 call budgets) + 9 * 3 methods * 3 wall budgets = 861',
                 repeated_runs=run_count,saved_plans=plan_count,
                 success=sum(r['status']=='complete' for r in rows),
                 all_861_repeat_aggregates_match=all(r['objective_matches'] and r['planning_matches'] and r['overrun_matches'] and r['repeats']==3 for r in rebuild),
                 all_repeat_plans_present=all(r['saved_plan_present'] for r in run_rows),
                 frozen_summary_disagreements=disagreements,summary_rows_checked=len(frozen['summaries']),comparison_rows_checked=len(frozen['comparisons']),
                 hash_total=len(checks),hash_matches=sum(r['matches'] for r in checks),
                 hash_mismatches=[r for r in checks if not r['matches']],
                 split_counts={s:len(v) for s,v in cases.items()},
                 split_seed_overlaps={f'{a}_{b}':sorted({r['seed'] for r in cases[a]}&{r['seed'] for r in cases[b]})
                                      for a,b in [('train','validation'),('train','test'),('validation','test')]},
                 bootstrap=dict(seed=SEED,resamples=BOOTSTRAPS,unit='scene; repeated wall objectives averaged first',
                                resetting='Same RNG seed reset per comparison; identical data aliases produce identical intervals.',
                                caveat='No training-seed uncertainty or multiple-comparison correction. Small n stress intervals are descriptive only.'),
                 validity_scope='This file reconstructs recorded aggregates; a different audit independently revalidates plan geometry.')
    for name,records in [('group_statistics.csv',stats),('paired_comparisons.csv',comparisons),('wall_budget_runs.csv',wall),
                         ('repeat_run_metrics.csv',run_rows),('repeat_aggregate_checks.csv',rebuild),('component_statistics.csv',component_stats)]:
        write_csv(name,records)
    (OUT/'hash_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('Main eight-call results:')
    for s in stats:
        if s['group'] in ('primary_up_to_196','in_distribution_100','tables_400') and s['config'] in ('fixed3','random_calls_8','detour_calls_8','learned_calls_8'):print(s)
    print('Key comparisons:')
    for c in comparisons:
        if c['group'] in ('primary_up_to_196','in_distribution_100','larger_transfer','stress_over_196') and c['config'] in ('learned_calls_8','learned_calls_1','learned_calls_2','learned_calls_4','learned_wall_400ms','learned_wall_800ms') and ('detour' in c['reference'] or c['config']=='learned_calls_8'):print(c)


if __name__=='__main__':main()
