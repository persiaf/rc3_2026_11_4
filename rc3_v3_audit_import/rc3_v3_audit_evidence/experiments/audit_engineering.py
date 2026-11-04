"""Rebuild engineering timings and focused 100/400-table components from logs."""
import csv
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-root',type=Path,default=Path.cwd())
parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent)
args=parser.parse_args()
OUT=args.output_dir.resolve();OUT.mkdir(parents=True,exist_ok=True)
ROOT=args.source_root.resolve()
BASE=ROOT/'rc3_v3'


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write_csv(name,rows):
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)


def main():
    profile=read(BASE/'engineering_profile.json')
    buckets=defaultdict(list)
    for run in profile['runs']:buckets[run['scene_id']].append(run)
    pairs=[]
    for index,runs in buckets.items():
        assert len(runs)==6 and all(r['status']=='complete' for r in runs)
        costs=[r['objective_m'] for r in runs]
        old=np.median([r['planning_s'] for r in runs if r['method']=='old_fixed3'])
        new=np.median([r['planning_s'] for r in runs if r['method']=='new_fixed3'])
        pre=runs[0]['shared_preprocessing_s']
        pairs.append(dict(scene_id=index,seed=runs[0]['seed'],mode=runs[0]['mode'],
                          old_planning_ms=float(old*1000),new_planning_ms=float(new*1000),
                          shared_preprocessing_ms=pre*1000,old_cold_total_ms=float((old+pre)*1000),
                          new_cold_total_ms=float((new+pre)*1000),speedup=float(old/new),
                          reduction_pct=float((old-new)/old*100),costs_equal=max(costs)-min(costs)<1e-8,
                          logged_independent_checks_passed=all(r['independently_rechecked'] and r['valid_model_plan'] and r['all_tasks_served'] for r in runs),
                          new_faster=bool(new<old)))
    old=np.array([r['old_planning_ms'] for r in pairs]);new=np.array([r['new_planning_ms'] for r in pairs])
    summary=dict(independent_validation_scenes=len(pairs),raw_runs=len(profile['runs']),
                 old_median_planning_ms=float(np.median(old)),new_median_planning_ms=float(np.median(new)),
                 median_per_scene_speedup=float(np.median(old/new)),ratio_of_medians=float(np.median(old)/np.median(new)),
                 median_per_scene_reduction_pct=float(np.median((old-new)/old*100)),
                 old_median_cold_total_ms=float(np.median([r['old_cold_total_ms'] for r in pairs])),
                 new_median_cold_total_ms=float(np.median([r['new_cold_total_ms'] for r in pairs])),
                 new_faster_scenes=sum(r['new_faster'] for r in pairs),all_costs_equal=all(r['costs_equal'] for r in pairs),
                 logged_all_independent_checks_passed=all(r['logged_independent_checks_passed'] for r in pairs),
                 hardware=profile['hardware'],scope=profile['timing_scope'],
                 audit_boundary='Original raw records reconstructed; no re-timing or re-execution of these 108 runs. The original code independently checks each returned plan outside the timed call; required internal validation is included.',
                 contribution='Engineering change in Fixed3 certification schedule; no model loaded and no test scenes used.')
    checks=[]
    for source,expected in profile['source_sha256'].items():
        parts=source.replace('\\','/').split('/');idx=next(i for i,p in enumerate(parts) if p.startswith('rc3_'))
        local=ROOT.joinpath(*parts[idx:]);actual=hashlib.sha256(local.read_bytes()).hexdigest()
        checks.append(dict(source=source,expected=expected,actual=actual,matches=expected==actual))
    summary['source_hashes_match']=all(r['matches'] for r in checks)
    summary['source_hash_checks']=checks
    metrics=read(BASE/'results/metrics.json');metric_lookup={(r['seed'],r['config']):r for r in metrics}
    allruns=list(csv.DictReader((OUT/'repeat_run_metrics.csv').open()))
    focused=[]
    for r in allruns:
        tables=int(r['tables'])
        if tables not in (100,400) or r['config'] not in ('fixed3','detour_calls_8','learned_calls_8'):continue
        metric=metric_lookup[int(r['seed']),r['config']]
        row=dict(seed=int(r['seed']),tables=tables,mode=r['mode'],config=r['config'],repeat=int(r['repeat']),
                 objective_m=float(r['objective_m']),reference_objective_m=metric['reference_objective_m'],
                 preprocessing_s=metric['preprocessing_s'],planning_s=float(r['planning_s']),
                 cold_total_s=metric['preprocessing_s']+float(r['planning_s']))
        for field in ('reference_s','feature_s','model_s','repair_s','final_validation_s'):
            row[field]=float(r[field])
        row['other_overhead_s']=row['planning_s']-sum(row[k] for k in ('reference_s','feature_s','model_s','repair_s','final_validation_s'))
        row['dp_calls']=float(r['dp_calls']);row['accepted']=float(r['accepted']);row['stop_reason']=r['stop_reason']
        focused.append(row)
    bycase=defaultdict(list)
    for row in focused:bycase[row['seed'],row['config']].append(row)
    focused_summaries=[]
    for (_,config),runs in bycase.items():
        first=runs[0]
        result={k:first[k] for k in ('seed','tables','mode','config','objective_m','reference_objective_m','preprocessing_s')}
        result['repeats']=len(runs)
        for field in ('planning_s','cold_total_s','reference_s','feature_s','model_s','repair_s','final_validation_s','other_overhead_s'):
            result[field[:-2]+'_median_ms']=float(np.median([r[field] for r in runs]))*1000
        result['improvement_pct']=(first['reference_objective_m']-first['objective_m'])/first['reference_objective_m']*100
        focused_summaries.append(result)
    write_csv('engineering_pairs_recomputed.csv',pairs)
    write_csv('focused_100_400_raw_runs.csv',focused)
    write_csv('focused_100_400_scene_medians.csv',focused_summaries)
    (OUT/'engineering_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='source_hash_checks'},ensure_ascii=False,indent=2))
    print('400-table scene timing breakdown (ms; component medians are not additive):')
    for r in focused_summaries:
        if r['tables']==400:print(r)


if __name__=='__main__':main()
