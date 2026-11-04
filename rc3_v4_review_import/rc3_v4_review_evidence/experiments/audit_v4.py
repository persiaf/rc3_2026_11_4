"""Read-only reconstruction of frozen v4 logs and full independent plan validation.

Only the stdlib geometric validator is imported from the inspected source tree.
No pickle/PT loads, training, source changes, or repeated route optimization.
"""
from __future__ import annotations
import argparse
import csv
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def key(row): return row['seed'],row['nx'],row['ny'],row['mode']
def dump(path,value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def csvout(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)
def paired(current,reference):
    lookup={key(r):r for r in reference}
    pairs=sorted([(r,lookup[key(r)]) for r in current if key(r) in lookup],key=lambda x:key(x[0]))
    if not pairs:return None
    d=np.array([100*(b['objective_m']-a['objective_m'])/b['objective_m'] for a,b in pairs])
    rng=np.random.default_rng(731203)
    ci=np.quantile(d[rng.integers(len(d),size=(20000,len(d)))].mean(1),[.025,.975])
    return dict(n=len(d),mean_reduction_pct=float(d.mean()),ci_low=float(ci[0]),ci_high=float(ci[1]),
        wins=int((d>1e-7).sum()),ties=int((np.abs(d)<=1e-7).sum()),losses=int((d< -1e-7).sum()),
        median_paired_time_ratio=float(np.median([a['planning_s']/b['planning_s'] for a,b in pairs])),
        ratio_of_time_medians=float(np.median([a['planning_s'] for a,b in pairs])/np.median([b['planning_s'] for a,b in pairs])))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent)
    args=ap.parse_args();root=args.source_root.resolve();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(root))
    from rc3_baseline.validator import validate
    base=root/'rc3_v4/results';manifest=read(base/'manifest.json')
    oldrows=read(base/'metrics.json');oldraw=read(base/'repeat_metrics.json');oldsummary=read(base/'summary.json')
    oldlookup={(key(r),r['config']):r for r in oldrows}
    oldrawlookup={(r['seed'],r['label']):r for r in oldraw}
    checks=[];errors=[]
    for kind in ('model_sha256','source_sha256','data_sha256'):
        for name,expected in manifest[kind].items():
            path=root.joinpath(*name.replace('\\','/').split('/'))
            actual=sha(path) if path.is_file() else None
            checks.append(dict(kind=kind,path=name,expected=expected,actual=actual,match=actual==expected))
    dump(out/'frozen_hash_checks.json',checks)
    files=sorted((base/'plans').glob('scene_plan_*.json'))
    raw=[];rows=[];planchecks=[];aggregates=[];seenmeta=[]
    started=time.perf_counter()
    for file in files:
        payload=read(file);meta=payload['metadata'];seenmeta.append(meta);groups=defaultdict(list)
        if set(payload['runs'])!=set(payload['plans']):errors.append(file.name+':run/plan names differ')
        if set(payload['execution_order'])!=set(payload['runs']) or len(payload['execution_order'])!=len(payload['runs']):errors.append(file.name+':bad schedule')
        fixed_costs=[]
        for name in payload['runs']:
            record=payload['runs'][name];spec=record['config'];plan=payload['plans'].get(name)
            if plan is None:
                errors.append(file.name+':missing '+name);continue
            v=validate(payload['scene'],plan,payload['feasibility'])
            objective=v['independent_polyline_length_m']+.75*len(plan['selected_candidate_ids'])
            valid=bool(v['valid_model_plan'] and v['all_tasks_served'] and abs(plan['stop_penalty_equivalent_m']-.75)<1e-12)
            entry=dict(file=file.name,label=name,seed=meta['seed'],valid=valid,
                objective_m=objective,cost_matches_record=abs(objective-record['objective_m'])<1e-6,
                cost_matches_export=abs(objective-plan['objective_length_plus_stops_m'])<1e-6,
                reported_reference=record['reference_objective_m'],errors=v['errors'])
            if not all(entry[k] for k in ('valid','cost_matches_record','cost_matches_export')):errors.append(file.name+':'+name+':invalid')
            planchecks.append(entry)
            if spec['id']=='fixed3':fixed_costs.append(objective)
            rr=dict(**meta,label=name,config=spec['id'],kind=spec['kind'],repeat=record['repeat'],
                objective_m=objective,reference_objective_m=record['reference_objective_m'],status=record['status'],
                planning_s=record['planning_s'],preprocessing_s=payload['preprocessing_s'],
                dp_calls=record['dp_calls'],accepted=record['accepted'],fallback_used=record['fallback_used'],
                stop_reason=record['stop_reason'],deadline_exceeded=record['deadline_exceeded'],overrun_s=record['overrun_s'],
                policy_graph_s=record['policy_graph_s'],policy_inference_s=record['policy_inference_s'],
                policy_calls=record['policy_calls'],policy_stops=record['policy_stops'])
            for field in ('reference_s','feature_s','model_s','repair_s','final_validation_s','thread_session_s'):
                rr[field]=record['runtime'][field]
            rr['runtime_other_s']=rr['planning_s']-sum(rr[f] for f in ('reference_s','feature_s','model_s','repair_s','final_validation_s','thread_session_s'))
            old=oldrawlookup[(meta['seed'],name)]
            if any(abs(rr[f]-old[f])>1e-8 for f in ('objective_m','planning_s','policy_graph_s','policy_inference_s')):errors.append(file.name+':'+name+':repeat aggregate mismatch')
            raw.append(rr);groups[spec['id']].append(rr)
        reference=float(np.mean(fixed_costs))
        for pc in planchecks[-len(payload['runs']):]:
            pc['fixed3_objective_m']=reference;pc['fixed3_bound']=pc['objective_m']<=reference+1e-7
            pc['reference_matches_fixed3']=abs(pc['reported_reference']-reference)<1e-6
            if not pc['fixed3_bound'] or not pc['reference_matches_fixed3']:errors.append(file.name+':'+pc['label']+':reference bound')
        for config,items in groups.items():
            first=items[0];spec=payload['runs'][first['label']]['config']
            row=dict(**meta,config=config,kind=spec['kind'],repeats=len(items),status='complete',
                objective_m=float(np.mean([v['objective_m'] for v in items])),reference_objective_m=reference,
                improvement_pct=float(np.mean([100*(reference-v['objective_m'])/reference for v in items])),
                planning_s=float(np.median([v['planning_s'] for v in items])),preprocessing_s=payload['preprocessing_s'],
                dp_calls=float(np.mean([v['dp_calls'] for v in items])),accepted=float(np.mean([v['accepted'] for v in items])),
                overrun_rate=float(np.mean([v['deadline_exceeded'] for v in items])),
                policy_stop_rate=float(np.mean([v['stop_reason']=='policy_stop' for v in items])),
                max_overrun_s=float(max(v['overrun_s'] for v in items)))
            row['total_s']=row['planning_s']+row['preprocessing_s']
            for field in ('policy_graph_s','policy_inference_s','reference_s','feature_s','model_s','repair_s','final_validation_s','thread_session_s','runtime_other_s'):
                row[field]=float(np.median([v[field] for v in items]))
            old=oldlookup[(key(row),config)]
            diffs={f:row[f]-old[f] for f in ('objective_m','improvement_pct','planning_s','total_s','dp_calls','accepted','overrun_rate','policy_stop_rate')}
            same=all(abs(x)<1e-8 for x in diffs.values()) and len(items)==spec['repeats'] and len({v['repeat'] for v in items})==len(items)
            aggregates.append(dict(seed=meta['seed'],config=config,matching=same,diffs=diffs))
            if not same:errors.append(file.name+':'+config+':aggregate mismatch')
            rows.append(row)
        dump(out/'progress.json',dict(scenes=len(seenmeta),plans=len(raw),errors=len(errors)))
    full_validation_s=time.perf_counter()-started
    csvout(out/'reconstructed_scene_metrics.csv',rows);csvout(out/'reconstructed_repeat_metrics.csv',raw)
    dump(out/'all_plan_checks.json',planchecks);dump(out/'aggregate_checks.json',aggregates)
    sizes=Counter(m['tables'] for m in seenmeta);kinds=Counter(r['kind'] for r in rows);runs_by_kind=Counter(r['kind'] for r in raw)
    stats=[];comparisons=[]
    groups={'main_100':lambda r:r['tables']==100,'transfer_36':lambda r:r['tables']==36,
        'transfer_196':lambda r:r['tables']==196,'stress_400':lambda r:r['tables']==400,'all_39':lambda r:True}
    groups.update({f'main_mode_{m}':lambda r,m=m:r['tables']==100 and r['mode']==m for m in {r['mode'] for r in rows}})
    for group,predicate in groups.items():
        subset=[r for r in rows if predicate(r)]
        for config in sorted({r['config'] for r in subset}):
            rr=[r for r in subset if r['config']==config]
            s=dict(group=group,config=config,n=len(rr),kind=rr[0]['kind'],
                mean_objective_m=float(np.mean([r['objective_m'] for r in rr])),
                mean_improvement_pct=float(np.mean([r['improvement_pct'] for r in rr])),
                mean_dp_calls=float(np.mean([r['dp_calls'] for r in rr])),
                mean_stop_rate=float(np.mean([r['policy_stop_rate'] for r in rr])),
                mean_overrun_rate=float(np.mean([r['overrun_rate'] for r in rr])),
                max_overrun_ms=1000*max(r['max_overrun_s'] for r in rr))
            for field in ('planning_s','total_s','policy_graph_s','policy_inference_s','reference_s','feature_s','model_s','repair_s','final_validation_s','thread_session_s','runtime_other_s'):
                s['median_'+field[:-2]+'_ms']=1000*float(np.median([r[field] for r in rr]))
            s['p95_planning_ms']=1000*float(np.quantile([r['planning_s'] for r in rr],.95))
            stats.append(s)
        specs=[('structured8',r) for r in ('control8','detour8','v3_mlp8','fixed3')]
        specs += [('control8','detour8'),('detour8','fixed3'),('v3_mlp8','detour8'),('v3_mlp8','fixed3')]
        specs += [('structured_wall400',r) for r in ('control_wall400','detour_wall400')]
        for seed in (17,29,43):specs += [(f'structured_seed{seed}_calls8',r) for r in (f'control_seed{seed}_calls8','detour8')]
        for config,reference in specs:
            value=paired([r for r in subset if r['config']==config],[r for r in subset if r['config']==reference])
            if value:comparisons.append(dict(group=group,config=config,reference=reference,**value))
    csvout(out/'group_statistics.csv',stats);csvout(out/'paired_comparisons.csv',comparisons)
    # Reproduce all frozen summary and comparison values without calling report.py.
    look={(s['group'],s['config']):s for s in stats};clook={(c['group'],c['config'],c['reference']):c for c in comparisons}
    summarychecks=[]
    for s in oldsummary['summaries']:
        new=look[s['group'],s['config']]
        mapping={'total':'n','mean_objective_m':'mean_objective_m','mean_improvement_pct':'mean_improvement_pct',
            'median_planning_ms':'median_planning_ms','median_total_ms':'median_total_ms','mean_dp_calls':'mean_dp_calls',
            'median_graph_feature_ms':'median_policy_graph_ms','median_graph_inference_ms':'median_policy_inference_ms'}
        ok=all(abs(s[a]-new[b])<1e-7 for a,b in mapping.items())
        summarychecks.append(dict(group=s['group'],config=s['config'],kind='summary',match=ok))
    for c in oldsummary['comparisons']:
        new=clook[c['group'],c['config'],c['reference']]
        ok=all(abs(c[f]-new[f])<1e-7 for f in ('mean_reduction_pct','wins','ties','losses','median_paired_time_ratio')) and np.allclose(c['ci95_pct'],[new['ci_low'],new['ci_high']],atol=1e-9,rtol=0)
        summarychecks.append(dict(group=c['group'],config=c['config'],reference=c['reference'],kind='comparison',match=bool(ok)))
    dump(out/'summary_comparison_checks.json',summarychecks)
    mainstats={s['config']:s for s in stats if s['group']=='main_100'}
    gate={'quality_vs_control8':False,'quality_vs_detour8':False,'median_time_ratio_at_most_1_25':False,'all_plans_valid':not errors and all(c['match'] for c in checks)}
    for reference in ('control8','detour8'):
        c=clook['main_100','structured8',reference]
        gate['quality_vs_'+reference]=c['n']==18 and c['mean_reduction_pct']>=.2 and c['ci_low']>0
    time_ratio=mainstats['structured8']['median_planning_ms']/mainstats['detour8']['median_planning_ms']
    gate['median_time_ratio_at_most_1_25']=time_ratio<=1.25
    decision=dict(adopt_structured=all(gate.values()),gates=gate,ratio_of_median_times=time_ratio)
    dump(out/'adoption_recomputed.json',decision)
    summary=dict(scene_count=len(files),sizes=dict(sizes),scene_method_records=len(rows),method_kind_rows=dict(kinds),repeated_runs=len(raw),run_kind_counts=dict(runs_by_kind),
        plans_valid=sum(p['valid'] for p in planchecks),fixed3_bound_passed=sum(p['fixed3_bound'] for p in planchecks),
        raw_costs_match=sum(p['cost_matches_record'] for p in planchecks),aggregate_records_match=sum(a['matching'] for a in aggregates),
        frozen_hashes=len(checks),frozen_hashes_match=sum(c['match'] for c in checks),
        summaries_and_comparisons=len(summarychecks),all_summary_comparison_checks=all(c['match'] for c in summarychecks),
        full_validation_elapsed_s=full_validation_s,errors=errors,all_passed=not errors and all(c['match'] for c in checks),
        counts_match_manifest=len(files)==manifest['cases'] and len(raw)==manifest['planned_runs'],
        bootstrap_samples=20000,bootstrap_seed=731203,bootstrap_unit='scene, using sorted scene keys; training-seed uncertainty not included',
        original_performance_platform=manifest['platform'],scope='Recorded original timings reconstructed; all saved plans revalidated. No complete performance re-benchmark or model training.')
    dump(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
