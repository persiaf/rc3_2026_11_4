"""Independent read-only reconstruction of v5 cache and native MLP records.

No model training, timing benchmark, PT/pickle loading, or input mutation.
Uses the reviewed scalar validator for actual saved-plan revalidation.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys
import time
import numpy as np

sys.dont_write_bytecode = True

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--relocate-from', help='Explicit archived Windows project root for cache manifest absolute paths')
    args=parser.parse_args()
    root=args.source_root.resolve();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(root))
    from rc3_baseline.validator import validate
    started=time.perf_counter();errors=[];checks=[];hashes=[];plans=[]

    def require(value,label):
        checks.append(dict(label=label,passed=bool(value)))
        if not value:errors.append(label)

    def close(actual,expected,label,tol=1e-7):
        require(math.isfinite(float(actual)) and math.isclose(float(actual),float(expected),rel_tol=0,abs_tol=tol),label)

    def file_hashes(manifest,field,label):
        for name,expected in manifest[field].items():
            rel=PurePosixPath(name.replace('\\','/'))
            if PureWindowsPath(name).is_absolute() and args.relocate_from:
                try:
                    rel=PurePosixPath(*PureWindowsPath(name).relative_to(PureWindowsPath(args.relocate_from)).parts)
                except ValueError:
                    pass
            safe=not rel.is_absolute() and '..' not in rel.parts and ':' not in str(rel)
            p=root.joinpath(*rel.parts)
            actual=sha(p) if safe and p.is_file() else None
            same=actual==expected
            hashes.append(dict(group=label,path=name,matches=same))
            require(same,label+':hash:'+name)

    def validate_plan(scene,plan,sparse,label):
        verdict=validate(scene,plan,sparse)
        valid=verdict['valid_model_plan'] and verdict['all_tasks_served']
        cost=sum(math.dist(a,b) for a,b in zip(plan['route_polyline'],plan['route_polyline'][1:]))+.75*len(plan['selected_candidate_ids'])
        close(plan['stop_penalty_equivalent_m'],.75,label+':penalty',1e-12)
        close(cost,plan['objective_length_plus_stops_m'],label+':cost',1e-6)
        require(valid,label+':validation')
        plans.append(dict(label=label,valid=bool(valid),cost=cost,errors=verdict['errors']))
        return cost

    cache={};cache_rows=[];canonical={};expected_cases=read(root/'rc3_v3/data_v3_20261102/validation_cases.json')
    for stage,expected_count in [('equivalence',36),('benchmark',108)]:
        folder=root/'rc3_v5/cache_results'/stage
        manifest=read(folder/'manifest.json');summary=read(folder/'summary.json')
        require(manifest['status']=='complete',stage+':complete')
        require(manifest['cases']==expected_cases,stage+':original_validation_cases')
        file_hashes(manifest,'source_sha256',stage)
        count=0;fs=sorted((folder/'plans').glob('scene_plan_*.json'))
        require(len(fs)==18,stage+':scene_count')
        for i,path in enumerate(fs):
            data=read(path);scene=data['scene'];seed=scene['seed']
            require(seed==expected_cases[i]['seed'],stage+':case:'+str(i))
            require(set(data['plans'])==set(data['runs']),stage+':run_plan_keys:'+str(i))
            first=next(iter(data['plans'].values()))
            if stage=='equivalence':canonical[seed]=first
            else:require(first==canonical[seed],stage+':cross_stage_plan_identity:'+str(i))
            for name,plan in data['plans'].items():
                label=f'{stage}:{seed}:{name}'
                cost=validate_plan(scene,plan,data['feasibility'],label)
                require(plan==first,label+':identity')
                runtime=data['runs'][name]['runtime']
                require(runtime['status']=='complete' and runtime['failures']==[] and not runtime['fallback_used'],label+':runtime')
                close(cost,runtime['objective_m'],label+':runtime_cost',1e-6)
                require(cost<=runtime['reference_objective_m']+1e-6,label+':recorded_anchor_bound')
                require(runtime['dp_calls']==8,label+':eight_calls')
                if stage=='benchmark':
                    row=data['runs'][name]['record'];cache_rows.append(row)
                    require(row['scene']==i and row['seed']==seed,label+':record_scene')
                    close(row['objective'],cost,label+':record_cost',1e-6)
                    require(row['planning_ms']>=runtime['planning_s']*1000,label+':timer_encloses_runtime')
                count+=1
        require(count==expected_count==summary['saved_plans'],stage+':plan_count')
        cache[stage]=dict(scenes=len(fs),plans_revalidated=count,all_plan_objects_equal=True)
    heads=read(root/'rc3_v5/cache_results/equivalence/head_checks.json')
    require(len(heads)==144,'heads:count')
    require({(x['scene'],x['step']) for x in heads}=={(i,j) for i in range(18) for j in range(8)},'heads:state_grid')
    require(all(x['all_heads_bitwise_equal'] and x['max_output_error']==0 and x['original_action']==x['cached_action'] for x in heads),'heads:records_agree')
    cache['equivalence']['stored_head_records_recounted']=144
    cache['equivalence']['actual_full_144_head_recomputation_this_audit']=False
    csv_rows=list(csv.DictReader((root/'rc3_v5/cache_results/benchmark/repeat_metrics.csv').open(encoding='utf-8-sig',newline='')))
    indexed={(int(x['scene']),x['method'],int(x['repeat'])):x for x in csv_rows}
    require(len(indexed)==108==len(cache_rows),'benchmark:csv_row_count')
    for row in cache_rows:
        other=indexed[row['scene'],row['method'],row['repeat']]
        for k,v in row.items():
            valid=other[k]==str(v) or (type(v) in (float,int) and float(other[k])==v)
            require(valid,f'benchmark:csv:{row["scene"]}:{row["method"]}:{row["repeat"]}:{k}')
    medians=[]
    for i in range(18):
        item={'scene':i}
        for method in ['original','cached']:
            rows=[r for r in cache_rows if r['scene']==i and r['method']==method]
            require(len(rows)==3,f'benchmark:repeats:{i}:{method}')
            for key in ['planning_ms','model_ms','graph_ms','inference_ms']:
                item[method+'_'+key]=float(np.median([r[key] for r in rows]))
        item['speedup']=item['original_planning_ms']/item['cached_planning_ms']
        item['reduction_pct']=100*(1-item['cached_planning_ms']/item['original_planning_ms'])
        medians.append(item)
    groups={method:{k:float(np.median([x[method+'_'+k] for x in medians])) for k in ['planning_ms','model_ms','graph_ms','inference_ms']} for method in ['original','cached']}
    result=dict(summary=groups,median_paired_speedup=float(np.median([x['speedup'] for x in medians])),median_paired_reduction_pct=float(np.median([x['reduction_pct'] for x in medians])),faster_scenes=sum(x['speedup']>1 for x in medians),slower_scenes=sum(x['speedup']<1 for x in medians),ratio_of_group_medians=groups['original']['planning_ms']/groups['cached']['planning_ms'])
    result['reduction_from_group_medians_pct']=100*(1-groups['cached']['planning_ms']/groups['original']['planning_ms'])
    expected=read(root/'rc3_v5/cache_results/benchmark/summary.json')
    require(groups==expected['summary'],'benchmark:group_statistics')
    for k in ['median_paired_speedup','median_paired_reduction_pct','faster_scenes','slower_scenes','ratio_of_group_medians']:
        close(result[k],expected[k],'benchmark:summary:'+k)
    cache['benchmark'].update(result)
    save(out/'cache_scene_medians.json',medians)

    folder=root/'rc3_v5/mlp_budget_windows';manifest=read(folder/'manifest.json');expected=read(folder/'summary.json')
    file_hashes(manifest,'frozen_source_sha256','native_mlp')
    require(manifest['status']=='complete','native_mlp:complete')
    native_rows=[];scene_rows=[];reference_checks=0
    fs=sorted((folder/'plans').glob('scene_plan_*.json'))
    require(len(fs)==18,'native_mlp:scene_count')
    for i,path in enumerate(fs):
        data=read(path);scene=data['scene'];seed=scene['seed'];sparse=data['feasibility']
        ref=validate_plan(scene,data['fixed3_reference_plan'],sparse,f'native_reference:{seed}');reference_checks+=1
        declared=manifest['cases'][i]
        require(all(data['metadata'][k]==declared[k] for k in ['seed','nx','ny','mode']) and data['metadata']['case_index']==declared['original_case_index'] and data['metadata']['tables']==len(scene['tables']) and data['metadata']['tasks']==len(scene['tasks']),'native_mlp:case:'+str(i))
        order=[]
        for name,r in data['runs'].items():
            label=f'native_mlp:{seed}:{name}';plan=data['plans'][name]
            cost=validate_plan(scene,plan,sparse,label)
            require(cost<=ref+1e-6,label+':certified_reference_bound')
            close(cost,r['objective_m'],label+':recorded_cost',1e-6)
            close(ref,r['reference_objective_m'],label+':recorded_reference',1e-6)
            require(r['status']=='complete' and not r['runtime']['failures'] and not r['fallback_used'],label+':runtime')
            config=r['config']
            require(config['seconds']==.4 and config['horizon']==256 and config['repeats']==3 and config['kind']=='wall',label+':budget')
            require(r['planning_s']>=r['inner_planning_s'] and r['stop_reason']=='deadline',label+':timing')
            close(r['overrun_s'],max(0,r['planning_s']-.4),label+':overrun',1e-10)
            row=dict(scene=i,seed=seed,method=config['id'],repeat=r['repeat'],cost=cost,reference=ref,planning_ms=r['planning_s']*1000,calls=r['dp_calls'],overrun_ms=r['overrun_s']*1000,deadline_exceeded=r['deadline_exceeded'],fallback=r['fallback_used'],execution_order=r['execution_order'])
            native_rows.append(row);order.append((r['execution_order'],config['id'],r['repeat']))
        require(len(order)==6 and sorted(x[0] for x in order)==list(range(6)),f'native_mlp:{seed}:execution_positions')
        for method in ['detour_wall400','v3_mlp_wall400']:
            rows=[r for r in native_rows if r['seed']==seed and r['method']==method]
            require(len(rows)==3 and {r['repeat'] for r in rows}=={0,1,2},f'native_mlp:{seed}:{method}:repeats')
            scene_rows.append(dict(scene=i,seed=seed,method=method,cost=float(np.mean([r['cost'] for r in rows])),reference=ref,time_ms=float(np.median([r['planning_ms'] for r in rows])),calls=float(np.mean([r['calls'] for r in rows])),different_costs=len(set(round(r['cost'],8) for r in rows)),different_calls=len(set(r['calls'] for r in rows))))
    groups={}
    for method in ['detour_wall400','v3_mlp_wall400']:
        rows=[r for r in native_rows if r['method']==method];scenes=[r for r in scene_rows if r['method']==method]
        values=dict(scenes=len(scenes),runs=len(rows),mean_objective_m=float(np.mean([r['cost'] for r in scenes])),mean_reduction_vs_fixed3_pct=float(np.mean([100*(r['reference']-r['cost'])/r['reference'] for r in scenes])),median_scene_planning_ms=float(np.median([r['time_ms'] for r in scenes])),p95_scene_planning_ms=float(np.quantile([r['time_ms'] for r in scenes],.95)),p95_all_run_planning_ms=float(np.quantile([r['planning_ms'] for r in rows],.95)),max_run_planning_ms=max(r['planning_ms'] for r in rows),mean_dp_calls=float(np.mean([r['calls'] for r in rows])),median_dp_calls=float(np.median([r['calls'] for r in rows])),deadline_exceeded_runs=sum(r['deadline_exceeded'] for r in rows),max_overrun_ms=max(r['overrun_ms'] for r in rows),mean_overrun_ms=float(np.mean([r['overrun_ms'] for r in rows])),fallback_runs=sum(r['fallback'] for r in rows))
        for k,v in values.items():close(v,expected['methods'][method][k],f'native_mlp:aggregate:{method}:{k}')
        values['scenes_with_repeat_cost_changes']=sum(r['different_costs']>1 for r in scenes)
        values['scenes_with_repeat_call_changes']=sum(r['different_calls']>1 for r in scenes)
        groups[method]=values
    indexed={(r['seed'],r['method']):r for r in scene_rows};deltas=[];pairs=[]
    for seed in sorted({r['seed'] for r in scene_rows}):
        a=indexed[seed,'v3_mlp_wall400'];b=indexed[seed,'detour_wall400']
        delta=100*(b['cost']-a['cost'])/b['cost'];deltas.append(delta)
        pairs.append(dict(seed=seed,mlp_cost=a['cost'],detour_cost=b['cost'],reduction_pct=delta,time_ratio=a['time_ms']/b['time_ms']))
    delta=np.asarray(deltas);rng=np.random.default_rng(731203)
    ci=np.quantile(delta[rng.integers(len(delta),size=(20000,len(delta)))].mean(1),[.025,.975])
    paired=dict(scenes=len(delta),mean_reduction_pct=float(delta.mean()),ci95_pct=ci.tolist(),wins=int((delta>1e-9).sum()),ties=int((abs(delta)<=1e-9).sum()),losses=int((delta<-1e-9).sum()),median_paired_time_ratio=float(np.median([r['time_ratio'] for r in pairs])),ratio_of_median_times=groups['v3_mlp_wall400']['median_scene_planning_ms']/groups['detour_wall400']['median_scene_planning_ms'])
    for k,v in paired.items():
        if isinstance(v,list):require(np.allclose(v,expected['paired_mlp_vs_detour'][k],rtol=0,atol=1e-7),'native_mlp:paired:'+k)
        else:close(v,expected['paired_mlp_vs_detour'][k],'native_mlp:paired:'+k)
    native=dict(scenes=18,revalidated_returned_plans=108,revalidated_references=reference_checks,groups=groups,paired=paired,platform=manifest['platform'])
    save(out/'native_repeat_metrics.json',native_rows);save(out/'native_scene_metrics.json',scene_rows);save(out/'native_paired_scenes.json',pairs)

    summary=dict(all_passed=not errors,errors=errors,cache=cache,native_mlp=native,revalidated_returned_plans=252,revalidated_reference_plans=18,all_plan_records_valid=all(x['valid'] for x in plans),hash_entries_checked=len(hashes),hash_entries_matched=sum(x['matches'] for x in hashes),checks_count=len(checks),elapsed_s=time.perf_counter()-started,scope='Actual saved-plan revalidation with reviewed scalar validator; independent statistic reconstruction. Recorded original timing values, no new timing benchmarks or training. 144 stored head checks recounted, not all recomputed; cache semantics exercised in separate 11 actual unit tests.')
    save(out/'all_plan_checks.json',plans);save(out/'hash_checks.json',hashes);save(out/'record_checks.json',checks);save(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    raise SystemExit(0 if summary['all_passed'] else 2)

if __name__=='__main__':main()
