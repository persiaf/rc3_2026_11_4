"""Independent reconstruction of uploaded or native frozen-MLP budget runs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from rc3_baseline.validator import validate


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def audit(folder,root):
    folder,root=Path(folder),Path(root);manifest=read(folder/'manifest.json');original=read(folder/'summary.json')
    errors=[];scenes=[];all_runs=[];hash_checks=[]
    for relative,expected in manifest['frozen_source_sha256'].items():
        path=root.joinpath(*relative.replace('\\','/').split('/'))
        if '..' in Path(relative).parts:raise ValueError('unsafe source reference')
        actual=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        same=actual==expected;hash_checks.append(dict(path=relative,unchanged=same))
        if not same:errors.append('source mismatch: '+relative)
    for path in sorted((folder/'plans').glob('scene_plan_*.json')):
        data=read(path);scene=data['scene'];sparse=data['feasibility'];ref=data['fixed3_reference_plan']
        ref_check=validate(scene,ref,sparse)
        if not ref_check['valid_model_plan'] or not ref_check['all_tasks_served']:errors.append('invalid saved reference')
        ref_cost=sum(math.dist(a,b) for a,b in zip(ref['route_polyline'],ref['route_polyline'][1:]))+.75*len(ref['selected_candidate_ids'])
        methods={}
        for label,record in data['runs'].items():
            plan=data['plans'].get(label);valid=False
            if plan is not None:
                check=validate(scene,plan,sparse)
                cost=sum(math.dist(a,b) for a,b in zip(plan['route_polyline'],plan['route_polyline'][1:]))+.75*len(plan['selected_candidate_ids'])
                valid=(record['status']=='complete' and check['valid_model_plan'] and check['all_tasks_served']
                    and abs(plan['stop_penalty_equivalent_m']-.75)<1e-12
                    and abs(cost-record['objective_m'])<1e-6 and abs(cost-plan['objective_length_plus_stops_m'])<1e-6
                    and abs(record['reference_objective_m']-ref_cost)<1e-6 and cost<=ref_cost+1e-6)
            if not valid:errors.append(f'{path.name}:{label}:invalid record')
            row=dict(seed=scene['seed'],method=record['config']['id'],cost=cost if plan is not None else None,
                planning_s=record['planning_s'],calls=record.get('dp_calls'),valid=bool(valid),overrun=record.get('deadline_exceeded'))
            all_runs.append(row);methods.setdefault(row['method'],[]).append(row)
        for method,rows in methods.items():
            if len(rows)!=3:errors.append('expected three repetitions')
            if all(r['valid'] for r in rows):
                scenes.append(dict(seed=scene['seed'],method=method,cost=float(np.mean([r['cost'] for r in rows])),
                    time_ms=float(np.median([r['planning_s'] for r in rows])*1000),ref=ref_cost,
                    calls=float(np.mean([r['calls'] for r in rows]))))
    groups={}
    for method in sorted({r['method'] for r in scenes}):
        rs=[r for r in scenes if r['method']==method]
        groups[method]=dict(scenes=len(rs),mean_cost=float(np.mean([r['cost'] for r in rs])),
            mean_reduction_vs_fixed3_pct=float(np.mean([100*(r['ref']-r['cost'])/r['ref'] for r in rs])),
            median_ms=float(np.median([r['time_ms'] for r in rs])),mean_calls=float(np.mean([r['calls'] for r in rs])),
            overruns=sum(r['overrun'] for r in all_runs if r['method']==method))
        saved=original['methods'][method]
        for a,b in [('mean_cost','mean_objective_m'),('mean_reduction_vs_fixed3_pct','mean_reduction_vs_fixed3_pct'),
                    ('median_ms','median_scene_planning_ms'),('mean_calls','mean_dp_calls')]:
            if abs(groups[method][a]-saved[b])>1e-6:errors.append('summary mismatch: '+method+':'+a)
    lookup={(r['seed'],r['method']):r for r in scenes};delta=[]
    for seed in sorted({r['seed'] for r in scenes}):
        a=lookup[seed,'v3_mlp_wall400'];b=lookup[seed,'detour_wall400'];delta.append(100*(b['cost']-a['cost'])/b['cost'])
    delta=np.asarray(delta);rng=np.random.default_rng(731203)
    interval=np.quantile(delta[rng.integers(len(delta),size=(20000,len(delta)))].mean(1),[.025,.975])
    paired=dict(scenes=len(delta),mean_reduction_pct=float(delta.mean()),ci95_pct=interval.tolist(),
                wins=int((delta>1e-9).sum()),ties=int((abs(delta)<=1e-9).sum()),losses=int((delta < -1e-9).sum()))
    saved=original['paired_mlp_vs_detour']
    for key in ('mean_reduction_pct','wins','ties','losses'):
        if abs(paired[key]-saved[key])>1e-7:errors.append('paired mismatch: '+key)
    if not np.allclose(interval,saved['ci95_pct'],rtol=0,atol=1e-7):errors.append('CI mismatch')
    if len(all_runs)!=108 or len(delta)!=18 or manifest['status']!='complete':errors.append('incomplete supplement')
    return dict(all_passed=not errors,errors=errors,scenes=len(delta),runs=len(all_runs),valid=sum(r['valid'] for r in all_runs),
        method_records=len(scenes),groups=groups,paired=paired,hash_checks=hash_checks,
        platform=manifest['platform'],scope='Descriptive post-v4 same-scene supplement; platforms must remain separate.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=audit(a.input,a.root)
    a.output.parent.mkdir(exist_ok=True,parents=True);a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='hash_checks'}));return 0 if result['all_passed'] else 2


if __name__=='__main__':raise SystemExit(main())
