"""Exploratory 18-validation-scene diagnostic, not a frozen-test improvement.

Compare detour, the frozen MLP, and feasible zero/one/two-stop local-gain
ranking. All use the original runtime, 8 original repair calls, identical
reference and final validation. Extra candidate scanning is fully timed.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from unittest.mock import patch

import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]/'v3_upload'
if '--source-root' in sys.argv:
    source_index=sys.argv.index('--source-root')+1
    if source_index>=len(sys.argv):raise SystemExit('--source-root requires a directory')
    ROOT=Path(sys.argv[source_index]).resolve()
if not (ROOT/'rc3_v3/runtime.py').is_file():
    raise SystemExit('Pass --source-root pointing to the unpacked project directory containing rc3_v3/')
sys.path.insert(0,str(ROOT))
from rc3_fast.solver import prepare
from rc3_baseline.validator import validate
from rc3_v3.data import make_case_scene
from rc3_v3.model import load_policy
from rc3_improve.repair import _local
from rc3_v3.environment import FEATURE_NAMES
import rc3_v3.runtime as runtime


class TwoStopFeasibleGain:
    def __init__(self):
        self.env=None
        self.actions=[]

    def select(self,features,mask):
        env=self.env;problem=env.problem;route=env.route
        if env is None:raise ValueError('runtime environment was not bound')
        cover=problem.F.any(2);nodes=np.asarray(route,dtype=np.int64)
        counts=cover[:,nodes[1:-1]-1].sum(1)
        distance=problem.nav['costs'];penalty=problem.penalty
        gains=np.full(len(mask),-np.inf)
        feasible_counts=[]
        for action in np.flatnonzero(mask):
            window=env.windows[int(action)]
            start,end,required,pool,_=_local(problem,route,window,cover,counts)
            local=nodes[start-1:end+1]
            old=float(distance[local[:-1],local[1:]].sum()+penalty*(end-start))
            entrance,exit_node=int(local[0]),int(local[-1])
            feasible=0
            if len(required)==0:
                best=float(distance[entrance,exit_node]);feasible=1
            else:
                full=(1<<len(required))-1
                bits=(cover[np.ix_(required,pool-1)].astype(np.int64)*(1<<np.arange(len(required)))[:,None]).sum(0)
                one_mask=bits==full
                one=distance[entrance,pool]+distance[pool,exit_node]+penalty
                best=float(np.min(one[one_mask])) if one_mask.any() else np.inf
                feasible+=int(one_mask.sum())
                if len(pool)>1:
                    pair_mask=(bits[:,None]|bits[None,:])==full
                    np.fill_diagonal(pair_mask,False)
                    pair_cost=(distance[entrance,pool][:,None]+distance[np.ix_(pool,pool)]
                               +distance[pool,exit_node][None,:]+2*penalty)
                    if pair_mask.any():best=min(best,float(np.min(pair_cost[pair_mask])))
                    feasible+=int(pair_mask.sum())
            gains[action]=max(0.,old-best) if np.isfinite(best) else 0.
            feasible_counts.append(feasible)
        detour=features[:,FEATURE_NAMES.index('detour_excess')]
        eligible=np.flatnonzero(mask)
        best_gain=float(gains[eligible].max())
        if best_gain>1e-8:
            # Stable detour tie-break for numerical equality at the gain scale.
            ties=eligible[np.isclose(gains[eligible],best_gain,rtol=0,atol=1e-9)]
            action=int(ties[np.argmax(detour[ties])])
        else:
            action=int(eligible[np.argmax(detour[eligible])])
        self.actions.append(dict(step=env.steps,action=action,feasible_gain_lower_bound=float(gains[action]),
                                 positive_bound_windows=int(np.sum(gains[eligible]>1e-8)),eligible_windows=len(eligible),
                                 enumerated_feasible_zero_one_two_stop_sequences=int(sum(feasible_counts)),
                                 fallback_to_detour=best_gain<=1e-8))
        return action


def save_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUT/'results')
    parser.add_argument('--source-root',type=Path,default=ROOT)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Use a new output directory; preserve existing diagnostic evidence')
    args.output.mkdir(parents=True)
    base=ROOT/'rc3_v3';cases=json.loads((base/'data_v3_20261102/validation_cases.json').read_text())
    if len(cases)!=18 or any(c['nx']*c['ny']!=100 for c in cases):raise ValueError('expected exactly the frozen 18 hundred-table validation cases')
    policy=load_policy(base/'deployment/policy.npz')
    original_env=runtime.Env
    rows=[];detailed=[];started=time.perf_counter()
    # Predetermined complete validation set. No test cases are loaded.
    for index,case in enumerate(cases):
        problem=prepare(make_case_scene(case))
        methods=['detour','learned','two_stop_feasible_gain']
        rng=np.random.default_rng(int(case['seed'])+20261103)
        for label in rng.permutation(methods):
            label=str(label);ranker=TwoStopFeasibleGain() if label=='two_stop_feasible_gain' else policy
            def factory(*a,**kw):
                env=original_env(*a,**kw);ranker.env=env;return env
            context=patch.object(runtime,'Env',factory) if label=='two_stop_feasible_gain' else nullcontext()
            tick=time.perf_counter()
            try:
                with context:
                    plan,info=runtime.run(problem,method='detour' if label=='detour' else 'learned',
                                          policy=None if label=='detour' else ranker,horizon=8)
                elapsed=time.perf_counter()-tick
                check=validate(problem.scene,plan,problem.sparse)
                if not check['valid_model_plan'] or not check['all_tasks_served']:raise ValueError(check)
                trace=[t for t in info['trace'] if t['event']=='provisional_repair']
                gain_checks=[]
                if label=='two_stop_feasible_gain':
                    before=info['reference_objective_m']
                    for action,event in zip(ranker.actions,trace):
                        actual=before-event['objective_m'];before=event['objective_m']
                        gain_checks.append(dict(**action,realized_gain=actual,
                                                lower_bound_holds=action['feasible_gain_lower_bound']<=actual+1e-7))
                row=dict(scene_id=index,seed=case['seed'],mode=case['mode'],method=label,status='complete',
                         objective_m=info['objective_m'],reference_m=info['reference_objective_m'],
                         gain_pct=(info['reference_objective_m']-info['objective_m'])/info['reference_objective_m']*100,
                         planning_ms=elapsed*1000,cold_total_ms=(elapsed+problem.preprocessing_s)*1000,
                         preprocessing_ms=problem.preprocessing_s*1000,dp_calls=info['dp_calls'],accepted=info['accepted'],
                         fallback_used=info['fallback_used'],stop_reason=info['stop_reason'],
                         valid=True,all_tasks_served=True,reference_bound=info['objective_m']<=info['reference_objective_m']+1e-8,
                         model_ms=info['model_s']*1000,feature_ms=info['feature_s']*1000,repair_ms=info['repair_s']*1000,
                         reference_ms=info['reference_s']*1000,final_validation_ms=info['final_validation_s']*1000,
                         lower_bound_checks=len(gain_checks),lower_bound_failures=sum(not r['lower_bound_holds'] for r in gain_checks))
                detailed.append(dict(scene_id=index,seed=case['seed'],method=label,info=info,
                                     independent_validation=check,gain_checks=gain_checks))
            except Exception as error:
                row=dict(scene_id=index,seed=case['seed'],mode=case['mode'],method=label,status='failed',error=str(error),planning_ms=(time.perf_counter()-tick)*1000)
            rows.append(row)
        print(f'validation diagnostic {index+1}/18 elapsed={time.perf_counter()-started:.1f}s',flush=True)
    summaries=[];comparisons=[]
    for label in methods:
        selected=[r for r in rows if r['method']==label and r['status']=='complete']
        summaries.append(dict(method=label,total=sum(r['method']==label for r in rows),success=len(selected),
                              mean_gain_pct=float(np.mean([r['gain_pct'] for r in selected])),
                              median_planning_ms=float(np.median([r['planning_ms'] for r in selected])),
                              median_cold_total_ms=float(np.median([r['cold_total_ms'] for r in selected])),
                              mean_dp_calls=float(np.mean([r['dp_calls'] for r in selected])),
                              mean_accepted=float(np.mean([r['accepted'] for r in selected])),
                              median_ranking_ms=float(np.median([r['model_ms'] for r in selected]))))
    bykey={(r['seed'],r['method']):r for r in rows if r['status']=='complete'}
    for reference in ('detour','learned'):
        paired=[(r,bykey[r['seed'],reference]) for r in rows if r['method']=='two_stop_feasible_gain' and r['status']=='complete' and (r['seed'],reference) in bykey]
        delta=np.array([(b['objective_m']-a['objective_m'])/b['objective_m']*100 for a,b in paired])
        comparisons.append(dict(method='two_stop_feasible_gain',reference=reference,n=len(paired),
                                mean_cost_reduction_pct=float(delta.mean()),wins=int((delta>1e-7).sum()),ties=int((np.abs(delta)<=1e-7).sum()),losses=int((delta< -1e-7).sum()),
                                median_reference_over_new_time=float(np.median([b['planning_ms']/a['planning_ms'] for a,b in paired]))))
    summary=dict(status='complete',elapsed_s=time.perf_counter()-started,validation_scenes=len(cases),
                 test_scenes_accessed=0,methods=methods,dp_call_cap=8,repeats_per_method_case=1,
                 model_sha256=hashlib.sha256((base/'deployment/policy.npz').read_bytes()).hexdigest(),
                 case_sha256=hashlib.sha256((base/'data_v3_20261102/validation_cases.json').read_bytes()).hexdigest(),
                 platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,
                 all_independently_valid=all(r.get('valid',False) for r in rows),all_reference_bounds=all(r.get('reference_bound',False) for r in rows),
                 chosen_lower_bound_checks=sum(r.get('lower_bound_checks',0) for r in rows),chosen_lower_bound_failures=sum(r.get('lower_bound_failures',0) for r in rows),
                 summaries=summaries,comparisons=comparisons,
                 scope='Post-freeze exploratory validation-set diagnostic, one timed run per method per scene on current Linux hardware; not an independent test or final performance claim.',
                 work_budget_caveat='All use the same 8 original exact-repair calls, but the two-stop rule performs extra fully timed candidate scanning, so equal call counts do not imply equal computational work.',
                 ranking_semantics='Exact feasible zero/one/two-stop restricted replacement gives a lower bound on achievable local improvement, not an upper bound. No windows are pruned based on this gain; all-zero gains fall back to detour. Outside canonicalization can increase realized gain.')
    save_csv(args.output/'case_metrics.csv',rows)
    (args.output/'details.json').write_text(json.dumps(detailed,ensure_ascii=False,indent=2))
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
