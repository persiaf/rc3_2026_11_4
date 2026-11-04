"""Compact Fixed3-rooted intermediate-state supervision; fresh metadata-only test split.

Only train/validation scenes are constructed here. Data collection mixes oracle,
detour and random trajectories; continuing zero-gain oracle actions is deliberate
state sampling, not the proposed deployed oracle stopping policy.
"""
import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
from pathlib import Path
import pickle
import re
import subprocess
import time

import numpy as np

from rc3_baseline.baseline import write_json
from rc3_fast.scenes import make_complex_scene
from rc3_fast.solver import prepare, route_objective, export_plan


MODES = ('regular', 'mixed', 'blocked')
TEACHERS = ('oracle', 'detour', 'random')
TEST_SCALES = ((4,3), (6,6), (8,8), (10,10), (12,12), (14,14))
STRESS_SCALES = ((16,16), (20,20))


def make_case_scene(case):
    """One scene construction entry point shared with later evaluation."""
    if 'scene' in case:
        return copy.deepcopy(case['scene'])
    return make_complex_scene(int(case['nx']), int(case['ny']), int(case['seed']), case.get('mode','regular'))


def historical_seeds(workspace, exclude=None):
    """Collect seed metadata across previous runs, manifests and imported patch/dev files.

    No objective, labels, paths, or timing values are used for seed selection.
    JSON seed arrays and CSV seed columns are included, not merely *cases.json.
    """
    workspace=Path(workspace).resolve()
    exclude=Path(exclude).resolve() if exclude else None
    command=['rg','--files','-g','*.json','-g','*.jsonl','-g','*.csv',
             '-g','!**/.venv/**','-g','!**/.git/**','-g','!**/node_modules/**']
    found=subprocess.run(command,cwd=workspace,capture_output=True,text=True,encoding='utf-8')
    if found.returncode not in (0,1):
        raise RuntimeError(f'cannot inventory historical metadata: {found.stderr}')
    keys={'seed','scene_seed','instance_seed'}
    seeds=set();sources=[];failures=[]
    def visit(value, local):
        if isinstance(value,dict):
            for key,item in value.items():
                lower=str(key).lower()
                if lower in keys and isinstance(item,(int,np.integer)) and not isinstance(item,bool):
                    local.add(int(item))
                elif lower.endswith('seeds') and isinstance(item,list):
                    local.update(int(v) for v in item if isinstance(v,int) and not isinstance(v,bool))
                elif isinstance(item,(dict,list)):
                    visit(item,local)
        elif isinstance(value,list):
            for item in value:
                if isinstance(item,(dict,list)):visit(item,local)
    started=time.perf_counter()
    for name in found.stdout.splitlines():
        path=workspace/name
        if exclude and path.is_relative_to(exclude):continue
        local=set()
        # Preserve plan seeds encoded in names even if the file's old schema differs.
        if 'scene_plan' in path.name:
            match=re.search(r'seed(\d+)',path.stem) or re.search(r'_(\d+)$',path.stem)
            if match:local.add(int(match.group(1)))
        try:
            if path.suffix=='.csv':
                with path.open(encoding='utf-8-sig',newline='') as handle:
                    reader=csv.DictReader(handle)
                    columns=[key for key in (reader.fieldnames or []) if key.lower() in keys]
                    if columns:
                        for row in reader:
                            for key in columns:
                                try:local.add(int(row[key]))
                                except (TypeError,ValueError):pass
            elif path.suffix=='.jsonl':
                with path.open(encoding='utf-8-sig') as handle:
                    for line in handle:
                        if line.strip():visit(json.loads(line),local)
            else:
                visit(json.loads(path.read_text(encoding='utf-8-sig')),local)
        except (OSError,UnicodeError,ValueError) as error:
            failures.append(dict(path=name,error=str(error)))
        if local:
            seeds.update(local)
            sources.append(dict(path=name,count=len(local),seed_metadata_only=True))
    return seeds,dict(source_files=len(sources),unique_seeds=len(seeds),sources=sources,
                      unreadable=failures,scan_s=time.perf_counter()-started,
                      scope='All discoverable workspace JSON/JSONL seed fields and seed arrays, CSV seed columns, and plan filenames; no costs or labels used.')


def generate_cases(rng, forbidden, train_per_mode=32, val_per_mode=6, test_per_group=3):
    used=set(forbidden)
    def fresh():
        while True:
            value=int(rng.integers(100000,2**31-1))
            if value not in used:
                used.add(value);return value
    cases={split:[dict(nx=10,ny=10,seed=fresh(),mode=mode,scene_id=i*count+j)
                  for i,mode in enumerate(MODES) for j in range(count)]
           for split,count in [('train',train_per_mode),('validation',val_per_mode)]}
    cases['test']=[dict(nx=nx,ny=ny,seed=fresh(),mode=mode,split='test' if nx*ny<=100 else 'ood')
                   for nx,ny in TEST_SCALES for mode in MODES for _ in range(test_per_group)]
    cases['test'].extend(dict(nx=nx,ny=ny,seed=fresh(),mode=mode,split='stress')
                         for nx,ny in STRESS_SCALES for mode in MODES)
    sets={split:{r['seed'] for r in rows} for split,rows in cases.items()}
    if sum(map(len,sets.values()))!=len(set.union(*sets.values())):
        raise AssertionError('fresh split seed collision')
    if set.union(*sets.values()) & forbidden:
        raise AssertionError('historical scene reused')
    return cases


class TeacherLabels:
    """Per-problem full-route cache; outside-coverage changes invalidate the key."""
    def __init__(self, problem, reference_cost, stats, feature_names):
        self.problem,self.reference_cost,self.stats=problem,reference_cost,stats
        self.required_column=feature_names.index('required_tasks')
        self.cache={}

    def evaluate(self, env, features, mask):
        from .environment import canonical_route, repair_window
        from rc3_improve.repair import route_is_valid
        route_key=tuple(map(int,env.route))
        cache=self.cache.setdefault(route_key,{})
        before=route_objective(self.problem,env.route)
        gains=np.zeros(len(env.windows),np.float32)
        for action in np.flatnonzero(mask):
            window=env.windows[int(action)]
            key=(int(window['start']),int(window['end']))
            if key in cache:
                self.stats['teacher_cache_hits']+=1
                gains[action]=cache[key]
                continue
            tick=time.perf_counter()
            candidate=repair_window(self.problem,env.route,window)
            if not route_is_valid(self.problem,candidate):
                raise ValueError('teacher repair returned an invalid candidate')
            candidate=canonical_route(self.problem,candidate)
            gain=100*max(0.,before-route_objective(self.problem,candidate))/max(self.reference_cost,1e-8)
            seconds=time.perf_counter()-tick
            self.stats['teacher_repair_calls']+=1
            self.stats['teacher_repair_s']+=seconds
            if features[action,self.required_column]>0:
                self.stats['teacher_dp_calls']+=1
                self.stats['teacher_dp_s']+=seconds
            cache[key]=float(gain);gains[action]=gain
        return gains


def collect_trajectory(problem, scene_id, horizon, stats):
    from .environment import Env, FEATURE_NAMES
    env=Env(problem,horizon=horizon)
    initial=list(env.route)
    reference=export_plan(problem,env.route,'fixed3_teacher_reference')
    if abs(reference['objective_length_plus_stops_m']-env.reference_cost)>1e-7:
        raise AssertionError('Fixed3 reference cost differs from exported route')
    teacher=TEACHERS[scene_id%len(TEACHERS)]
    rng=np.random.default_rng(int(problem.scene['seed'])+49819)
    labels=TeacherLabels(problem,env.reference_cost,stats,FEATURE_NAMES)
    saved=[];trace=[]
    while True:
        features,mask=env.observe()
        if env.done or not mask.any():break
        step=int(env.steps)
        retain=(step%2==scene_id%2)
        gains=labels.evaluate(env,features,mask) if retain or teacher=='oracle' else None
        if retain:
            legal_gains=gains[mask]
            positive=bool(np.any(legal_gains>1e-8))
            saved.append(dict(features=features.copy(),mask=mask.copy(),gains=gains.copy(),
                              scene_id=int(scene_id),step=step,teacher=teacher,
                              route=np.asarray(env.route,dtype=np.int32),
                              reference_cost=float(env.reference_cost),
                              current_cost=float(route_objective(problem,env.route)),
                              oracle_stop=not positive))
            stats['saved_states']+=1
            stats['positive_states']+=int(positive)
            stats['zero_gain_states']+=int(not positive)
            stats['positive_actions']+=int(np.sum(legal_gains>1e-8))
            stats['zero_gain_actions']+=int(np.sum(legal_gains<=1e-8))
            stats['step_counts'][str(step)]+=1
            stats['teacher_counts'][teacher]+=1
        if teacher=='oracle':
            action=int(np.where(mask,gains,-np.inf).argmax())
        elif teacher=='detour':
            column=FEATURE_NAMES.index('detour_excess')
            action=int(np.where(mask,features[:,column],-np.inf).argmax())
        else:
            action=int(rng.choice(np.flatnonzero(mask)))
        reward,done,info=env.step(action)
        trace.append(dict(step=step,action=action,reward_pct=float(reward),accepted=bool(info['accepted']),
                          objective_m=float(route_objective(problem,env.route))))
    final=export_plan(problem,env.route,'teacher_trajectory_final')
    final_gain=100*(env.reference_cost-final['objective_length_plus_stops_m'])/max(env.reference_cost,1e-8)
    if abs(sum(t['reward_pct'] for t in trace)-final_gain)>1e-7:
        raise AssertionError('teacher reward and canonical exported objective disagree')
    if len(trace)<horizon:stats['early_terminal_scenes']+=1
    stats['trajectory_steps']+=len(trace)
    stats['trajectory_accepted']+=sum(t['accepted'] for t in trace)
    return initial,float(env.reference_cost),saved,dict(scene_id=scene_id,teacher=teacher,trace=trace,
               saved_steps=[s['step'] for s in saved],final_improvement_pct=final_gain,
               early_terminal=len(trace)<horizon)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workspace',type=Path,default=Path.cwd())
    p.add_argument('--train-per-mode',type=int,default=32)
    p.add_argument('--val-per-mode',type=int,default=6)
    p.add_argument('--test-per-group',type=int,default=3)
    p.add_argument('--seed',type=int,default=6241237)
    p.add_argument('--horizon',type=int,default=8)
    args=p.parse_args(argv)
    if min(args.train_per_mode,args.val_per_mode,args.test_per_group,args.horizon)<1:
        p.error('split counts and horizon must be positive')
    if args.output.exists():p.error('use a new output directory')
    from .environment import Env, FEATURE_NAMES, FEATURE_SCHEMA
    forbidden,audit=historical_seeds(args.workspace,args.output)
    # Existing damaged reports do not block new experiments. The audit explicitly
    # limits historical exclusion to readable seed metadata and plan filenames.
    audit['historical_exclusion_complete_for_readable_metadata']=True
    audit['historical_exclusion_has_unreadable_sources']=bool(audit['unreadable'])
    cases=generate_cases(np.random.default_rng(args.seed),forbidden,args.train_per_mode,args.val_per_mode,args.test_per_group)
    args.output.mkdir(parents=True)
    for split,rows in cases.items():write_json(args.output/f'{split}_cases.json',rows)
    audit.update(new_seed=args.seed,split_sizes={k:len(v) for k,v in cases.items()},all_disjoint=True,
                 overlap={k:[] for k in cases},test_costs_not_generated=True)
    write_json(args.output/'SPLIT_AUDIT.json',audit)
    metadata=dict(status='running',schema=FEATURE_SCHEMA,feature_names=FEATURE_NAMES,
                  splits={k:len(v) for k,v in cases.items()},seed=args.seed,horizon=args.horizon,
                  training_scales=['10x10'],validation_scales=['10x10'],reference='Fixed3 (0,1,7), canonical best',
                  test_policy='Frozen case metadata only; test scenes, costs and teacher labels not constructed.',
                  supervision='all legal window gains on real intermediate canonical states; retain all-zero states',
                  sampling='oracle/detour/random rotated by scene_id; parity retains 0,2,4,6 or 1,3,5,7',
                  label_units='percentage of initial Fixed3 canonical objective',
                  cache_key='entire canonical route plus window start/end, separately per problem',
                  teacher_dp_time_scope='repair, canonicalization and objective for positive-required-task windows; not isolated inner DP timing',
                  source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                                 (Path(__file__),Path(__file__).with_name('environment.py'))})
    write_json(args.output/'manifest.json',metadata)
    stats=dict(saved_states=0,positive_states=0,zero_gain_states=0,positive_actions=0,zero_gain_actions=0,
               teacher_repair_calls=0,teacher_dp_calls=0,teacher_cache_hits=0,teacher_repair_s=0.,teacher_dp_s=0.,
               step_counts=Counter(),teacher_counts=Counter(),trajectory_steps=0,trajectory_accepted=0,early_terminal_scenes=0)
    dataset=dict(schema=FEATURE_SCHEMA,feature_names=FEATURE_NAMES,scenes={'train':[],'validation':[]},
                 states=[],initial_routes={'train':[],'validation':[]},reference_costs={'train':[],'validation':[]},
                 trajectories=[],case_metadata={k:cases[k] for k in ('train','validation')})
    started=time.perf_counter()
    try:
        for split in ('train','validation'):
            for index,case in enumerate(cases[split]):
                scene=make_case_scene(case)
                problem=prepare(scene)
                if split=='train':
                    route,reference,states,trajectory=collect_trajectory(problem,index,args.horizon,stats)
                    dataset['states'].extend(states);dataset['trajectories'].append(trajectory)
                else:
                    env=Env(problem,horizon=args.horizon)
                    route,reference=list(env.route),float(env.reference_cost)
                    plan=export_plan(problem,route,'validation_fixed3_reference')
                    if abs(plan['objective_length_plus_stops_m']-reference)>1e-7:
                        raise AssertionError('validation reference export differs')
                dataset['scenes'][split].append(scene)
                dataset['initial_routes'][split].append(route)
                dataset['reference_costs'][split].append(reference)
                # No Problem/F/nav/predecessor matrices are retained in the pickle.
                del problem
                progress=dict(split=split,completed=index+1,total=len(cases[split]),elapsed_s=time.perf_counter()-started,
                              statistics=stats)
                write_json(args.output/'progress.json',progress)
                if (index+1)%6==0 or index+1==len(cases[split]):
                    print(f'{split} {index+1}/{len(cases[split])}; states={len(dataset["states"])} '
                          f'dp_calls={stats["teacher_dp_calls"]}; seconds={progress["elapsed_s"]:.1f}',flush=True)
        dataset['statistics']=dict(stats)
        with (args.output/'data.pkl').open('wb') as handle:
            pickle.dump(dataset,handle,protocol=pickle.HIGHEST_PROTOCOL)
        metadata.update(status='complete',generation_s=time.perf_counter()-started,statistics=stats,
                        saved_step_distribution=dict(stats['step_counts']),
                        all_horizon_steps_represented=all(str(i) in stats['step_counts'] for i in range(args.horizon)),
                        data_bytes=(args.output/'data.pkl').stat().st_size,
                        data_sha256=hashlib.sha256((args.output/'data.pkl').read_bytes()).hexdigest())
        write_json(args.output/'statistics.json',dict(stats))
        write_json(args.output/'manifest.json',metadata)
        print(json.dumps(dict(status='complete',states=len(dataset['states']),data_bytes=metadata['data_bytes'],
                              positive_states=stats['positive_states'],zero_gain_states=stats['zero_gain_states'],
                              step_counts=stats['step_counts'],generation_s=metadata['generation_s'])),flush=True)
    except Exception as error:
        metadata.update(status='failed',error=str(error),statistics=stats,generation_s=time.perf_counter()-started)
        write_json(args.output/'manifest.json',metadata)
        raise


if __name__=='__main__':main()
