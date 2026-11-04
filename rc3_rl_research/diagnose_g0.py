"""Predeclared finite-depth development oracle; every executed query is logged."""
from __future__ import annotations
import argparse
import copy
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import time

from rc3_fast.scenes import make_complex_scene
from .adapter import ResearchAdapter
from .rl_env import SearchEnv
from .types import SearchConfig, SCHEMA


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def _fork(env):
    # Trusted in-memory branches of an already certified state. A persisted
    # state is never loaded this way: load_state_dict independently certifies it.
    child=copy.copy(env)
    shared={'adapter','proposer','action_builder','certifier'}
    for key,value in vars(env).items():
        if key not in shared:setattr(child,key,copy.deepcopy(value))
    return child


def branch_ids(obs,limit):
    search=[i for i,ok in enumerate(obs['mask']) if ok and obs['actions'][i]['kind']!='stop']
    stop=[i for i,ok in enumerate(obs['mask']) if ok and obs['actions'][i]['kind']=='stop']
    return search[:limit-1]+stop


def diagnose_case(case,protocol,directory):
    started=time.perf_counter();directory.mkdir(parents=True,exist_ok=True)
    scene=make_complex_scene(case['nx'],case['ny'],case['seed'],case['mode'])
    write_json(directory/'scene.json',scene)
    adapter=ResearchAdapter.from_scene(scene)
    write_json(directory/'input.json',adapter.payload())
    preparation_s=time.perf_counter()-started
    env=SearchEnv(adapter,SearchConfig(**protocol['config']),seed=case['seed'])
    initial=env.state_dict()
    rows=[];menus=[];leaves=[];physical=Counter();failures=Counter()
    menu_log=gzip.open(directory/'menus.jsonl.gz','wt',encoding='utf-8')
    query_log=gzip.open(directory/'queries.jsonl.gz','wt',encoding='utf-8')
    maxdepth=protocol['diagnostic_depth'];branchlimit=protocol['branch_limit_including_stop']
    root_obs=env.observe()
    physical['menu_builds']+=1;physical['menu_work_units']+=root_obs['pending_menu_work']

    def visit(parent,path):
        obs=parent.observe();menu_id=len(menus);selected=branch_ids(obs,branchlimit)
        menus.append(menu_id)
        menu_log.write(json.dumps(dict(menu_id=menu_id,path=path,selected_action_ids=selected,observation=obs),allow_nan=False)+'\n')
        for action_id in selected:
            child=_fork(parent);next_path=path+[action_id]
            prev_calls=child.counters['certification_calls'];prev_builds=child.counters['menu_builds']
            tick=time.perf_counter()
            nextobs,reward,terminated,_,info=child.step(action_id)
            step_s=time.perf_counter()-tick
            newmenus=child.counters['menu_builds']-prev_builds
            physical['menu_builds']+=newmenus
            physical['menu_work_units']+=nextobs['pending_menu_work'] if newmenus else 0
            physical['proposal_calls']+=int(info['proposal'] is not None)
            physical['proposal_work_units']+=info['proposal_work_units']
            physical['selected_edges']+=1
            if info['failure_reason']:failures[info['failure_reason']]+=1
            leaf=terminated or len(next_path)>=maxdepth
            if leaf and not child.done:child.terminate('diagnostic_depth')
            physical['certification_calls']+=child.counters['certification_calls']-prev_calls
            compact_info={k:v for k,v in info.items() if k!='sampled_observation'}
            row=dict(query_id=len(rows),menu_id=menu_id,path=next_path,depth=len(next_path),
                action_id=action_id,action=info['action'],before_current=parent.current,
                current=list(child.current),best=list(child.best),
                current_cost=child.current_cost,best_cost=child.best_cost,
                current_change=child.current_cost-parent.current_cost,
                reward=reward,total_reward=child.total_reward,work_used=child.work_used,
                info=compact_info,leaf=leaf,step_elapsed_s=step_s,
                finalization=child.finalization,history=list(child.history))
            query_log.write(json.dumps(row,allow_nan=False)+'\n')
            rows.append(row)
            if leaf:
                leaves.append(dict(path=next_path,returned_cost=child.returned_certificate.cost,
                    total_reward=child.total_reward,work_used=child.work_used,
                    fallback=child.finalization['fallback_used'],
                    work_overrun=child.finalization['work_overrun']))
            else:visit(child,next_path)

    try:visit(env,[])
    finally:menu_log.close();query_log.close()
    rootrows=[r for r in rows if r['depth']==1]
    one_step=min(r['best_cost'] for r in rootrows)
    per_root=[]
    for root in rootrows:
        relevant=[v for v in leaves if v['path'][0]==root['action_id']]
        terminal=min(relevant,key=lambda x:(x['returned_cost'],x['path']))
        per_root.append(dict(action_id=root['action_id'],action=root['action'],
            immediate_best_cost=root['best_cost'],first_current_cost=root['current_cost'],
            first_accepted=root['info']['accepted'],descendant=terminal))
    anchor=adapter.anchor.cost;tol=1e-8
    worse=[v for v in per_root if v['first_accepted'] and v['first_current_cost']>anchor+tol]
    neutral=[v for v in per_root if v['first_accepted'] and v['first_current_cost']>=anchor-tol
             and v['action']['kind']!='stop']
    strict=[v for v in worse if v['descendant']['returned_cost']<one_step-tol]
    weak=[v for v in worse if v['descendant']['returned_cost']<anchor-tol]
    nonimprove=[v for v in neutral if v['descendant']['returned_cost']<anchor-tol]
    reversals=[dict(preferred_immediate=a['action_id'],preferred_after_depth3=b['action_id'])
        for a in per_root for b in per_root
        if a['immediate_best_cost']<b['immediate_best_cost']-tol
        and a['descendant']['returned_cost']>b['descendant']['returned_cost']+tol]
    globally_best=min(leaves,key=lambda x:(x['returned_cost'],x['path']))
    monotone=[v for v in per_root if v['first_current_cost']<=anchor+tol]
    best_monotone_first=min(v['descendant']['returned_cost'] for v in monotone)
    # This is only a first-action restriction, not a complete monotone tree.
    summary=dict(case=case,status='complete',problem_id=adapter.problem_id,task_count=len(scene['tasks']),
        anchor_cost=anchor,root_one_step_best=one_step,best_depth3_cost=globally_best['returned_cost'],
        best_monotone_first_depth3_cost=best_monotone_first,
        worse_first_beats_one_step=bool(strict),worse_first_then_record=bool(weak),
        nonimproving_first_then_record=bool(nonimprove),ranking_reversal=bool(reversals),
        worse_first_beats_all_monotone_first=any(v['descendant']['returned_cost']<best_monotone_first-tol for v in worse),
        root_action_summaries=per_root,ranking_reversals=reversals,
        queries=len(rows),menus_expanded=len(menus),leaves=len(leaves),physical_oracle_work=dict(physical),
        failures=dict(failures),accepted=sum(r['info']['accepted'] for r in rows),
        accepted_worse=sum(r['info']['accepted'] and r['current_change']>tol for r in rows),
        globally_pruned_retained=sum(bool((r['info']['proposal'] or {}).get('diagnostics',{}).get('outside_removed_nodes')) for r in rows),
        certification_failures=sum('certification_failed' in (r['info']['failure_reason'] or '') for r in rows),
        work_count_incomplete=any(r['info']['work_count_incomplete'] for r in rows),
        fallback_leaves=sum(v['fallback'] for v in leaves),overrun_leaves=sum(v['work_overrun']>0 for v in leaves),
        max_leaf_work_overrun=max(v['work_overrun'] for v in leaves),
        preparation_s=preparation_s,tree_elapsed_s=time.perf_counter()-started-preparation_s,
        mechanism_scope='Finite shared-menu depth-3 development oracle. No learned model, necessity proof, or deployment speed claim.')
    # Save one mechanically selected replay per scene: lowest action-id strong
    # witness when present, otherwise the globally cheapest diagnosed sequence.
    chosen=(strict[0]['descendant'] if strict else globally_best)
    replay=SearchEnv(adapter,SearchConfig(**protocol['config']),seed=case['seed'])
    replay_started=time.perf_counter()
    for a in chosen['path']:replay.step(a)
    if not replay.done:replay.terminate('diagnostic_depth')
    if abs(replay.returned_certificate.cost-chosen['returned_cost'])>tol:
        raise AssertionError('selected sequence replay changed cost')
    write_json(directory/'selected_replay.json',dict(schema=SCHEMA,adapter=adapter.payload(),
        initial_state=initial,state=replay.state_dict(),plan=replay.returned_certificate.plan))
    summary['selected_path']=chosen['path'];summary['selected_is_strict_witness']=bool(strict)
    summary['verification_replay']=dict(elapsed_s=time.perf_counter()-replay_started,
        proposal_calls=replay.counters['proposal_calls'],proposal_work=replay.counters['proposal_work'],
        menu_builds=replay.counters['menu_builds'],menu_work=replay.counters['menu_work'],
        certification_calls=replay.counters['certification_calls'])
    summary['input_unchanged']=adapter.input_identity()==adapter.problem_id
    write_json(directory/'summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('rc3_rl_research/outputs/g0'))
    parser.add_argument('--case-id',type=int)
    parser.add_argument('--shard-index',type=int,default=0)
    parser.add_argument('--shard-count',type=int,default=1)
    parser.add_argument('--bind-source',action='store_true')
    args=parser.parse_args();d=args.directory
    source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py')) if not p.name.startswith('test_')}
    binding=d/'implementation.json'
    if args.bind_source:
        if binding.exists():raise ValueError('implementation is already bound')
        write_json(binding,source_hashes);print('Implementation bound');return
    if not binding.exists() or json.loads(binding.read_text())!=source_hashes:
        raise ValueError('implementation differs from registered source; do not resume old results')
    if not 0<=args.shard_index<args.shard_count:raise ValueError('invalid shard')
    raw=(d/'protocol.json').read_bytes();actual=hashlib.sha256(raw).hexdigest()
    if actual!=(d/'protocol.sha256').read_text().split()[0]:raise ValueError('protocol hash changed')
    protocol=json.loads(raw);results=[]
    for case in protocol['cases']:
        if args.case_id is not None and case['case_id']!=args.case_id:continue
        if case['case_id']%args.shard_count!=args.shard_index:continue
        target=d/f"case_{case['case_id']:02d}"
        if (target/'summary.json').exists():
            result=json.loads((target/'summary.json').read_text(encoding='utf-8'))
        else:
            try:result=diagnose_case(case,protocol,target)
            except Exception as error:
                target.mkdir(exist_ok=True,parents=True)
                result=dict(case=case,status='failed',error=f'{type(error).__name__}: {error}')
                write_json(target/'summary.json',result)
        results.append(result)
        print(json.dumps(dict(case=case['case_id'],status=result['status'],queries=result.get('queries'),
            strict_witness=result.get('worse_first_beats_one_step'),reversal=result.get('ranking_reversal'),
            error=result.get('error'))),flush=True)
    if args.case_id is None and args.shard_count==1:
        completed=[r for r in results if r['status']=='complete']
        summary=dict(protocol_sha256=actual,states_planned=32,states_completed=len(completed),
            failed_states=[r for r in results if r['status']!='complete'],
            counts={key:sum(r[key] for r in completed) for key in
                    ('worse_first_beats_one_step','worse_first_then_record','nonimproving_first_then_record',
                     'ranking_reversal','worse_first_beats_all_monotone_first')},
            total_queries=sum(r['queries'] for r in completed),
            physical_oracle_work=dict(sum((Counter(r['physical_oracle_work']) for r in completed),Counter())),
            work_overrun_leaves=sum(r['overrun_leaves'] for r in completed),
            fallback_leaves=sum(r['fallback_leaves'] for r in completed),
            work_count_incomplete=any(r['work_count_incomplete'] for r in completed),
            all_inputs_unchanged=all(r['input_unchanged'] for r in completed),
            case_summaries=[dict(case_id=r['case']['case_id'],status=r['status'],summary=f"case_{r['case']['case_id']:02d}/summary.json") for r in results])
        write_json(d/'summary.json',summary)
        print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
