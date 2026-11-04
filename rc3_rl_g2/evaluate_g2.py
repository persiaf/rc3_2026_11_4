"""Fresh-scene paired evaluation with complete planning-time boundaries."""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import time
import numpy as np

from rc3_fast.scenes import make_complex_scene
from rc3_fast.solver import prepare
from rc3_v4.runtime import run as frozen_run
from rc3_rl_research.types import SearchConfig
from .adapter_g2 import ResearchAdapterG2
from .env_g2 import SearchEnvG2
from .baselines import make_policy
from .graph_observation import build_policy_observation
from .collector_contract import merge_terminal_events

SCHEMA='rc3-g2-evaluation-v1'


def _json_default(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)


def _write(path,value,compressed=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(value,ensure_ascii=False,indent=None if compressed else 2,
                       allow_nan=False,default=_json_default)
    if compressed:
        with gzip.open(path,'wt',encoding='utf-8') as f:f.write(encoded)
    else:path.write_text(encoded,encoding='utf-8')


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        allow_nan=False,default=_json_default).encode()).hexdigest()


def _scene(case):
    if 'scene' in case:return copy.deepcopy(case['scene'])
    return make_complex_scene(case.get('nx',10),case.get('ny',10),int(case['seed']),case.get('mode','regular'))


def _model_identity(model):
    digest=hashlib.sha256()
    architecture=model.architecture()
    digest.update(json.dumps(architecture,sort_keys=True).encode())
    for name,tensor in sorted(model.state_dict().items()):
        array=tensor.detach().cpu().numpy()
        digest.update(name.encode());digest.update(str(array.shape).encode());digest.update(array.dtype.str.encode());digest.update(array.tobytes())
    return dict(sha256=digest.hexdigest(),architecture=architecture,
                parameters=sum(p.numel() for p in model.parameters()))


def _run_original(scene,name,config,time_limit_s):
    """Same frozen solver, with its time budget reduced by actual preparation."""
    started=time.perf_counter()
    problem=prepare(copy.deepcopy(scene),penalty=.75)
    preprocessing=time.perf_counter()-started
    remaining=None if time_limit_s is None else max(0.,time_limit_s-preprocessing)
    method='fixed3' if name=='original_fixed3' else 'detour'
    plan,diagnostics=frozen_run(problem,method=method,horizon=config.horizon,time_limit_s=remaining)
    elapsed=time.perf_counter()-started
    timing=dict(preprocessing_and_anchor_s=None,preprocessing_s=preprocessing,
        reference_s=diagnostics.get('reference_s',0.),menu_s=diagnostics.get('feature_s',0.),
        graph_feature_s=0.,inference_s=diagnostics.get('model_s',0.),
        proposal_s=diagnostics.get('repair_s',0.),certification_s=0.,
        final_validation_s=diagnostics.get('final_validation_s',0.),policy_update_s=0.)
    explained=sum(float(v) for v in timing.values() if v is not None)
    timing.update(complete_s=elapsed,residual_s=elapsed-explained)
    if not plan.get('validation',{}).get('valid_model_plan'):raise ValueError('original reference export is invalid')
    row=dict(status='complete',J=float(plan['objective_length_plus_stops_m']),
        returned_cost=float(plan['objective_length_plus_stops_m']),anchor_J=float(diagnostics['reference_objective_m']),
        quality_gain_percent=100*(diagnostics['reference_objective_m']-plan['objective_length_plus_stops_m'])/max(diagnostics['reference_objective_m'],1e-8),
        total_reward=None,work_used=None,steps=int(diagnostics.get('dp_calls',0)),
        proposal_calls=int(diagnostics.get('dp_calls',0)),stop_reason=diagnostics.get('stop_reason'),
        explicit_stop=False,stop_at_root=False,fallback=bool(diagnostics.get('fallback_used',False)),valid=True,
        complete_s=elapsed,time_limit_s=time_limit_s,time_overrun_s=max(0.,elapsed-time_limit_s) if time_limit_s is not None else 0.,
        work_overrun=None,shared_action_protocol=False,repeated_proposals=None,unchanged_proposals=None,
        cold_model_load_s=None,timing=timing,
        budget_scope='Frozen solver with full-scene soft-deadline wrapper; no comparable shared-work reward.')
    row['return']=None
    evidence=dict(schema=SCHEMA,scene=scene,plan=plan,diagnostics=diagnostics,row=row,
                  wrapper=dict(frozen_solver=True,preparation_deducted_from_deadline=True,
                               runtime_time_limit_s=remaining,original_sources_modified=False))
    return row,evidence


def _run_shared(scene,name,model,config,time_limit_s,policy_kwargs,seed):
    started=time.perf_counter();deadline=math.inf if time_limit_s is None else started+time_limit_s
    adapter=ResearchAdapterG2.from_scene(scene)
    preprocessing=time.perf_counter()-started
    env=SearchEnvG2(adapter,config,seed,record_transitions=True)
    initial_state=env.state_dict()
    selector=None if model is not None else make_policy('stop' if name=='environment_stop' else name,**policy_kwargs)
    initial_policy=None if selector is None else selector.state_dict()
    graph_s=inference_s=selection_s=update_s=0.
    policy_samples=[];failures=[];records=[];selections=[]
    while not env.done:
        if time.perf_counter()>=deadline:
            env.terminate('external_deadline');break
        try:
            observation=env.observe()
            if time.perf_counter()>=deadline:
                env.terminate('external_deadline');break
            rng_before=copy.deepcopy(env.rng.bit_generator.state)
            if model is not None:
                tick=time.perf_counter();raw=build_policy_observation(env);graph_s+=time.perf_counter()-tick
                if time.perf_counter()>=deadline:
                    env.terminate('external_deadline');break
                tick=time.perf_counter();actions,logp,value=model.act([raw],deterministic=True)
                action=int(actions[0].cpu());inference_s+=time.perf_counter()-tick
                policy_samples.append(dict(observation=raw,action_id=action,
                    old_logp=float(logp[0].cpu()),old_value=float(value[0].cpu()),executed=False))
            else:
                tick=time.perf_counter();action=int(selector.select(observation,env.rng));selection_s+=time.perf_counter()-tick
            selections.append(dict(action_id=action,action=observation['actions'][action],
                rng_before=rng_before,rng_after=copy.deepcopy(env.rng.bit_generator.state),
                transition_index=len(env.transitions),executed=False))
            if time.perf_counter()>=deadline:
                env.terminate('external_deadline');break
            next_obs,reward,terminal,truncated,info=env.step(action)
            selections[-1]['executed']=True
            if model is not None:policy_samples[-1]['executed']=True
            records.append(dict(action_id=action,reward=float(reward),terminated=terminal,
                truncated=truncated,next_observation=next_obs,bootstrap_value=0. if terminal else None))
            if selector is not None:
                tick=time.perf_counter();selector.update(info);update_s+=time.perf_counter()-tick
        except Exception as error:
            failures.append(f'{type(error).__name__}: {error}')
            env.terminate('evaluation_error');break
    certificate,finalization=env.finalize()
    elapsed=time.perf_counter()-started
    ledger=sum(float(r['reward']) for r in records)+sum(float(e['reward']) for e in env.events)
    expected=100*(env.anchor_cost-certificate.cost)/max(env.anchor_cost,1e-8)-config.lambda_work*env.work_used/config.work_scale
    if abs(ledger-env.total_reward)>1e-7 or abs(expected-env.total_reward)>1e-7:
        raise AssertionError('complete episode reward differs from ledger or actual returned plan')
    merged,consumed=merge_terminal_events(records,env.events,env.observe())
    if merged and abs(sum(r['reward'] for r in merged)-env.total_reward)>1e-7:
        raise AssertionError('external event was not folded into actor rewards exactly once')
    if not certificate.plan.get('validation',{}).get('valid_model_plan'):raise ValueError('returned plan is invalid')
    if certificate.cost>env.anchor_cost+1e-7:raise ValueError('returned plan worsens certified anchor')
    timing=dict(preprocessing_and_anchor_s=preprocessing,**env.timings,graph_feature_s=graph_s,
                inference_s=inference_s,selection_s=selection_s,policy_update_s=update_s)
    explained=sum(timing.values());timing.update(complete_s=elapsed,residual_s=elapsed-explained)
    records_info=[t['info'] for t in env.transitions]
    row=dict(status='complete' if not failures else 'complete_with_runtime_error',
        J=certificate.cost,returned_cost=certificate.cost,anchor_J=env.anchor_cost,
        quality_gain_percent=100*(env.anchor_cost-certificate.cost)/max(env.anchor_cost,1e-8),
        total_reward=env.total_reward,work_used=env.work_used,steps=env.steps,
        proposal_calls=env.counters['proposal_calls'],stop_reason=env.stop_reason,
        explicit_stop=bool(records_info and records_info[-1]['action']['kind']=='stop'),
        stop_at_root=bool(len(records_info)==1 and records_info[0]['action']['kind']=='stop'),
        fallback=finalization['fallback_used'],valid=True,complete_s=elapsed,
        time_limit_s=time_limit_s,time_overrun_s=max(0.,elapsed-time_limit_s) if time_limit_s is not None else 0.,
        work_overrun=finalization['work_overrun'],shared_action_protocol=True,
        repeated_proposals=sum(max(0,v['attempts']-1) for v in env.attempt_memory.values()),
        unchanged_proposals=sum(bool(t['proposal'] is not None and t['proposal'].get('valid') and not t['proposal'].get('changed')) for t in records_info),
        cold_model_load_s=None,timing=timing,cache_stats=adapter.cache_stats,failures=failures,
        actor_sample_count=len(merged),actor_reward_sum=sum(v['reward'] for v in merged) if merged else None,
        metrics_only_zero_action=not bool(merged),budget_scope='Shared soft work budget; optional complete scene-to-plan soft deadline.')
    row['return']=env.total_reward
    evidence=dict(schema=SCHEMA,scene=scene,adapter=adapter.payload(),initial_state=initial_state,
        state=env.state_dict(),plan=certificate.plan,row=row,policy_samples=policy_samples,selections=selections,
        actor_records=merged,consumed_terminal_event_ids=sorted(consumed),
        policy_initial_state=initial_policy,policy_state=None if selector is None else selector.state_dict())
    return row,evidence


def evaluate_methods(cases,methods,output_dir,*,config=SearchConfig(),time_limit_s=None,policy_kwargs=None):
    """Evaluate preloaded models and specified rules on identical fresh inputs.

    Learned models are copied to CPU once, outside planning timing. Disk model
    cold-loading is not measured by this object-taking API and is recorded as
    None, rather than misleadingly reported as zero. Every scene/method obtains
    a fresh adapter; per-case method order rotates to reduce fixed-order bias.
    """
    if time_limit_s is not None and (isinstance(time_limit_s,bool) or not math.isfinite(time_limit_s) or time_limit_s<0):
        raise ValueError('invalid complete soft time limit')
    if not cases or not methods:raise ValueError('nonempty cases and methods are required')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+',name) for name in methods):raise ValueError('unsafe method name')
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    if (output/'results.json').exists():raise ValueError('completed evaluation exists; use a fresh output directory')
    policy_kwargs={} if policy_kwargs is None else copy.deepcopy(policy_kwargs)
    evaluators={};setup={}
    if any(model is not None for model in methods.values()):
        import torch
        torch.set_num_threads(1)
    for name,model in methods.items():
        if model is None:
            if name not in ('original_fixed3','original_detour','stop','environment_stop','detour_memory','ils','random'):
                raise ValueError('unknown nonlearning method '+name)
            evaluators[name]=None;setup[name]=dict(model=None,cold_load_s=None)
        else:
            tick=time.perf_counter();evaluators[name]=copy.deepcopy(model).cpu().eval()
            setup[name]=dict(model=_model_identity(evaluators[name]),cpu_copy_and_setup_s=time.perf_counter()-tick,
                             cold_load_s=None,cold_load_scope='Caller supplied an already loaded model; measure disk loading separately.')
    request=dict(schema=SCHEMA,cases=cases,config=asdict(config),time_limit_s=time_limit_s,
                 policy_kwargs=policy_kwargs,method_setup=setup,method_order='rotated by case list position',
                 scene_generation_timed=False,serialization_timed=False)
    _write(output/'request.json',request)
    rows=[];names=list(methods)
    for case_index,case in enumerate(cases):
        scene=_scene(case)
        order=names[case_index%len(names):]+names[:case_index%len(names)]
        for method in order:
            destination=output/f"case_{case_index:03d}_{method}"
            destination.mkdir(parents=True,exist_ok=True)
            _write(destination/'scene.json',scene)
            try:
                if method.startswith('original_'):
                    row,evidence=_run_original(scene,method,config,time_limit_s)
                else:
                    row,evidence=_run_shared(scene,method,evaluators[method],config,time_limit_s,
                        policy_kwargs.get(method,policy_kwargs.get('ils',{}) if method=='ils' else {}),int(case.get('seed',0))+991)
                row.update(method=method,case_id=case.get('case_id',case_index),case_index=case_index,
                    seed=case.get('seed',scene.get('seed')),mode=case.get('mode',scene.get('complexity','regular')),
                    task_count=len(scene['tasks']),table_count=len(scene['tables']),
                    evidence_path=(destination/'evidence.json.gz').as_posix(),
                    model_identity=setup[method].get('model'))
                _write(destination/'evidence.json.gz',evidence,compressed=True)
            except Exception as error:
                row=dict(method=method,case_id=case.get('case_id',case_index),case_index=case_index,
                         seed=case.get('seed',scene.get('seed')),mode=case.get('mode','regular'),
                         status='failed',valid=False,error=f'{type(error).__name__}: {error}')
                _write(destination/'failure.json',row)
            rows.append(row)
            print(json.dumps({k:row.get(k) for k in ('case_id','method','status','J','total_reward','complete_s')}),flush=True)
    rows.sort(key=lambda r:(r['case_index'],names.index(r['method'])))
    _write(output/'results.json',rows)
    flat=[]
    for row in rows:
        flat.append({k:v for k,v in row.items() if not isinstance(v,(dict,list))})
    fields=list(dict.fromkeys(key for row in flat for key in row))
    with (output/'results.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(flat)
    return rows


def select_ils(protocol_path,output_dir):
    """Pre-register the four-setting rule-development search, then freeze it."""
    protocol_path=Path(protocol_path);protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    cases=protocol['splits']['rule_dev']
    grid=[dict(stagnation_limit=s,max_perturbations=p) for s in (2,3) for p in (1,2)]
    spec=dict(schema=SCHEMA,phase='rule_development_only',cases=cases,grid=grid,
        protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        criterion='highest mean total_reward, then lowest mean work_used, then fixed grid order',
        validation_used=False,config=protocol.get('config',asdict(SearchConfig())),
        source_hashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                       for name in ('env_g2.py','action_memory.py','baselines.py','adapter_g2.py','evaluate_g2.py')})
    registered=output/'selection_protocol.json'
    if registered.exists():
        if json.loads(registered.read_text(encoding='utf-8'))!=spec:raise ValueError('registered rule selection protocol changed')
    else:_write(registered,spec)
    if (output/'selected_ils.json').exists():return json.loads((output/'selected_ils.json').read_text(encoding='utf-8'))
    config=SearchConfig(**spec['config']);scores=[]
    for index,setting in enumerate(grid):
        destination=output/f'grid_{index}'
        saved=destination/'results.json'
        rows=json.loads(saved.read_text(encoding='utf-8')) if saved.exists() else evaluate_methods(cases,{'ils':None},destination,config=config,policy_kwargs={'ils':setting})
        if len(rows)!=len(cases) or any(r['status']!='complete' for r in rows):
            raise ValueError('rule-development failure retained; cannot select from incomplete grid')
        scores.append(dict(grid_index=index,settings=setting,mean_return=float(np.mean([r['total_reward'] for r in rows])),
                           mean_work=float(np.mean([r['work_used'] for r in rows])),scenes=len(rows)))
    winner=min(scores,key=lambda r:(-r['mean_return'],r['mean_work'],r['grid_index']))
    result=dict(schema=SCHEMA,selection_protocol_sha256=hashlib.sha256(registered.read_bytes()).hexdigest(),
                selected=winner['settings'],scores=scores,validation_used=False,frozen=True)
    _write(output/'selected_ils.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=Path('rc3_rl_g2/outputs/protocol.json'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--select-ils',action='store_true')
    parser.add_argument('--baselines',action='store_true')
    parser.add_argument('--split',choices=('rule_dev','validation'),default='validation')
    parser.add_argument('--time-limit-s',type=float)
    parser.add_argument('--ils-selection',type=Path,default=Path('rc3_rl_g2/outputs/rule_development/selected_ils.json'))
    args=parser.parse_args()
    if args.select_ils:
        print(json.dumps(select_ils(args.protocol,args.output)));return
    if not args.baselines:parser.error('Use --baselines, or supply models through evaluate_methods')
    protocol=json.loads(args.protocol.read_text(encoding='utf-8'))
    selection=json.loads(args.ils_selection.read_text(encoding='utf-8'))
    methods={name:None for name in ('original_fixed3','original_detour','environment_stop','detour_memory','ils','random')}
    config=SearchConfig(**protocol.get('config',asdict(SearchConfig())))
    evaluate_methods(protocol['splits'][args.split],methods,args.output,config=config,
                     time_limit_s=args.time_limit_s,policy_kwargs={'ils':selection['selected']})


if __name__=='__main__':main()
