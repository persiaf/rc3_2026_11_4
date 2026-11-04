"""Gated G2.1 evaluation adapted from frozen G2, with isolated prepared inputs.

No CLI action constructs an environment until the final offline opportunity
receipt has been verified.  Cold runs include prepare/cache-write/plan time;
warm-input runs are explicitly labelled and never called cold planning latency.
"""
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
from .baselines import DETOUR_EARLY_GRID

from rc3_fast.scenes import make_complex_scene
from rc3_fast.solver import prepare
from rc3_v4.runtime import run as frozen_run
from rc3_rl_research.types import SearchConfig
from rc3_rl_g2.env_g2 import SearchEnvG2
from .baselines import make_policy
from rc3_rl_g2.graph_observation import build_policy_observation
from rc3_rl_g2.collector_contract import merge_terminal_events

SCHEMA='rc3-g21-gated-evaluation-v1'
ROOT=Path(__file__).resolve().parent


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _inside(path, root):
    path=Path(path).resolve();root=Path(root).resolve()
    if path==root or root not in path.parents:
        raise ValueError('path must be strictly within '+str(root))
    return path


def require_opportunity_gate(protocol_path=None,gate_path=None):
    """Read-only gate check; no optimizer, preparation, model load or env calls."""
    protocol_path=ROOT/'outputs/protocol.json' if protocol_path is None else Path(protocol_path)
    gate_path=ROOT/'outputs/opportunity_gate.json' if gate_path is None else Path(gate_path)
    protocol=_read(protocol_path);digest=_sha(protocol_path)
    if protocol.get('schema')!='rc3-g21-offline-gated-v1':raise ValueError('wrong G21 protocol')
    if digest!=protocol_path.with_suffix('.sha256').read_text().strip():raise ValueError('protocol changed')
    toy=_read(protocol_path.parent/'toy16/receipt.json')
    if (toy.get('completed') is not True or toy.get('updates')!=protocol['toy']['updates']
            or toy.get('protocol_sha256')!=digest or toy.get('gate',{}).get('passed') is not True
            or toy['gate'].get('composition_valid') is not True
            or toy['gate'].get('evidence_near_best',0)<15 or toy['gate'].get('positive_continue',0)<7):
        raise ValueError('final toy gate not passed')
    gate=_read(gate_path)
    if gate.get('passed') is not True or gate.get('protocol_sha256')!=digest:
        raise ValueError('root has not authorized the final opportunity gate')
    checkpoint=_inside(gate['checkpoint_path'],protocol_path.parent)
    if checkpoint.parent.name=='toy16':raise ValueError('memorized toy checkpoint cannot authorize deployment')
    receipt_path=checkpoint.parent/'receipt.json';receipt=_read(receipt_path)
    if (receipt.get('completed') is not True or receipt.get('stage') not in ('sampling','gain')
            or receipt.get('updates')!=protocol[receipt['stage']]['updates']
            or receipt.get('protocol_sha256')!=digest or _sha(checkpoint)!=gate['checkpoint_sha256']
            or receipt.get('checkpoint_sha256')!=gate['checkpoint_sha256']):
        raise ValueError('opportunity checkpoint is not the final prescribed checkpoint')
    if gate.get('receipt_sha256') is not None and gate['receipt_sha256']!=_sha(receipt_path):
        raise ValueError('opportunity training receipt changed')
    extra_path=checkpoint.parent/'extra_existing_teacher_diagnostic.json'
    extra=_read(extra_path)
    if gate.get('extra_diagnostic_sha256') is not None and gate['extra_diagnostic_sha256']!=_sha(extra_path):
        raise ValueError('extra teacher diagnostic changed')
    initial=receipt['final_diagnostic'];limits=protocol['opportunity_gate']
    final_diagnostic_path=checkpoint.parent/f"diagnostic_{receipt['updates']:04d}.json"
    if _read(final_diagnostic_path)['groups']!=initial:raise ValueError('final diagnostic and receipt disagree')
    diagnostic_hash=gate.get('diagnostic_sha256',gate.get('final_diagnostic_sha256'))
    if (gate.get('diagnostic_sha256') is not None and gate.get('final_diagnostic_sha256') is not None
            and gate['diagnostic_sha256']!=gate['final_diagnostic_sha256']):
        raise ValueError('diagnostic hash aliases disagree')
    if diagnostic_hash is not None and diagnostic_hash!=_sha(final_diagnostic_path):
        raise ValueError('final teacher diagnostic changed')
    def mean(groups,group,metric):
        item=groups[group]['metrics'][metric]
        value=item['mean']
        if (type(item['n']) is not int or item['n']<1 or isinstance(value,bool)
                or not isinstance(value,(float,int)) or not math.isfinite(value) or not 0<=value<=1):
            raise ValueError('invalid or absent opportunity evidence')
        return value
    metrics=dict(
        initial_positive=mean(initial,'positive_roots','full_known_positive_continue'),
        extra_positive=mean(extra['groups'],'positive_roots','full_known_positive_continue'),
        initial_clear_stop=mean(initial,'clear_stop_roots','full_greedy_stop'),
        initial_unknown=mean(initial,'roots','full_greedy_unknown'))
    if (metrics['initial_positive']<limits['initial_positive_full_known_min_fraction']
            or metrics['extra_positive']<limits['extra_positive_full_known_min_fraction']
            or metrics['initial_clear_stop']<limits['initial_clear_stop_full_stop_min_fraction']
            or metrics['initial_unknown']>limits['initial_root_full_unknown_max_fraction']):
        raise ValueError('full-action opportunity thresholds failed')
    return dict(protocol=protocol,protocol_sha256=digest,gate=gate,gate_sha256=_sha(gate_path),
                metrics=metrics,checkpoint_path=str(checkpoint))


def _prepare_input(case,cache_dir,cache_mode):
    from rc3_rl_g2.prepared_cache import prepare_case
    directory=_inside(cache_dir,ROOT/'outputs')
    if cache_mode=='cold' and directory.exists() and any(directory.glob('*.rc3prep')):
        raise ValueError('cold evaluation cannot reuse an already prepared input')
    adapter,ledger=prepare_case(case,directory)
    ledger['evaluation_cache_mode']=cache_mode
    if cache_mode=='cold' and ledger['cache_hit']:raise ValueError('unexpected cold-cache hit')
    return adapter,ledger


def _one_step_stop(observation,proposal_calls,enabled):
    if not enabled or proposal_calls<1:return None
    return next(i for i,(action,allowed) in enumerate(zip(observation['actions'],observation['mask']))
                if allowed and action['kind']=='stop')


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


def _run_original(scene,name,config,time_limit_s,case,cache_dir,cache_mode):
    """Same frozen solver, with its time budget reduced by actual preparation."""
    started=time.perf_counter()
    # Preserve the frozen solver's single-anchor boundary. The prepared adapter
    # contains a certified anchor already, which would otherwise be paid twice.
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
    row.update(preparation=None,cache_mode='frozen_solver_uncached',
        timing_scope='frozen cold problem preparation and planning, without prepared-cache serialization')
    evidence=dict(schema=SCHEMA,scene=scene,plan=plan,diagnostics=diagnostics,row=row,
                  wrapper=dict(frozen_solver=True,preparation_deducted_from_deadline=True,
                               runtime_time_limit_s=remaining,original_sources_modified=False))
    return row,evidence


def _run_shared(scene,name,model,config,time_limit_s,policy_kwargs,seed,case,cache_dir,cache_mode,one_step=False):
    started=time.perf_counter();deadline=math.inf if time_limit_s is None else started+time_limit_s
    adapter,preparation=_prepare_input(case,cache_dir,cache_mode)
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
            inferred=False
            forced_stop=_one_step_stop(observation,env.counters['proposal_calls'],model is not None and one_step)
            if forced_stop is not None:
                action=forced_stop
            elif model is not None:
                tick=time.perf_counter();raw=build_policy_observation(env);graph_s+=time.perf_counter()-tick
                if time.perf_counter()>=deadline:
                    env.terminate('external_deadline');break
                tick=time.perf_counter();actions,logp,value=model.act([raw],deterministic=True)
                action=int(actions[0].cpu());inference_s+=time.perf_counter()-tick
                policy_samples.append(dict(observation=raw,action_id=action,
                    old_logp=float(logp[0].cpu()),old_value=float(value[0].cpu()),executed=False))
                inferred=True
            else:
                tick=time.perf_counter();action=int(selector.select(observation,env.rng));selection_s+=time.perf_counter()-tick
            selections.append(dict(action_id=action,action=observation['actions'][action],
                rng_before=rng_before,rng_after=copy.deepcopy(env.rng.bit_generator.state),
                transition_index=len(env.transitions),executed=False))
            if time.perf_counter()>=deadline:
                env.terminate('external_deadline');break
            next_obs,reward,terminal,truncated,info=env.step(action)
            selections[-1]['executed']=True
            if inferred:policy_samples[-1]['executed']=True
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
    row.update(preparation=preparation,cache_mode=cache_mode,one_step_model=one_step,
        timing_scope='cold scene preparation, cache writing and planning' if cache_mode=='cold' else 'prepared-input load and planning; not cold scene-to-plan latency')
    evidence=dict(schema=SCHEMA,scene=scene,adapter=adapter.payload(),initial_state=initial_state,
        state=env.state_dict(),plan=certificate.plan,row=row,policy_samples=policy_samples,selections=selections,
        actor_records=merged,consumed_terminal_event_ids=sorted(consumed),
        policy_initial_state=initial_policy,policy_state=None if selector is None else selector.state_dict())
    return row,evidence


def evaluate_methods(cases,methods,output_dir,*,config=SearchConfig(),time_limit_s=None,policy_kwargs=None,
                     protocol_path=None,gate_path=None,cache_dir=None,cache_mode='cold',one_step_models=()):
    """Evaluate preloaded models and specified rules on identical fresh inputs.

    Learned models are copied to CPU once, outside planning timing. Disk model
    cold-loading is not measured by this object-taking API and is recorded as
    None, rather than misleadingly reported as zero. Every scene/method obtains
    a fresh adapter; per-case method order rotates to reduce fixed-order bias.
    """
    authorization=require_opportunity_gate(protocol_path,gate_path)
    if cache_mode not in ('cold','warm'):raise ValueError('cache_mode must be cold or warm')
    if any(name not in methods or methods[name] is None for name in one_step_models):raise ValueError('one-step model name is absent or not a model')
    if time_limit_s is not None and (isinstance(time_limit_s,bool) or not math.isfinite(time_limit_s) or time_limit_s<0):
        raise ValueError('invalid complete soft time limit')
    if not cases or not methods:raise ValueError('nonempty cases and methods are required')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+',name) for name in methods):raise ValueError('unsafe method name')
    output=_inside(output_dir,ROOT/'outputs');output.mkdir(parents=True,exist_ok=True)
    cache_root=_inside(ROOT/'outputs/prepared_cache' if cache_dir is None else cache_dir,ROOT/'outputs')
    run_key=hashlib.sha256(str(output).encode()).hexdigest()[:16]
    if (output/'results.json').exists():raise ValueError('completed evaluation exists; use a fresh output directory')
    policy_kwargs={} if policy_kwargs is None else copy.deepcopy(policy_kwargs)
    evaluators={};setup={}
    if any(model is not None for model in methods.values()):
        import torch
        torch.set_num_threads(1)
    for name,model in methods.items():
        if model is None:
            if name not in ('original_fixed3','original_detour','stop','environment_stop','detour_memory','ils','random','detour_early','one_step'):
                raise ValueError('unknown nonlearning method '+name)
            evaluators[name]=None;setup[name]=dict(model=None,cold_load_s=None)
        else:
            tick=time.perf_counter();evaluators[name]=copy.deepcopy(model).cpu().eval()
            setup[name]=dict(model=_model_identity(evaluators[name]),cpu_copy_and_setup_s=time.perf_counter()-tick,
                             cold_load_s=None,cold_load_scope='Caller supplied an already loaded model; measure disk loading separately.')
    request=dict(schema=SCHEMA,cases=cases,config=asdict(config),time_limit_s=time_limit_s,
                 policy_kwargs=policy_kwargs,method_setup=setup,method_order='rotated by case list position',
                 evidence_scene_generation_timed=False,actual_cold_scene_generation_timed=cache_mode=='cold',serialization_timed=False,
                 g21_protocol_sha256=authorization['protocol_sha256'],opportunity_gate_sha256=authorization['gate_sha256'],
                 cache_mode=cache_mode,cache_dir=str(cache_root),one_step_models=list(one_step_models),
                 cold_timing_includes_cache_write=cache_mode=='cold')
    _write(output/'request.json',request)
    rows=[];names=list(methods)
    for case_index,case in enumerate(cases):
        scene=_scene(case)
        if cache_mode=='warm':
            # Explicit untimed preparation; every method starts from the same safe cache.
            _,ledger=_prepare_input(case,cache_root/run_key/'warm_inputs','warm')
            _write(output/f'prepared_{case_index:03d}.json',ledger)
        order=names[case_index%len(names):]+names[:case_index%len(names)]
        for method in order:
            destination=output/f"case_{case_index:03d}_{method}"
            destination.mkdir(parents=True,exist_ok=True)
            _write(destination/'scene.json',scene)
            try:
                cache=cache_root/run_key/(f'case_{case_index:03d}_{method}' if cache_mode=='cold' else 'warm_inputs')
                if method.startswith('original_'):
                    row,evidence=_run_original(scene,method,config,time_limit_s,case,cache,cache_mode)
                else:
                    row,evidence=_run_shared(scene,method,evaluators[method],config,time_limit_s,
                        policy_kwargs.get(method,policy_kwargs.get('ils',{}) if method=='ils' else {}),int(case.get('seed',0))+991,
                        case,cache,cache_mode,method in one_step_models)
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


def _winner(scores):
    """Predeclared tolerance, paid-work, cap and patience tie-break order."""
    best=max(row['mean_return'] for row in scores)
    tied=[row for row in scores if best-row['mean_return']<=1e-12]
    return min(tied,key=lambda row:(row['mean_work'],row['settings']['max_proposals'],row['settings']['patience']))


def select_early(output_dir,*,protocol_path=None,gate_path=None,cache_dir=None):
    authorization=require_opportunity_gate(protocol_path,gate_path)
    output=_inside(output_dir,ROOT/'outputs')
    if output.exists() and any(output.iterdir()):raise ValueError('use a fresh rule-development directory')
    old_protocol_path=ROOT.parent/'rc3_rl_g2/outputs/protocol.json'
    old=_read(old_protocol_path);cases=old['splits']['rule_dev']
    if len(cases)!=12 or any(case['split']!='rule_dev' for case in cases):raise ValueError('expected existing rule_dev12 split')
    config=SearchConfig(**old['config'])
    spec=dict(schema=SCHEMA,phase='rule_development_only',cases=cases,grid=DETOUR_EARLY_GRID,
        g21_protocol_sha256=authorization['protocol_sha256'],opportunity_gate_sha256=authorization['gate_sha256'],
        g2_protocol_sha256=_sha(old_protocol_path),config=asdict(config),cache_mode='cold',
        criterion='highest mean return; ties within 1e-12: lowest mean paid work, smallest proposal cap, smallest patience',
        validation_used=False,source_sha256={name:_sha(ROOT/name) for name in ('baselines.py','evaluate_g21.py')})
    _write(output/'selection_protocol.json',spec);scores=[]
    for index,setting in enumerate(DETOUR_EARLY_GRID):
        rows=evaluate_methods(cases,{'detour_early':None},output/f'grid_{index}',config=config,
            policy_kwargs={'detour_early':setting},protocol_path=protocol_path,gate_path=gate_path,cache_dir=cache_dir)
        if len(rows)!=12 or any(row['status']!='complete' or not row['valid'] for row in rows):
            raise ValueError('incomplete rule-development evidence; no configuration frozen')
        scores.append(dict(grid_index=index,settings=setting,mean_return=float(np.mean([row['total_reward'] for row in rows])),
            mean_work=float(np.mean([row['work_used'] for row in rows])),scenes=12))
    selected=_winner(scores)
    result=dict(schema=SCHEMA,frozen=True,validation_used=False,selected=selected['settings'],scores=scores,
        g21_protocol_sha256=authorization['protocol_sha256'],opportunity_gate_sha256=authorization['gate_sha256'],
        selection_protocol_sha256=_sha(output/'selection_protocol.json'),scenes=12,runs=48,
        committed_actions=sum(row['steps'] for i in range(4) for row in _read(output/f'grid_{i}/results.json')),
        proposal_queries=sum(row['proposal_calls'] for i in range(4) for row in _read(output/f'grid_{i}/results.json')),
        query_count_scope='48 scene episodes; committed actions and proposal queries counted separately')
    _write(output/'selected_detour_early.json',result)
    return result


def load_evaluation_model(path,authorization,*,initial_bc=False):
    """Trusted local checkpoint loader; initial and G21 identity checked separately."""
    from rc3_rl_g2.checkpoints import load_model
    path=Path(path)
    if initial_bc:
        if _sha(path)!=authorization['protocol']['initial_bc_sha256']:
            raise ValueError('initial BC identity changed')
    else:
        _inside(path,ROOT/'outputs')
        if path.parent.name=='toy16':raise ValueError('toy memorization model is not a deployment checkpoint')
    started=time.perf_counter();model,payload=load_model(path,'cpu');load_s=time.perf_counter()-started
    if not initial_bc:
        if payload.get('protocol_sha256')!=authorization['protocol_sha256']:
            raise ValueError('G21 checkpoint protocol identity changed')
        if (_sha(path)!=authorization['gate']['checkpoint_sha256']
                and payload.get('parent_bc_sha256')!=authorization['gate']['checkpoint_sha256']):
            raise ValueError('model is neither the gate-approved BC nor its PPO child')
    return model.eval(),dict(path=str(path.resolve()),sha256=_sha(path),cold_disk_load_s=load_s,
                             scope='model disk load measured outside scene planning')


def run_comparison(output_dir,selection_path,*,checkpoint_path=None,protocol_path=None,gate_path=None,
                   cache_dir=None,cache_mode='cold',time_limit_s=None):
    authorization=require_opportunity_gate(protocol_path,gate_path)
    selected=_read(selection_path)
    if (selected.get('frozen') is not True or selected.get('validation_used') is not False
            or selected.get('g21_protocol_sha256')!=authorization['protocol_sha256']
            or selected.get('opportunity_gate_sha256')!=authorization['gate_sha256']):
        raise ValueError('early-stop configuration has not been frozen under the current gate')
    selection_spec_path=Path(selection_path).parent/'selection_protocol.json'
    selection_spec=_read(selection_spec_path)
    if (selected.get('selection_protocol_sha256')!=_sha(selection_spec_path)
            or selected.get('selected') not in DETOUR_EARLY_GRID
            or selected.get('selected')!=_winner(selected['scores'])['settings']
            or selection_spec.get('source_sha256')!={name:_sha(ROOT/name) for name in ('baselines.py','evaluate_g21.py')}):
        raise ValueError('frozen rule definition, selection evidence or implementation changed')
    old=_read(ROOT.parent/'rc3_rl_g2/outputs/protocol.json')
    initial,initial_load=load_evaluation_model(ROOT.parent/'rc3_rl_g2/outputs/training/bc_initial.pt',authorization,initial_bc=True)
    model,model_load=load_evaluation_model(authorization['checkpoint_path'] if checkpoint_path is None else checkpoint_path,authorization)
    methods={'original_fixed3':None,'original_detour':None,'stop':None,'detour_early':None,
             'initial_bc':initial,'g21_model':model,'g21_one_step':model}
    rows=evaluate_methods(old['splits']['validation'],methods,output_dir,config=SearchConfig(**old['config']),
        time_limit_s=time_limit_s,policy_kwargs={'detour_early':selected['selected']},protocol_path=protocol_path,
        gate_path=gate_path,cache_dir=cache_dir,cache_mode=cache_mode,one_step_models=('g21_one_step',))
    _write(Path(output_dir)/'checkpoint_loads.json',{'initial_bc':initial_load,'g21_model':model_load})
    from rc3_rl_g2.tools.evaluate_all import paired_statistics,summarize_methods
    pairs=[paired_statistics(rows,method,rows,'stop') for method in ('detour_early','initial_bc','g21_model','g21_one_step')]
    pairs.extend(paired_statistics(rows,'g21_model',rows,other) for other in ('detour_early','initial_bc','g21_one_step'))
    _write(Path(output_dir)/'summary.json',dict(schema=SCHEMA,methods=summarize_methods(rows),pairs=pairs,
        validation_scope='Existing G2 validation scenes, reused as development comparison; not independent confirmation.',
        timing_scope='cold preparation/cache-write/planning' if cache_mode=='cold' else 'warm prepared-input load/planning',
        primary_metrics=['stop_at_root','total_reward','J','work_used','complete_s'],
        original_fixed3_detour_return_scope='No shared-action work reward is assigned to original frozen solvers.'))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,default=ROOT/'outputs/protocol.json')
    parser.add_argument('--gate',type=Path,default=ROOT/'outputs/opportunity_gate.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache-dir',type=Path,default=ROOT/'outputs/prepared_cache')
    parser.add_argument('--cache-mode',choices=('cold','warm'),default='cold')
    parser.add_argument('--select-early',action='store_true')
    parser.add_argument('--compare',action='store_true')
    parser.add_argument('--selection',type=Path,default=ROOT/'outputs/rule_development/selected_detour_early.json')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--time-limit-s',type=float)
    args=parser.parse_args()
    if args.select_early==args.compare:parser.error('choose exactly one of --select-early and --compare')
    # Both called APIs re-check authorization immediately before mutable work.
    if args.select_early:
        if args.cache_mode!='cold' or args.time_limit_s is not None:
            parser.error('registered rule development uses cold fixed-work evaluation')
        print(json.dumps(select_early(args.output,protocol_path=args.protocol,gate_path=args.gate,cache_dir=args.cache_dir)))
    else:
        run_comparison(args.output,args.selection,checkpoint_path=args.checkpoint,protocol_path=args.protocol,
            gate_path=args.gate,cache_dir=args.cache_dir,cache_mode=args.cache_mode,time_limit_s=args.time_limit_s)


if __name__=='__main__':main()

