"""D1 frozen actors and native endpoints; every plan validated by old kernel."""
import time
import torch
import numpy as np
from .common import *
from .model import Actor,ARCH
from rc3_global_local.policy import GlobalLocalPolicy
from rc3_global_local.environment import Environment
from rc3_global_local.terminal import finish,export
from rc3_global_local.validation import validate
from rc3_global_local.native import run_endpoint
from rc3_nav_value.r1.oracle import _instrument,_empty_internal


def export_actor(actor,seed,role,update,source):
    state=actor.base.state_dict() if role=='SL256' else actor.state_dict()
    values={k:v.detach().cpu().clone() for k,v in state.items()};tensor=tensor_digest(values)
    p=OUT/'actors'/str(seed)/f'{role}_{update:04}_{tensor[:12]}.pt'
    payload=dict(schema='integrated-D1-actor-v1',architecture='base' if role=='SL256' else ARCH,model=values,
        base_tensor_sha=actor.base_sha,actor_tensor_sha=tensor,seed=seed,checkpoint_role=role,run_id=EXP,update=update,training_source=source)
    if not p.exists():atomic(p,payload)
    entry=dict(path=label(p),actor_sha=digest(p),tensor_sha=tensor,base_tensor_sha=actor.base_sha,
        seed=seed,study_method=role,checkpoint_role=role,run_id=EXP,update=update,architecture=payload['architecture'],training_source=source)
    sp=p.with_suffix('.pt.json')
    if not sp.exists():write_new(sp,entry)
    else:require(read(sp)==entry,'Actor metadata/source conflict')
    return entry


def load_actor(entry):
    require(entry['study_method']==entry['checkpoint_role'] and entry['run_id']==EXP,'Actor role mismatch')
    require(digest(ROOT/entry['path'])==entry['actor_sha'],'Actor source file changed')
    p=torch.load(ROOT/entry['path'],map_location='cpu',weights_only=True)
    require(p['checkpoint_role']==entry['checkpoint_role'] and p['seed']==entry['seed'] and p['update']==entry['update'],'Actor endpoint/source metadata mismatch')
    require(tensor_digest(p['model'])==entry['tensor_sha'],'Actor tensors changed')
    if p['architecture']=='base':
        with torch.device('meta'):model=GlobalLocalPolicy()
        model.load_state_dict(p['model'],assign=True);model.eval().requires_grad_(False)
    else:
        require(p['architecture']==ARCH,'Actor architecture mismatch')
        base={k[5:]:v for k,v in p['model'].items() if k.startswith('base.')};model=Actor(base,entry['seed']);model.load_state_dict(p['model']);model.eval().requires_grad_(False)
    require(tensor_digest(model.state_dict())==entry['tensor_sha'],'Loaded actor identity mismatch')
    return model


def plan(q,case,inputs,entry,clock):
    clock.check();t=time.perf_counter();components={};inputs.parents.clear()
    if q.get('cache')=='cold':inputs.nav.clear()
    p,evidence=inputs.view(case['spec']);components['shared_prepare']=time.perf_counter()-t
    nav_calls=0
    if q.get('cache')=='cold':
        from rc3_baseline.baseline import navigation
        tick=time.perf_counter();fresh=navigation(p.parent.scene);nav_calls=1
        require(np.array_equal(fresh['costs'],p.parent.D),'Cold navigation changed physical D')
        require(np.array_equal(fresh['predecessors'],p.parent.base.nav['predecessors']),'Cold navigation changed paths')
        components['cold_navigation']=time.perf_counter()-tick
    role=q['role'];internal=_empty_internal();stats=dict(full_H_calls=0,export_attempts=0,exports=0,validation_attempts=0,validation_completed=0)
    if role in ('Fixed3','Detour16'):
        tick=time.perf_counter();pi,info=run_endpoint(p,'fixed3' if role=='Fixed3' else 'detour16',time_limit_s=clock.remaining(),reject_fallback=True,counters=stats)
        components['native']=time.perf_counter()-tick;raw=route=list(info['local_route']);m=p.metrics(route);internal=info['internal_search']
        result=dict(m,raw_J=m['J'],raw_L=m['L'],raw_K=m['K'],plan=pi,route=route,raw_route=raw)
        require(not info['common_H_added'],'Native received extra H');model_id=None;trace=[]
    else:
        require(entry is not None and entry['seed']==q['seed'] and entry['checkpoint_role']==('RL' if role in ('RL','RL_raw') else role),'Request/actor role mismatch')
        tick=time.perf_counter();model=load_actor(entry);components['model_D1_load']=time.perf_counter()-tick
        before=tensor_digest(model.state_dict());env=Environment(p,lambda p,r:p.metrics(r));tick=time.perf_counter()
        with torch.inference_mode():
            encoded=model.encode(p) if not env.done else None
            while not env.done:
                clock.check();z=model.score(env,encoded);mask=env.mask();require(bool(torch.isfinite(z[torch.tensor(mask)]).all()),'Nonfinite greedy logits')
                node=int(z.argmax())+1;env.step(node)
        components['dynamic_features_construct']=time.perf_counter()-tick;raw=[0,*env.prefix,0];trace=env.trace
        if role=='RL_raw':
            tick=time.perf_counter();pi=export(p,raw,role);components['export']=time.perf_counter()-tick;tick=time.perf_counter();certificate=validate(p,pi)
            require(certificate['valid_model_plan'],'Raw actor plan infeasible');pi['validation']=certificate;components['validate']=time.perf_counter()-tick
            m=p.metrics(raw);result=dict(m,raw_J=m['J'],raw_L=m['L'],raw_K=m['K'],route=raw,plan=pi)
        else:
            tick=time.perf_counter()
            with _instrument(internal):result=finish(p,raw,stats)
            components.update(H=result['H_s'],export=result['export_s'],validate=result['validation_s'],full_H_pipeline=time.perf_counter()-tick)
        route=result['route'];require(before==tensor_digest(model.state_dict())==entry['tensor_sha'],'Inference updated parameters')
        require(all(v.grad is None and not v.requires_grad for v in model.parameters()),'Frozen actor has gradients');model_id=entry
    for k in ('J','L','K','raw_J','raw_L','raw_K'):require(np.isfinite(result[k]),'Nonfinite route result')
    result['plan']['method']=role
    result['plan']['model_identity']=model_id
    return dict(case_id=case['id'],study_method=role,checkpoint_role=None if model_id is None else model_id['checkpoint_role'],
        actor_sha=None if model_id is None else model_id['actor_sha'],tensor_sha=None if model_id is None else model_id['tensor_sha'],
        run_id=EXP,update=None if model_id is None else model_id['update'],seed=q.get('seed'),actor_file=None if model_id is None else model_id['path'],
        model_identity=model_id,physical_key=inputs.index[f"{case['spec']['map_seed']}_{case['spec']['task_seed']}_A{case['spec']['A_condition']}"]['physical_key'],
        spec=case['spec'],input_identity=p.identity,scope_evidence=evidence,problem=p.description(),raw_route=raw,route=route,trace=trace,
        metrics={k:result[k] for k in ('J','L','K','raw_J','raw_L','raw_K')},plan=result['plan'],valid=result['plan']['validation']['valid_model_plan'],
        counters=stats,internal_search=internal,navigation_solves=nav_calls,components_s=components,service_time_before_IO_s=time.perf_counter()-t)


def evaluate_set(cases,roles,kind,stage,round_,store,inputs,clock,*,order_seed=None,cache=None,repeat=0):
    rows=[];rng=np.random.Generator(np.random.PCG64(order_seed or 0))
    for case in cases:
        ordering=list(range(len(roles))) if order_seed is None else rng.permutation(len(roles)).tolist()
        if cache=='hot':
            # Navigation only; this preparation is outside request service and inside stage wall.
            t=time.perf_counter();inputs.view(case['spec']);store.log(dict(event='hot_prime',case=case['id'],elapsed_s=time.perf_counter()-t))
        for index in ordering:
            seed,role,entry=roles[index];q=dict(kind=kind,stage=stage,round=round_,case=case['id'],seed=seed,role=role,cache=cache,repeat=repeat)
            if role not in ('Fixed3','Detour16'):
                require(entry is not None and entry['seed']==seed and entry['run_id']==EXP and entry['checkpoint_role']==('RL' if role in ('RL','RL_raw') else role),'Wrong actor role/source before request')
            rows.append(store.request(q,lambda q=q,case=case,entry=entry:plan(q,case,inputs,entry,clock),clock))
    return rows


def gains(rows,control,seed=None,by_size=False):
    grouped={}
    for r in rows:
        if seed is not None and r['seed'] not in (seed,None):continue
        if r['study_method'] not in ('RL',control):continue
        key=(r['physical_key'],tuple(r['spec'][k] for k in ('width','height')) if by_size else None)
        grouped.setdefault(key,{}).setdefault(r['study_method'],[]).append(r['metrics']['J'])
    gs=[(np.mean(v[control])-np.mean(v['RL']))/np.mean(v[control]) for v in grouped.values() if 'RL' in v and control in v]
    require(gs,'Unpaired parent contrast');return float(np.mean(gs))
