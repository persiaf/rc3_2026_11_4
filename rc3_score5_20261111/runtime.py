import math
import time
from dataclasses import dataclass
import torch
from rc3_global_local.environment import Environment
from rc3_global_local.terminal import finish,export
from rc3_global_local.validation import validate
from rc3_conservative_refinement.costs import remaining_cost_targets
from rc3_conservative_refinement import runtime as old
from .common import HOME,ROLES,read
from .actor import load_export

@dataclass
class State:
    problem:object
    remaining:object
    current:int
    prefix:list
    raw_L:float
    done:bool=False
    def mask(self):return Environment.mask(self)

def capture(env):return State(env.problem,env.remaining.copy(),env.current,list(env.prefix),env.raw_L)

def replay_logits(actor,episode,ledger):
    encoded=actor.encode(episode['problem']);ledger.counts['cache_rebuilds']+=1
    return [actor.score(s,encoded) for s in episode['states']]

def request(ledger,inputs,spec,rid,role,*,actor=None,entry=None,greedy=False,seed=0,op=0,kind='engineering',case=None):
    def compute():
        tick=time.perf_counter();problem,evidence=inputs.view(spec);prepare=time.perf_counter()-tick;tick=time.perf_counter()
        if entry is not None:policy=load_export(entry,role,ledger.stage)
        elif role=='FrozenSL':policy=old.load_frozen_sl()
        else:policy=actor
        load_s=time.perf_counter()-tick if entry is not None or role=='FrozenSL' else 0.
        if role in ('Fixed3','Detour16'):
            from rc3_global_local_impl_review.endpoint_baselines import run_endpoint
            stats={};plan,info=run_endpoint(problem,'fixed3' if role=='Fixed3' else 'detour16',time_limit_s=max(.01,ledger.cap['wall_s']-ledger.elapsed()-15),reject_fallback=True,counters=stats)
            route=info['local_route'];m=problem.metrics(route)
            record=dict(spec=spec,physical_key=evidence['physical_key'],scope_evidence=evidence,plan=plan,raw_plan=plan,route=route,raw_route=route,
                metrics=dict(m,raw_J=m['J'],raw_L=m['L'],raw_K=m['K']),counts=stats,internal_search=info['internal_search'],
                raw_semantics='Native final pipeline, not pre-H; no extra common H')
            episode=None
        else:
            if role in ROLES and (policy is None or policy.role!=role):raise ValueError('Actor role')
            from rc3_nav_value.r1.oracle import _instrument,_empty_internal
            before=old.tensor_digest(policy.state_dict());stats=dict(full_H_calls=0);internal=_empty_internal()
            def terminal(p,r):
                with _instrument(internal):return finish(p,r,stats)
            env=Environment(problem,terminal);states=[];actions=[];logits=[]
            gen=torch.Generator().manual_seed(seed)
            with torch.no_grad():
                encoded=policy.encode(problem) if not env.done else None;ledger.counts['cache_rebuilds']+=int(encoded is not None)
                while not env.done:
                    ledger.check();z=policy.score(env,encoded);mask=torch.tensor(env.mask())
                    if not torch.isfinite(z[mask]).all() or not torch.isneginf(z[~mask]).all():raise ValueError('Logit/mask mismatch')
                    a=int(z.argmax()) if greedy else int(torch.multinomial(torch.softmax(z,-1),1,generator=gen))
                    states.append(capture(env));actions.append(a);logits.append(z.detach().clone());env.step(a+1)
            if old.tensor_digest(policy.state_dict())!=before:raise ValueError('Collection changed Actor weights')
            result=env.result;raw=[0,*env.prefix,0];raw_plan=export(problem,raw,role+'_raw');cert=validate(problem,raw_plan)
            if not cert['valid_model_plan']:raise ValueError('Invalid raw complete plan')
            raw_plan['validation']=cert;result['plan']['method']=role
            targets=remaining_cost_targets(raw_plan['route_leg_costs_m'],result['J'],len(problem.C),problem.penalty)
            if abs(math.fsum(env.rewards)+result['J'])>1e-7:raise ValueError('Complete terminal return mismatch')
            if any(abs(math.fsum(env.rewards[t:])/max(len(problem.C),1)-G)>1e-7 for t,G in enumerate(targets['returns'])):raise ValueError('Truncated reward-to-go')
            record=dict(spec=spec,physical_key=evidence['physical_key'],scope_evidence=evidence,plan=result['plan'],raw_plan=raw_plan,
                route=result['route'],raw_route=raw,metrics={k:result[k] for k in ('J','L','K','raw_J','raw_L','raw_K')},counts=stats,
                rewards=env.rewards,complete_returns=targets,actor_tensor_sha256=before,H_s=result['H_s'],internal_search=internal,raw_semantics='Actor complete raw before unchanged common H')
            episode=dict(problem=problem,states=states,actions=actions,old_logits=logits,G=torch.tensor(targets['returns'],dtype=torch.float32),
                role=role,stage=ledger.stage,episode_id=rid,actor_tensor_sha=before,task_count=len(problem.C))
        record.update(timing=dict(input_prepare_s=prepare,model_load_s=load_s),timer_scope='Input/source/nav restoration,model,original features/score,complete construction,H,raw/final export+validation; ledger adds archive/index/journal')
        if case:record.update(case_id=case['id'],cohort=case['group'])
        record=old._json_value(record)
        return dict(record=record,episode=episode)
    result=ledger.request(rid,compute,role=role,op=op,kind=kind)
    return result

def teacher_episode(ledger,inputs,actor,index,rid):
    ledger.check();ledger.counts['teacher_replays']+=1;ledger.counts['teacher_label_replays']+=1
    problem,actions,evidence=inputs.teacher(index);env=Environment(problem,lambda p,r:p.metrics(r));states=[]
    for node in actions:
        ledger.check();states.append(capture(env));env.step(node)
    if not env.done:raise ValueError('Incomplete saved teacher')
    return dict(problem=problem,states=states,actions=[a-1 for a in actions],role=actor.role,stage=ledger.stage,episode_id=rid,
        actor_tensor_sha=None,G=None,task_count=len(problem.C),teacher_source=evidence['teacher_source'])
