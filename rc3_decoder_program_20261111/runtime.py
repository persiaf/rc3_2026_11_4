import math
import time
from collections import Counter
import torch
from rc3_global_local.environment import Environment
from rc3_global_local.terminal import finish,export
from rc3_global_local.validation import validate
from rc3_score5_20261111.runtime import State,capture
from rc3_conservative_refinement import runtime as source
from .actor import load_export
from .common import ROLES

def request(ledger,inputs,spec,rid,role,*,actor=None,entry=None,greedy=False,seed=0):
    def compute():
        tick=time.perf_counter();problem,evidence=inputs.view(spec);prepare=time.perf_counter()-tick
        policy=load_export(entry) if entry else source.load_frozen_sl() if role=='FrozenSL' else actor
        if policy is None or (role in ROLES and policy.role!=role):raise ValueError('Exact Actor role required')
        from rc3_nav_value.r1.oracle import _instrument,_empty_internal
        internal=_empty_internal();stats=Counter();states=[];actions=[];logits=[]
        before=source.tensor_digest(policy.state_dict());forward_s=0.;forward_calls=0
        def terminal(p,r):
            with _instrument(internal):return finish(p,r,stats)
        try:
            env=Environment(problem,terminal);gen=torch.Generator().manual_seed(seed)
            with torch.no_grad():
                tick=time.perf_counter();cache=policy.encode(problem) if not env.done else None
                forward_s+=time.perf_counter()-tick;forward_calls+=int(cache is not None);ledger.counts['cache_rebuilds']+=int(cache is not None)
                while not env.done:
                    ledger.check();tick=time.perf_counter();z=policy.score(env,cache);forward_s+=time.perf_counter()-tick;forward_calls+=1
                    legal=torch.tensor(env.mask())
                    if not torch.isfinite(z[legal]).all() or not torch.isneginf(z[~legal]).all():raise ValueError('Mask/finite score contract')
                    probs=torch.softmax(z,-1)
                    if torch.count_nonzero(probs[~legal]):raise ValueError('Illegal probability leakage')
                    action=int(z.argmax()) if greedy else int(torch.multinomial(probs,1,generator=gen))
                    states.append(capture(env));actions.append(action);logits.append(z.clone());env.step(action+1)
            result=env.result
            if abs(math.fsum(env.rewards)+result['J'])>1e-7:raise ValueError('Return differs from negative complete terminal cost')
            if source.tensor_digest(policy.state_dict())!=before:raise ValueError('Request changed Actor')
            raw_plan=export(problem,result['raw_route'],role+'_raw');cert=validate(problem,raw_plan)
            if not cert['valid_model_plan']:raise ValueError('Invalid complete raw route')
            raw_plan['validation']=cert;result['plan']['method']=role
            record=dict(spec=spec,physical_key=evidence['physical_key'],scope_evidence=evidence,plan=result['plan'],raw_plan=raw_plan,
                raw_route=result['raw_route'],route=result['route'],metrics={k:result[k] for k in ('J','L','K','raw_J','raw_L','raw_K')},
                actor_tensor_sha256=before,parameter_version=getattr(policy,'revision',256),actions=actions,
                action_masks=[s.mask().tolist() for s in states],ordered_prefixes=[s.prefix for s in states],
                complete_return=-result['J'],environment_reward_sum=math.fsum(env.rewards),environment_rewards=env.rewards,
                counts=dict(stats),H_s=result['H_s'],internal_search=internal,
                timing=dict(input_prepare_s=prepare,original_features_and_forward_s=forward_s),
                raw_semantics='Complete original Actor route before unchanged H; exact same verifier',
                timer_scope='Input/source/navigation restore,original features,decode,H,raw/final export and validation; journal I/O included in request timing receipt')
            episode=dict(problem=problem,states=states,actions=actions,old_logits=logits,role=role,episode_id=rid,actor_tensor_sha=before,
                J_H=result['J'],complete_return=-result['J'])
            return dict(record=source._json_value(record),episode=episode)
        finally:
            # Charge terminal work even if later export, validation or journal fails.
            ledger.counts['full_H_calls']+=stats['full_H_calls'];ledger.counts['feature_states']+=len(states)
            ledger.counts['feature_s']+=forward_s;ledger.counts['original_forward_calls']+=forward_calls
            for key in ('counts','inclusive_s','exclusive_s'):
                for name,value in internal.get(key,{}).items():
                    target=ledger.counts['internal_search_'+key];target[name]=target.get(name,0)+value
    return ledger.request(rid,compute,role=role,kind='necessary_P0_engineering')

def teacher_episode(ledger,inputs,actor,index,rid):
    ledger.check();ledger.counts['teacher_replays']+=1
    p,actions,evidence=inputs.teacher(index);env=Environment(p,lambda p,r:p.metrics(r));states=[]
    for node in actions:states.append(capture(env));env.step(node)
    if not env.done:raise ValueError('Existing teacher sequence incomplete')
    return dict(problem=p,states=states,actions=[a-1 for a in actions],role=actor.role,episode_id=rid,
        actor_tensor_sha=source.tensor_digest(actor.state_dict()),teacher_source=evidence['teacher_source'])
