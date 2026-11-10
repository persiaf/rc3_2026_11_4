"""One paid H per completed on-policy episode; no H action relabeling."""
import time
import numpy as np
import torch
from .common import *
from .model import pack_states
from .ppo import returns_to_go
from rc3_global_local.environment import Environment
from rc3_global_local.terminal import finish
from rc3_nav_value.r1.oracle import _instrument,_empty_internal


def collect(actor,critic,p,q,store,clock,*,scale_tasks=True,greedy=False,equivalence=False):
    features=[];old=[];values=[];base=[];actions=[]
    before=tensor_digest(actor.head.state_dict());base_sha=actor.base_sha
    source=getattr(actor,'source_identity',None)
    require(source is not None and digest(ROOT/source['path'])==source['sha256'],'Rollout actor source is missing/changed before request')
    sampling_stage=q.get('sampling_stage',q.get('stage'))
    rng=torch.Generator().manual_seed(seed_for(q.get('seed'),sampling_stage,q.get('round'),q.get('slot'),q.get('row')))
    def compute():
        rng.manual_seed(seed_for(q.get('seed'),sampling_stage,q.get('round'),q.get('slot'),q.get('row')))
        features.clear();old.clear();values.clear();base.clear();actions.clear()
        stats=dict(full_H_calls=0,export_attempts=0,exports=0,validation_attempts=0,validation_completed=0);internal=_empty_internal()
        def terminal(p,raw):
            clock.check()
            with _instrument(internal):r=finish(p,raw,stats)
            return r
        env=Environment(p,terminal);t=time.perf_counter()
        with torch.no_grad():
            encoded=actor.encode(p) if not env.done else None
            while not env.done:
                clock.check();f=actor.features(env,encoded);z=actor.logits(f)
                if equivalence:require(torch.equal(z,f['z0']) and torch.equal(torch.isfinite(z),f['mask']),'Zero residual not exact SL logits/mask')
                lp=torch.log_softmax(z,0);p0=torch.log_softmax(f['z0'],0)
                node=int(z.argmax()) if greedy else int(torch.multinomial(lp.exp(),1,generator=rng))
                features.append(f);old.append(lp);base.append(p0);actions.append(node);values.append(0. if critic is None else float(critic(f['v'])))
                env.step(node+1)
        elapsed=time.perf_counter()-t;require(abs(sum(env.rewards)+env.result['J'])<1e-7,'Dense return not -J_H')
        require(before==tensor_digest(actor.head.state_dict()) and base_sha==tensor_digest(actor.base.state_dict()),'Rollout mutated actor/base')
        return dict(problem=p.description(),model_identity=dict(run_id=EXP,study_method=q.get('role','RL'),checkpoint_role=q.get('role','RL'),
            actor_sha=source['sha256'],actor_file=source['path'],head_tensor_sha=before,tensor_sha=tensor_digest(actor.state_dict()),base_tensor_sha=base_sha,seed=q.get('seed'),update=max(0,q.get('round',0)-1)),
            actions=[a+1 for a in actions],old_logp=[float(v[a]) for v,a in zip(old,actions)],V_old=values,rewards=env.rewards,
            return_sum=sum(env.rewards),normalization=max(1,len(p.C)) if scale_tasks else 1,
            result=env.result,internal_search=internal,counts=stats,policy_steps=len(actions),policy_feature_preparation_and_construction_s=elapsed,
            sampling_stage=sampling_stage,rng_seed=seed_for(q.get('seed'),sampling_stage,q.get('round'),q.get('slot'),q.get('row')),
            rng_end_sha256=hashlib.sha256(rng.get_state().numpy().tobytes()).hexdigest(),input_identity=p.identity)
    existed=identity(q)+':0' in store.done or identity(q)+':1' in store.done
    if existed:
        saved=store.request(q,compute,clock)
        require(saved['model_identity']['head_tensor_sha']==before,'Old rollout is off-policy for current actor')
        # Reconstruct detached states only, without resampling, H, or a terminal query.
        env=Environment(p,lambda p,r:saved['result']);encoded=actor.encode(p) if not env.done else None
        with torch.no_grad():
            for node,lp0,v0 in zip(saved['actions'],saved['old_logp'],saved['V_old']):
                f=actor.features(env,encoded);lp=torch.log_softmax(actor.logits(f),0)
                require(abs(float(lp[node-1])-lp0)<1e-5,'Recovered old likelihood differs')
                features.append(f);old.append(lp);base.append(torch.log_softmax(f['z0'],0));values.append(v0);actions.append(node-1);env.step(node)
    else:saved=store.request(q,compute,clock)
    b=pack_states(features);features.clear()
    return dict(batch=b,old_logp_all=torch.stack(old) if old else None,base_logp=torch.stack(base) if base else None,
        old_logp=torch.tensor(saved['old_logp'],dtype=torch.float32),V_old=torch.tensor(saved['V_old'],dtype=torch.float32),
        G=returns_to_go(saved['rewards'],saved['normalization']) if b is not None else torch.zeros(0),
        actions=torch.tensor([a-1 for a in saved['actions']],dtype=torch.long),record=saved,old_actor_sha=before)


def teacher_episode(actor,p,actions):
    states=[];env=Environment(p,lambda p,r:p.metrics(r));encoded=actor.encode(p) if not env.done else None
    with torch.no_grad():
        for node in actions:
            require(not env.done and env.mask()[node-1],'Illegal saved teacher action')
            states.append(actor.features(env,encoded));env.step(node)
    require(env.done,'Incomplete teacher labels')
    return dict(batch=pack_states(states),actions=torch.tensor([x-1 for x in actions],dtype=torch.long))
