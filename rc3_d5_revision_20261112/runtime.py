import time,math,base64,copy
from collections import Counter
import numpy as np
import torch
from rc3_global_local.environment import Environment
from rc3_global_local.terminal import finish,export
from rc3_global_local.validation import validate
from rc3_score5_20261111.runtime import capture
from rc3_conservative_refinement.runtime import tensor_digest,load_frozen_sl
from rc3_decoder_program_20261111.actor import load_actor,export as actor_export,load_export
from .common import ROOT,HOME,H_ID,read,sha,gzread,gzput

def generator_text(g):return base64.b64encode(g.get_state().numpy().tobytes()).decode()
def restore_generator(g,text):g.set_state(torch.from_numpy(np.frombuffer(base64.b64decode(text),dtype=np.uint8).copy()))
def sample_generator(seed):return torch.Generator().manual_seed(int(sha_seed(seed)))
def sha_seed(seed):
    import hashlib
    return int(hashlib.sha256(f'D5:sampling:{seed}'.encode()).hexdigest()[:15],16)

def instrument_actor(a,budget):
    if getattr(a,'_revision_metered',False):return a
    enc,score=a.encode,a.score
    def encode(*args,**kw):
        budget.counts['encode_calls']+=1;tick=time.perf_counter()
        try:return enc(*args,**kw)
        finally:budget.counts['encode_s']+=time.perf_counter()-tick
    def scoring(*args,**kw):
        budget.counts['score_calls']+=1;tick=time.perf_counter()
        try:return score(*args,**kw)
        finally:budget.counts['score_s']+=time.perf_counter()-tick
    a.encode=encode;a.score=scoring;a._revision_metered=True;return a

def new_actor(role,budget):return instrument_actor(load_actor(role),budget)
def frozen_actor(budget):return instrument_actor(load_frozen_sl(),budget)
def deployed_actor(entry,budget):return instrument_actor(load_export(entry,engineering=False),budget)

def policy_request(budget,pool,tid,rid,actor,*,purpose,folder,greedy=False,generator=None,seed=None,method=None):
    tick=budget.begin_request(purpose,rid);stats=Counter();internal=None
    try:
        prep=time.perf_counter();p,evidence=pool.get(tid);prepare_s=time.perf_counter()-prep
        load_tick=time.perf_counter()
        if not isinstance(actor,torch.nn.Module):actor=actor()
        model_load_s=time.perf_counter()-load_tick
        from rc3_nav_value.r1.oracle import _instrument,_empty_internal
        internal=_empty_internal()
        def terminal(problem,raw):
            if raw[1:-1]:budget.debit('full_H_calls')
            with _instrument(internal):return finish(problem,raw,stats)
        env=Environment(p,terminal);states=[];actions=[];masks=[];log_probs=[]
        before=tensor_digest(actor.state_dict());version=getattr(actor,'revision',256)
        feature_tick=time.perf_counter()
        with torch.no_grad():
            encoded=actor.encode(p) if not env.done else None;budget.counts['cache_rebuilds']+=int(encoded is not None)
            while not env.done:
                budget.check();z=actor.score(env,encoded);mask=env.mask();legal=torch.tensor(mask)
                if not torch.isfinite(z[legal]).all() or not torch.isneginf(z[~legal]).all():raise ValueError('Finite/mask contract')
                prob=torch.softmax(z,-1)
                if bool(prob[~legal].any()):raise ValueError('Illegal probability mass')
                a=int(z.argmax()) if greedy else int(torch.multinomial(prob,1,generator=generator))
                states.append(capture(env));actions.append(a);masks.append(base64.b64encode(np.packbits(mask,bitorder='little').tobytes()).decode())
                log_probs.append(float(torch.log_softmax(z,-1)[a]));env.step(a+1)
        algorithm_s=time.perf_counter()-feature_tick
        r=env.result
        if abs(math.fsum(env.rewards)+r['J'])>1e-7:raise ValueError('Return not negative complete H cost')
        if tensor_digest(actor.state_dict())!=before:raise ValueError('Collection changed Actor')
        raw_plan=export(p,r['raw_route'],(method or getattr(actor,'role','FrozenSL'))+'_raw');raw_cert=validate(p,raw_plan)
        if not raw_cert['valid_model_plan']:raise ValueError('Invalid complete raw plan')
        raw_plan['validation']=raw_cert;r['plan']['method']=method or getattr(actor,'role','FrozenSL')
        record=dict(request_id=rid,purpose=purpose,run_id=read(HOME/'STATUS.json')['run_id'],task_id=tid,problem_identity=p.identity,parent_identity=p.parent.identity,
            physical_key=pool.entries[tid]['physical_key'],spec=pool.entries[tid]['spec'],method=method or getattr(actor,'role','FrozenSL'),seed=seed,
            actor_tensor_sha256=before,parameter_version=version,input_sources=evidence['input_source_files'],scope=evidence,
            mask_bitorder='little',candidate_count=len(p.candidate_ids),mask_bits_b64=masks,actions=actions,sample_log_prob=log_probs,
            sampler_after=None if generator is None else generator_text(generator),raw_route=r['raw_route'],route=r['route'],raw_plan=raw_plan,plan=r['plan'],
            metrics={k:r[k] for k in ('J','L','K','raw_J','raw_L','raw_K')},complete_return=-r['J'],reward_sum=math.fsum(env.rewards),
            rewards=env.rewards,counts=dict(stats),internal_search=internal,H_identity=H_ID,raw_semantics='Actor raw before fixed common H',
            timing=dict(input_prepare_s=prepare_s,model_load_s=model_load_s,decode_H_s=algorithm_s,H_s=r['H_s'],export_s=r['export_s'],validation_s=r['validation_s']),
            cache_state='Immutable task snapshot reused when present; current Actor encode/key/prefix rebuilt every trajectory',valid=True)
        ep=dict(problem=p,states=states,actions=actions,role=getattr(actor,'role','FrozenSL'),episode_id=rid,actor_tensor_sha=before,J_H=r['J'],complete_return=-r['J'],task_id=tid)
        ref,full=budget.complete(purpose,rid,record,tick,folder)
        return dict(record=record,episode=ep,ref=ref,full_request_s=full)
    except BaseException as exc:budget.failed(purpose,rid,exc);raise
    finally:
        if internal is not None:budget.internal(internal);budget.save()

def recover_episode(pool,record,role):
    p,_=pool.get(record['task_id'])
    if p.identity!=record['problem_identity'] or role!=record['method']:raise ValueError('Recovered own-policy episode role/input')
    def saved_terminal(problem,raw):
        if raw!=record['raw_route']:raise ValueError('Recovered terminal raw identity')
        return dict(J=record['metrics']['J'])
    env=Environment(p,saved_terminal);states=[]
    for a,packed in zip(record['actions'],record['mask_bits_b64']):
        mask=np.unpackbits(np.frombuffer(base64.b64decode(packed),dtype=np.uint8),bitorder='little')[:len(p.candidate_ids)].astype(bool)
        if not np.array_equal(env.mask(),mask) or not bool(mask[a]):raise ValueError('Recovered current-mask mismatch')
        states.append(capture(env));env.step(a+1)
    if not env.done:raise ValueError('Recovered incomplete online trace')
    return dict(problem=p,states=states,actions=record['actions'],role=role,episode_id=record['request_id'],actor_tensor_sha=record['actor_tensor_sha256'],
        J_H=record['metrics']['J'],complete_return=record['complete_return'],task_id=record['task_id'])

def teacher_episode(budget,pool,actor,tid,rid):
    p,actions,source=pool.teacher(tid);env=Environment(p,lambda problem,route:problem.metrics(route));states=[]
    budget.debit('teacher_replays')
    budget.debit('engineering_teacher_replays' if budget.pool=='engineering' else 'scientific_teacher_replays',limit=None if budget.pool=='engineering' else 393216)
    for a in actions:
        if not bool(env.mask()[a-1]):raise ValueError('Supervised action not legal')
        states.append(capture(env));env.step(a)
    if not env.done:raise ValueError('Supervised full sequence incomplete')
    return dict(problem=p,states=states,actions=[a-1 for a in actions],role=actor.role,episode_id=rid,
        actor_tensor_sha=tensor_digest(actor.state_dict()),task_id=tid,teacher_source=source)

def native_request(budget,pool,tid,rid,method,folder,purpose='eval'):
    from rc3_global_local_impl_review.endpoint_baselines import run_endpoint
    tick=budget.begin_request(purpose,rid);stats=Counter()
    try:
        p,evidence=pool.get(tid);budget.debit('native_requests')
        plan,info=run_endpoint(p,'fixed3' if method=='Fixed3' else 'detour16',time_limit_s=None,reject_fallback=True,counters=stats)
        route=info['local_route'];m=p.metrics(route);record=dict(request_id=rid,purpose=purpose,task_id=tid,problem_identity=p.identity,parent_identity=p.parent.identity,
            physical_key=pool.entries[tid]['physical_key'],spec=pool.entries[tid]['spec'],method=method,plan=plan,raw_plan=plan,raw_route=route,route=route,
            metrics=dict(m,raw_J=m['J'],raw_L=m['L'],raw_K=m['K']),counts=dict(stats),internal_search=info['internal_search'],
            raw_semantics='Native final, including original internal search; not neural no-H ablation',valid=True,input_sources=evidence['input_source_files'])
        budget.internal(info['internal_search']);ref,wall=budget.complete(purpose,rid,record,tick,folder);return dict(record=record,ref=ref,full_request_s=wall)
    except BaseException as exc:budget.failed(purpose,rid,exc);raise

def reference_key(p):
    import hashlib,json
    from rc3_conservative_refinement.runtime import SL_TENSOR_SHA256
    sources={f:sha(ROOT/f) for f in ('rc3_global_local/environment.py','rc3_global_local/terminal.py','rc3_global_local/validation.py','rc3_global_local/policy.py')}
    return hashlib.sha256(json.dumps(dict(problem=p.identity,SL=SL_TENSOR_SHA256,H=H_ID,cost='J=L+.75K; complete unnormalized',sources=sources),sort_keys=True).encode()).hexdigest()

class References:
    def __init__(self,pool,budget):self.pool=pool;self.budget=budget;self.manifest=read(HOME/'REFERENCE_MANIFEST.json');self.cached={}
    def get(self,tid):
        if tid not in self.cached:
            p,_=self.pool.get(tid);e=self.manifest['records'][tid]
            if e['key']!=reference_key(p) or sha(ROOT/e['path'])!=e['sha256']:raise ValueError('Frozen reference cache identity')
            rec=gzread(ROOT/e['path'])
            if not rec['plan']['validation']['valid_model_plan'] or rec['problem_identity']!=p.identity:raise ValueError('Unvalidated reference')
            self.cached[tid]=rec['metrics']['J']
        self.budget.counts['reference_cache_reads']+=1;return self.cached[tid]
