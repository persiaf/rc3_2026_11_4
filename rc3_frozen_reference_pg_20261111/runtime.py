"""Read-only saved inputs; complete on-policy requests and label replay."""
import copy
import time
import torch
from rc3_conservative_refinement import runtime as old
from rc3_conservative_refinement.adapter import pack_candidate_states
from rc3_global_local.environment import Environment
from . import actor as production
from .common import ROLES

H_CALLS=[]
FEATURE_TIMES={}
FEATURE_COUNTS={}
def install_feature_timing():
    from rc3_conservative_refinement import adapter
    def wrapped(name,fn):
        def call(*a,**kw):
            t=time.perf_counter();FEATURE_COUNTS[name]=FEATURE_COUNTS.get(name,0)+1
            try:return fn(*a,**kw)
            finally:FEATURE_TIMES[name]=FEATURE_TIMES.get(name,0.)+time.perf_counter()-t
        return call
    for n in ('prepare_service_distances','prepared_post_service_distances','augment_candidate_features'):
        setattr(adapter,n,wrapped(n,getattr(adapter,n)))
def install_H_accounting(ledger):
    from rc3_constructive import baselines
    original=baselines.postprocess
    ledger.counts.update(H_function_attempts=0,H_function_completed=0,H_function_s=0.)
    def counted(problem,route,*a,**kw):
        ledger.check();ledger.counts['H_function_attempts']+=1
        raw=list(map(int,route));start=time.perf_counter()
        try:
            result=original(problem,route,*a,**kw);ledger.counts['H_function_completed']+=1
            H_CALLS.append(dict(raw_route=raw,postprocess_route=list(map(int,result)),seconds=time.perf_counter()-start))
            return result
        finally:ledger.counts['H_function_s']+=time.perf_counter()-start
    baselines.postprocess=counted

def request(ledger,inputs,spec,rid,role,*,op=0,kind='training',actor=None,entry=None,greedy=False,seed=0,equivalence=False,case=None):
    def compute():
        start=time.perf_counter();h0=len(H_CALLS);tick=time.perf_counter()
        ft=dict(FEATURE_TIMES);fc=dict(FEATURE_COUNTS)
        problem,evidence=inputs.view(spec);prepare_s=time.perf_counter()-tick;tick=time.perf_counter()
        if role in ROLES:
            policy=production.load_export(entry,role=role,usage=ledger.stage) if entry else actor
            if policy is None or policy.role!=role:raise ValueError('Production Actor role')
            load_s=time.perf_counter()-tick if entry else 0.
            episode=old.rollout(problem,policy,critic=None,greedy=greedy,sampling_seed=seed,check=ledger.check,equivalence=equivalence)
            record=episode['record'];episode.update(episode_id=rid,role=role,stage=ledger.stage,window=f"{spec['width']}x{spec['height']}")
            record['correction_semantics']=production.CORRECTION_SCHEMA
        elif role=='FrozenSL':
            policy=old.load_frozen_sl();load_s=time.perf_counter()-tick
            record=old.sl_route(problem,policy,check=ledger.check);episode=None
        elif role in ('Fixed3','Detour16'):
            from rc3_global_local_impl_review.endpoint_baselines import run_endpoint
            stats={};load_s=0.
            plan,info=run_endpoint(problem,'fixed3' if role=='Fixed3' else 'detour16',
                time_limit_s=max(.001,ledger.cap['wall_s']-ledger.elapsed()-15),reject_fallback=True,counters=stats)
            route=info['local_route'];m=problem.metrics(route)
            record=dict(input_identity=problem.identity,parent_identity=problem.parent.identity,problem=problem.description(),
                plan=plan,raw_plan=plan,route=route,raw_route=route,global_route=problem.global_route(route),raw_global_route=problem.global_route(route),
                metrics=dict(m,raw_J=m['J'],raw_L=m['L'],raw_K=m['K']),counts=stats,internal_search=info['internal_search'],native_info=info,
                raw_semantics='Native final searched/H route; no extra common H. All native pre/postprocess routes separately retained in H_invocations.')
            episode=None
        else:raise ValueError('Unregistered method')
        record.update(spec=copy.deepcopy(spec),physical_key=evidence['physical_key'],scope_evidence=evidence,
            H_invocations=copy.deepcopy(H_CALLS[h0:]),timing=dict(input_prepare_s=prepare_s,model_load_s=load_s,service_before_IO_s=time.perf_counter()-start),
            timer_scope='Saved input/provenance/navigation restoration, optional cold model, prepare/provider/features/actions,H,exports,validation,evidence; persistence separately timed in request index')
        record['new_feature_overhead_s']={k:v-ft.get(k,0.) for k,v in FEATURE_TIMES.items()}
        record['new_feature_calls']={k:v-fc.get(k,0) for k,v in FEATURE_COUNTS.items()}
        if case:record.update(case_id=case['id'],cohort=case['group'])
        if not record['plan']['validation']['valid_model_plan']:raise ValueError('Invalid complete route')
        # Per-request H history is already in the record; bound the live buffer.
        del H_CALLS[h0:]
        record=old._json_value(record)
        if episode is not None:episode['record']=record;return episode
        return record
    return ledger.request(rid,compute,role=role,op=op,kind=kind)

def teacher_episode(ledger,inputs,actor,index,rid):
    ledger.check();ledger.counts['teacher_replays']+=1
    ledger.event('teacher_replay_debit',index=index,episode_id=rid)
    tick=time.perf_counter();problem,actions,evidence=inputs.teacher(index)
    env=Environment(problem,lambda p,r:p.metrics(r));states=[]
    with torch.no_grad():
        encoded=actor.encode(problem) if not env.done else None
        for node in actions:
            ledger.check()
            if env.done or not env.mask()[node-1]:raise ValueError('Frozen teacher action invalid')
            states.append(actor.features(env,encoded));env.step(node)
    if not env.done:raise ValueError('Incomplete frozen teacher')
    n=problem.window
    spec=inputs.references['teachers'][index]
    # Window dimensions come from the actual scoped view, never final trajectory length.
    width=problem.window['width'];height=problem.window['height']
    ledger.event('teacher_replay_complete',episode_id=rid,states=len(states),wall_s=time.perf_counter()-tick,source=evidence['teacher_source'])
    return dict(batch=pack_candidate_states(states),actions=torch.tensor(actions,dtype=torch.long)-1,
        task_count=len(problem.C),episode_id=rid,role=actor.role,stage=ledger.stage,window=f'{width}x{height}')
