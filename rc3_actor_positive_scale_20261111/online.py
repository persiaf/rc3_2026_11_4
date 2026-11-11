"""Only executes after the preregistered positive-control trigger."""
import copy
import csv
import time
from collections import defaultdict
import numpy as np
import torch
from rc3_frozen_reference_pg_20261111 import actor as old_actor,runtime as saved_runtime
from rc3_frozen_reference_pg_20261111.trainer import Trainer
from rc3_frozen_reference_pg_20261111.reference import FrozenReference
from rc3_conservative_refinement import runtime as rt
from .common import HOME,ROOT,TABLE,read,put,sha,csvrow,identity

ROLE={'A':'PG-cond','B':'PG-cond','C':'continued_SL_A1'}
METHODS=('A','B','C','FrozenSL','Fixed3','Detour16')

class ArmLedger:
    def __init__(self,main,arm,lr):
        self.main,self.arm,self.lr=main,arm,lr;self.path=main.path/'arms'/arm;self.path.mkdir(parents=True)
        self.stage=main.stage;self.counts=main.counts;self.ids=main.ids
    def check(self):return self.main.check()
    def optimizer(self,role,op):return self.main.optimizer(self.arm,op)
    def snapshot(self):return dict(self.main.snapshot(),arm=self.arm,learning_rate=self.lr)
    def event(self,event,**kw):
        model_role=kw.pop('role',None)
        return self.main.event(event,role=self.arm,model_role=model_role,**kw)

def export(path,actor,arm,lr,op):
    package=dict(schema='rc3-actor-step-scale-A1-export-v1',arm=arm,model_role=ROLE[arm],usage='online_science',
        learning_rate=lr,opportunity=op,base=actor.base_identity,correction_schema=old_actor.CORRECTION_SCHEMA,
        feature_schema=rt.FEATURE_SCHEMA,distances_enabled=True,amplitude=.25,input_dim=206,hidden=[64,64],
        correction=copy.deepcopy(actor.correction.state_dict()),tensor_sha256=rt.tensor_digest(actor.correction.state_dict()))
    with path.open('xb') as f:torch.save(package,f)
    return dict(path=str(path),sha256=sha(path),arm=arm,learning_rate=lr)

def load_export(entry,arm):
    if entry['arm']!=arm or sha(entry['path'])!=entry['sha256']:raise ValueError('Exact online arm or bytes mismatch')
    p=torch.load(entry['path'],map_location='cpu',weights_only=True)
    expected={'schema','arm','model_role','usage','learning_rate','opportunity','base','correction_schema','feature_schema','distances_enabled','amplitude','input_dim','hidden','correction','tensor_sha256'}
    if set(p)!=expected or p['schema']!='rc3-actor-step-scale-A1-export-v1' or p['arm']!=arm or p['usage']!='online_science' or p['model_role']!=ROLE[arm]:raise ValueError('Online export schema/arm mismatch')
    if (p['feature_schema'],p['correction_schema'],p['distances_enabled'],p['amplitude'],p['input_dim'],p['hidden'])!=(rt.FEATURE_SCHEMA,old_actor.CORRECTION_SCHEMA,True,.25,206,[64,64]):raise ValueError('Online A1 semantics mismatch')
    actor=old_actor.load_actor(ROLE[arm])
    if actor.base_identity!=p['base']:raise ValueError('SL256 source mismatch')
    actor.correction.load_state_dict(p['correction']);assert rt.tensor_digest(actor.correction.state_dict())==p['tensor_sha256']
    actor.eval().requires_grad_(False);return actor

def request(ledger,inputs,spec,rid,method,op,*,actor=None,entry=None,seed=0,case=None):
    def compute():
        started=time.perf_counter();h0=len(saved_runtime.H_CALLS);ft=dict(saved_runtime.FEATURE_TIMES);fc=dict(saved_runtime.FEATURE_COUNTS)
        tick=time.perf_counter();problem,evidence=inputs.view(spec);prep=time.perf_counter()-tick;tick=time.perf_counter()
        if method in ROLE:
            policy=load_export(entry,method) if entry else actor
            if policy.role!=ROLE[method]:raise ValueError('Current policy arm role')
            load_s=time.perf_counter()-tick if entry else 0.
            choices=[];original_logits=policy.logits
            def measured_logits(features):
                z=original_logits(features)
                choices.append(dict(base_top1=int(features['z0'].argmax()),actor_top1=int(z.argmax()),legal_count=int(features['mask'].sum())))
                return z
            policy.logits=measured_logits
            try:episode=rt.rollout(problem,policy,critic=None,greedy=case is not None,sampling_seed=seed,check=ledger.check)
            finally:policy.logits=original_logits
            record=episode['record'];episode.update(episode_id=rid,role=ROLE[method],stage=ledger.stage,window=f"{spec['width']}x{spec['height']}")
            record['correction_semantics']=old_actor.CORRECTION_SCHEMA
            record['actual_prefix_first_choice_comparison']=choices
        elif method=='FrozenSL':
            model=rt.load_frozen_sl();load_s=time.perf_counter()-tick;record=rt.sl_route(problem,model,check=ledger.check);episode=None
        else:
            from rc3_global_local_impl_review.endpoint_baselines import run_endpoint
            stats={};load_s=0.
            plan,info=run_endpoint(problem,'fixed3' if method=='Fixed3' else 'detour16',
                time_limit_s=max(.001,ledger.cap['wall_s']-ledger.elapsed()-15),reject_fallback=True,counters=stats)
            route=info['local_route'];m=problem.metrics(route)
            record=dict(input_identity=problem.identity,problem=problem.description(),plan=plan,raw_plan=plan,
                route=route,raw_route=route,global_route=problem.global_route(route),raw_global_route=problem.global_route(route),
                metrics=dict(m,raw_J=m['J'],raw_L=m['L'],raw_K=m['K']),counts=stats,internal_search=info['internal_search'],native_info=info,
                raw_semantics='Native final pipeline; no extra common H. Internal recipe/search counters separately retained.')
            episode=None
        record.update(spec=copy.deepcopy(spec),physical_key=evidence['physical_key'],scope_evidence=evidence,
            H_invocations=copy.deepcopy(saved_runtime.H_CALLS[h0:]),
            new_feature_overhead_s={k:v-ft.get(k,0.) for k,v in saved_runtime.FEATURE_TIMES.items()},
            new_feature_calls={k:v-fc.get(k,0) for k,v in saved_runtime.FEATURE_COUNTS.items()},
            timing=dict(input_prepare_s=prep,model_load_s=load_s,service_before_IO_s=time.perf_counter()-started))
        if case:record.update(case_id=case['id'],cohort=case['group'])
        record=rt._json_value(record);del saved_runtime.H_CALLS[h0:]
        if episode is not None:episode['record']=record;return episode
        return record
    return ledger.request(rid,compute,role=method,op=op,kind='evaluation' if case else 'training')

def evaluate(ledger,op):
    rows=list(csv.DictReader((ledger.path/'REQUEST_INDEX.csv').open(encoding='utf-8',newline='')))
    rows=[r for r in rows if r['kind']=='evaluation' and int(r['opportunity'])==op]
    assert len(rows)==108 and len({r['case_id'] for r in rows})==18
    contrasts=[]
    for size in ('ALL','4x3','6x6','8x8'):
        selected=[r for r in rows if size=='ALL' or r['width']+'x'+r['height']==size]
        for control in ('FrozenSL','C','A','Fixed3','Detour16'):
            paired={}
            for r in selected:
                if r['role'] in ('B',control):paired.setdefault((r['physical_key'],r['case_id']),{})[r['role']]=r
            assert all(set(v)=={'B',control} for v in paired.values())
            groups=defaultdict(list)
            for (parent,_),cell in paired.items():groups[parent].append(cell)
            parents=sorted(groups);draw=np.random.Generator(np.random.PCG64(11652026)).integers(len(parents),size=(20000,len(parents)))
            for metric in ('J','raw_J','L','K'):
                gains=[]
                for parent in parents:
                    a=np.mean([float(v['B'][metric]) for v in groups[parent]]);b=np.mean([float(v[control][metric]) for v in groups[parent]])
                    gains.append((b-a)/b)
                gains=np.array(gains);low,high=np.quantile(gains[draw].mean(1),[.025,.975],method='linear')
                contrasts.append(dict(stage_status='COMPLETE',endpoint=op,size=size,candidate='B',control=control,metric=metric,
                    physical_parents=len(parents),relative_saving=float(gains.mean()),ci95_low=float(low),ci95_high=float(high)))
    for r in contrasts:csvrow(HOME/'CONDITIONAL_ONLINE_RESULTS.csv',r)
    main=[r for r in contrasts if r['size']=='ALL' and r['control'] in ('FrozenSL','C') and r['metric'] in ('J','raw_J')]
    sizes=[r for r in contrasts if r['size']!='ALL' and r['control'] in ('FrozenSL','C') and r['metric'] in ('J','raw_J')]
    success=all(r['relative_saving']>=.005 and r['ci95_low']>0 for r in main) and all(r['relative_saving']>=0 for r in sizes)
    d=dict(endpoint=op,status='QUALITY_SCREEN_PASS' if success else 'QUALITY_SCREEN_NOT_MET',primary_checks=main,size_checks=sizes,
        compared_to_A=[r for r in contrasts if r['size']=='ALL' and r['control']=='A'],all_routes_valid=all(r['valid']=='True' for r in rows))
    put(ledger.path/f'ENDPOINT_{op:04d}_DECISION.json',d);return d

def run(ledger,trace,inputs):
    trigger=read(HOME/'TRIGGER_DECISION.json');rate=trigger['selected_learning_rate']
    if not trigger['triggered'] or rate not in (3e-4,1e-3):raise ValueError('Online trigger missing')
    from .common import TABLE
    reference=FrozenReference(read(HOME/'PROTOCOL.json')['table_sha256'])
    saved_runtime.install_H_accounting(ledger);saved_runtime.install_feature_timing()
    schedule=read(HOME/'SCHEDULE.json');rates={'A':3e-5,'B':rate,'C':rate};trainers={};proxies={}
    for arm in ROLE:
        proxy=ArmLedger(ledger,arm,rates[arm]);proxies[arm]=proxy
        model=old_actor.load_actor(ROLE[arm]);ident=dict(source_sha256=sha(HOME/'SOURCE_BINDING.json'),protocol_sha256=sha(HOME/'PROTOCOL.json'),
            schedule_sha256=sha(HOME/'SCHEDULE.json'),table_sha256=reference.sha256,arm=arm,learning_rate=rates[arm])
        t=Trainer(model,reference,proxy,ident);t.optimizer.param_groups[0]['lr']=rates[arm];trainers[arm]=t
    initial={a:rt.tensor_digest(t.actor.correction.state_dict()) for a,t in trainers.items()};assert len(set(initial.values()))==1
    put(ledger.path/'INITIALIZATION.json',dict(fresh_SL256_and_zero_A1=True,initial_tensors=initial,positive_diagnostic_weights_loaded=False,rates=rates))
    models=ledger.path/'models';models.mkdir()
    for arm,t in trainers.items():t.save(models/f'{arm}_0000.training.pt')
    decisions=[]
    for row in schedule['training']:
        op=row['opportunity']
        for arm in ('A','B'):
            episodes=[]
            for slot,spec in enumerate(row['specs']):
                seed=int(identity(['actor-positive-scale',11622026,op,slot])[:15],16)
                episodes.append(request(ledger,inputs,spec,f'train:{arm}:{op}:{slot}',arm,op,actor=trainers[arm].actor,seed=seed))
            trainers[arm].update(episodes);del episodes
        t=trainers['C'];episodes=[saved_runtime.teacher_episode(proxies['C'],inputs,t.actor,i,f'teacher:C:{op}:{slot}') for slot,i in enumerate(row['teacher_indices'])]
        t.update(episodes);del episodes
        if any(t.counts['consecutive_rejected']>=16 for t in trainers.values()):raise RuntimeError('16 guard rejections: stop without refill')
        if op%8==0:print(f'Online opportunity {op}/64; requests={ledger.counts["requests"]}; Actor={ledger.counts["actor_optimizer_attempts"]}',flush=True)
        if op in (32,64):
            with trace.stage('fixed_endpoint_'+str(op)):
                entries={}
                for arm,t in trainers.items():
                    t.save(models/f'{arm}_{op:04d}.training.pt');entries[arm]=export(models/f'{arm}_{op:04d}.actor.pt',t.actor,arm,rates[arm],op)
                for ci,case in enumerate(schedule['evaluation']):
                    offset=(ci+op)%6;order=METHODS[offset:]+METHODS[:offset]
                    for method in order:request(ledger,inputs,case['spec'],f'eval:{op}:{case["id"]}:{method}',method,op,entry=entries.get(method),case=case)
                decisions.append(evaluate(ledger,op))
    assert ledger.counts['requests']==984 and ledger.counts['actor_optimizer_attempts']==192 and ledger.counts['teacher_replays']==384
    put(ledger.path/'FINAL_DECISION.json',dict(status='QUALITY_SCREEN_PASS' if decisions[-1]['status']=='QUALITY_SCREEN_PASS' else 'STOP_CURRENT_CONFIGURATION',
        endpoints=decisions,counts={a:t.counts for a,t in trainers.items()},automatic_extension=False,
        scope='Single-seed18 exposed DEV mechanism screen only'))
