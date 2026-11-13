import copy,math,time,os
import torch
from .common import ROOT,HOME,D5,read,put,sha,identity,state,unarchive
from .data import Pool
from .runtime import new_actor,policy_request,teacher_episode,sample_generator,References,recover_episode
from .training import Trainer
from . import evaluation
from rc3_conservative_refinement.runtime import tensor_digest,SL_TENSOR_SHA256

LABELS={'A':'D5-PG-A','B':'D6-PG','SL':'D6-SL'}
def export_actor(t,point):
    target=t.path/f'step{point:05}.actor.pt';meta=target.with_suffix('.json')
    if meta.exists():
        e=read(meta)
        if sha(ROOT/e['path'])!=e['sha256'] or e['identity']!=identity():raise ValueError('Existing export identity')
        return e
    a=t.actor
    if a.frozen_digest()!=a.initial_frozen_digest:raise ValueError('Frozen export')
    payload=dict(schema='rc3-d6-actor-v1',identity=identity(),stream=t.stream,role=a.role,model=copy.deepcopy(a.state_dict()),base=a.base_identity,
        revision=a.revision,tensor_sha256=tensor_digest(a.state_dict()),engineering_only=False,critic_included=False)
    with target.open('xb') as f:torch.save(payload,f)
    e=dict(path=target.relative_to(ROOT).as_posix(),sha256=sha(target),identity=identity(),stream=t.stream,role=a.role,endpoint=point,
        tensor_sha256=payload['tensor_sha256'],engineering_only=False)
    put(meta,e);return e

def train(budget,stream,target):
    if read(HOME/'COST_DECOMPOSITION_VALIDATION.json')['status']!='PASS':raise PermissionError('Old raw/H decomposition prerequisite failed')
    pool=Pool(budget);schedule=read(HOME/'COMMON_SCHEDULE.json')['batches'];gen=sample_generator(2026111101)
    a=new_actor('D5-SL' if stream=='SL' else 'D5-PG',budget);a.d6_stream=stream;t=Trainer(a,budget,stream,gen)
    if (t.path/'LATEST.json').exists():t.restore_current(read(t.path/'LATEST.json'))
    else:
        if tensor_digest(a.state_dict())!=SL_TENSOR_SHA256:raise ValueError('Exact fresh SL256 required')
        if budget.counts['optimizer_'+stream]:raise ValueError('Missing checkpoint with charged optimizer')
        export_actor(t,0);t.save_current()
    if t.cursor>target:raise ValueError('Cannot rewind or select earlier endpoint')
    refs=References(pool,budget) if stream!='SL' else None;started=time.time()
    while t.cursor<target:
        budget.check();op=t.cursor+1;ids=schedule[op-1];before=tensor_digest(a.state_dict());batchfile=t.path/'ACTIVE_BATCH.json'
        active=read(batchfile) if batchfile.exists() else None
        if active and active['op']<=t.cursor:active=None
        if active is None:active=dict(op=op,stream=stream,before=before,task_ids=ids,episodes=[])
        if active['op']!=op or active['stream']!=stream or active['before']!=before or active['task_ids']!=ids:raise ValueError('Pending batch exact identity')
        put(batchfile,active,batchfile.exists());episodes=[]
        for slot,tid in enumerate(ids):
            rid=f'{stream}:{op}:{slot}:attempt{budget.counts["requests"]+1}'
            if stream=='SL':
                ep=teacher_episode(budget,pool,a,tid,f'SL:{op}:{slot}');ep['stream']=stream
            elif slot<len(active['episodes']):
                r=unarchive(active['episodes'][slot]);ep=recover_episode(pool,r,a.role)
                from .runtime import restore_generator
                restore_generator(gen,r['sampler_after']);budget.counts['recovered_own_online_episodes']+=1
            else:
                out=policy_request(budget,pool,tid,rid,a,purpose='train',folder=f'streams/{stream}',generator=gen,seed=2026111101,method=LABELS[stream],stream=stream)
                ep=out['episode'];active['episodes'].append(out['ref']);put(batchfile,active,True)
            if refs is not None:ep['reference_J_H']=refs.get(tid)
            ep['stream']=stream;episodes.append(ep)
        result=t.update(episodes);t.save_current()
        if t.cursor in (128,256):export_actor(t,t.cursor)
        if t.cursor%16==0:
            state(status='TRAINING',stream=stream,update=t.cursor)
            print(dict(phase='TRAIN',stream=stream,update=t.cursor,gradient_norm=result['gradient_norm'],job_s=budget.elapsed()),flush=True)
    entry=export_actor(t,target)
    put(t.path/f'COMPLETE_{target}.json',dict(status='COMPLETE',stream=stream,updates=t.cursor,export=entry,identity=identity(),
        actual_Adam_optimizer_attempts=budget.counts['optimizer_'+stream],job_training_s=time.time()-started),False)
    return entry

def evaluate(budget,endpoint):
    pool=Pool(budget);reports={}
    order=['FrozenSL','A','B','SL','Fixed3','Detour16']
    for label in order:
        method=LABELS.get(label,label);entry=None
        if label in LABELS:
            entry=read(HOME/'streams'/label/f'step{endpoint:05}.actor.json')
            if entry['stream']!=label or entry['endpoint']!=endpoint:raise ValueError('Fixed endpoint role')
        reports[label]=evaluation.evaluate(budget,pool,method,2026111101,endpoint,entry=entry,tag=f'{label}_{endpoint}')
        print(dict(phase='EVAL',method=method,endpoint=endpoint,complete=reports[label]['complete']),flush=True)
    comparisons={}
    for name in ('FrozenSL','SL','A'):
        comparisons[name]={m:evaluation.gains(reports['B']['rows'],reports[name]['rows'],m) for m in ('J','raw_J','L','K')}
        comparisons[name]['groups']={g:evaluation.gains(reports['B']['rows'],reports[name]['rows'],'J',g) for g in ('trained_parent_new_tasks_exposed_selection','independent_parent_exposed_development')}
    valid=all(r['valid'] for rep in reports.values() for r in rep['rows'])
    passed=valid and all(comparisons[c]['J']['complete'] and comparisons[c]['J']['mean']>=.005 for c in ('FrozenSL','SL'))
    passed=passed and all(v['complete'] and v['mean']>=0 for c in ('FrozenSL','SL') for v in comparisons[c]['groups'].values())
    gate=dict(endpoint=endpoint,primary_endpoint=256,status='DEVELOPMENT_SIGNAL' if passed else 'NO_DEVELOPMENT_SIGNAL',passed=bool(passed),comparisons=comparisons,
        all396_current_requests_valid=valid,development_exposed=True,independent_generalization=False)
    put(HOME/f'QUALITY_GATE_{endpoint}.json',gate)
    if endpoint==128:
        put(HOME/'STEP0_QUALITY_BINDING.json',dict(quality_source='evaluations/development/FrozenSL_128.json',same_tensor_sha256=SL_TENSOR_SHA256,
            all_three_step0_exports_exact=True,new_step0_requests=0,timing='Step0 separate timing NOT_RUN; quality reuses current exact FrozenSL only'))
    if endpoint==256:
        state(status='COMPLETE',closed=True,scientific_result=gate['status'],remaining_stages='NONE',next_action='Saved-output audit, independent review and bounded publication only')
        put(HOME/'NEXT_STEP_DECISION.json',dict(status=gate['status'],primary_endpoint=256,no_automatic_additional_training=True,gate=gate))
