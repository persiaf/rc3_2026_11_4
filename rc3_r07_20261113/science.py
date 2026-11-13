import time,math,copy,base64,hashlib,json
from collections import Counter,defaultdict
import numpy as np
import torch
from rc3_d5_revision_20261112.data import Pool as OldPool
from rc3_global_local.environment import Environment
from rc3_global_local.terminal import finish,export
from rc3_global_local.validation import validate
from rc3_global_local.problem import array_hash
from rc3_conservative_refinement.runtime import load_frozen_sl,tensor_digest,SL_TENSOR_SHA256
from .common import ROOT,HOME,D5,RUN,H_ID,read,put,sha,gzread,identity,state,unarchive

STAGES=('early','middle','late')
class Pool(OldPool):
    def __init__(self,budget):
        super().__init__(budget);ids={r['id'] for r in read(HOME/'TASKS.json')['records']}
        self.entries={k:v for k,v in self.entries.items() if k in ids};self.full={k:v for k,v in self.full.items() if k in ids};self.local={k:v for k,v in self.local.items() if k in ids};self.cases=[]

def packed(mask):return base64.b64encode(np.packbits(mask,bitorder='little').tobytes()).decode()
def fingerprint(p,env):
    return hashlib.sha256(json.dumps(dict(problem=p.identity,prefix=env.prefix,current=env.current,remaining=np.flatnonzero(env.remaining).tolist(),raw_L=env.raw_L,mask=packed(env.mask())),sort_keys=True).encode()).hexdigest()
def prefix_env(p,prefix,terminal):
    env=Environment(p,terminal)
    for node in prefix:
        if env.done:raise ValueError('Prefix reaches terminal before intervention')
        env.step(node)
    if env.done:raise ValueError('Intervention after terminal')
    return env
def no_terminal(*args):raise RuntimeError('Selection/prefix restoration cannot issue a complete/H request')
def reference(p,tid):
    e=read(D5/'REFERENCE_MANIFEST.json')['records'][tid]
    if sha(ROOT/e['path'])!=e['sha256']:raise ValueError('FrozenSL reference file')
    r=gzread(ROOT/e['path'])
    if r['problem_identity']!=p.identity or r['actor_tensor_sha256']!=SL_TENSOR_SHA256 or r['H_identity']!=H_ID:raise ValueError('FrozenSL reference identity')
    return r
def logits(actor,env,e,budget):
    t=time.perf_counter();z=actor.score(env,e);budget.counts['score_calls']+=1;budget.counts['score_s']+=time.perf_counter()-t
    mask=env.mask()
    if not torch.isfinite(z[torch.tensor(mask)]).all() or not torch.isneginf(z[torch.tensor(~mask)]).all():raise ValueError('Logit/mask contract')
    return z
def encode(actor,p,budget):
    t=time.perf_counter();e=actor.encode(p);budget.counts['encode_calls']+=1;budget.counts['encode_s']+=time.perf_counter()-t;return e

def select_states(pool,actor,budget):
    states=[];missing=[];tasktimes=[]
    for task in read(HOME/'TASKS.json')['records']:
        budget.check();t0=time.perf_counter();p,evidence=pool.get(task['id']);input_s=time.perf_counter()-t0;r=reference(p,task['id'])
        if r['raw_route']!=[0,*[x+1 for x in r['actions']],0]:raise ValueError('Saved SL action-route identity')
        env=Environment(p,no_terminal);eligible=defaultdict(list);selectstart=time.perf_counter()
        for t,a in enumerate(r['actions']):
            budget.check();mask=env.mask();u=1-float(env.remaining.sum())/len(p.C)
            bi=min((3*(len(p.C)-int(env.remaining.sum())))//len(p.C),2)
            if mask.sum()>=2:eligible[bi].append((abs(u-(1/6,1/2,5/6)[bi]),t,u,copy.deepcopy(env.prefix)))
            if t<len(r['actions'])-1:env.step(a+1);budget.counts['selection_prefix_steps']+=1
        for si,stage in enumerate(STAGES):
            if not eligible[si]:missing.append(dict(task_id=task['id'],stage=stage,reason='No before-action state in fixed progress bin with>=2 legal actions'));continue
            _,t,u,prefix=min(eligible[si],key=lambda x:(x[0],x[1]));env=prefix_env(p,prefix,no_terminal)
            z=logits(actor,env,encode(actor,p,budget),budget);legal=np.flatnonzero(env.mask()).tolist();rank=sorted(legal,key=lambda j:(-float(z[j]),j))
            if rank[0]!=r['actions'][t]:raise ValueError('Current FrozenSL greedy action differs from exact saved route')
            states.append(dict(state_id=task['id']+':'+stage,task_id=task['id'],stage=stage,size=task['size'],physical_key=task['physical_key'],
                problem_identity=p.identity,parent_identity=p.parent.identity,source=evidence,step=t,progress=u,prefix_local=prefix,
                prefix_global=[int(p.candidate_ids[n-1]) for n in prefix],remaining_tasks_global=p.task_ids[env.remaining].tolist(),mask_bits_b64=packed(env.mask()),
                fingerprint=fingerprint(p,env),top1=rank[0]+1,top2=rank[1]+1,top_global=[int(p.candidate_ids[j]) for j in rank[:2]],
                logits=[float(z[j]) for j in rank[:2]],F_sha256=array_hash(p.F),D_sha256=array_hash(p.D),actor_tensor_sha256=SL_TENSOR_SHA256,H_identity=H_ID))
        tasktimes.append(dict(task_id=task['id'],input_restore_s=input_s,state_selection_s=time.perf_counter()-selectstart))
    replicates=[]
    for size in ('10x10','4x3','6x6','8x8'):
        used=set()
        for stage in ('early','late'):
            choices=sorted([s for s in states if s['size']==size and s['stage']==stage and s['physical_key'] not in used],key=lambda s:(s['physical_key'],s['state_id']))
            if choices:replicates.append(choices[0]['state_id']);used.add(choices[0]['physical_key'])
    put(HOME/'SELECTED_STATES.json',dict(states=states,missing=missing,task_timings=tasktimes,repeat_state_ids=replicates,
        actor_tensor_sha256=SL_TENSOR_SHA256,source='Saved exact FrozenSL training routes; current mask/prefix and greedy rank verified; no teacher/cost selection',actual_states=len(states)))
    budget.save();return states,replicates

def teacher_gap(pool,task):
    p,_=pool.get(task['id']);tid=task['id']
    if tid in pool.full:
        e=read(D5/'FULL_TEACHER_MANIFEST.json')['records'][tid]
        if sha(ROOT/e['path'])!=e['sha256']:raise ValueError('Existing full strong teacher')
        label=gzread(ROOT/e['path']);actions=label['actions_local_nodes'];expected=label['problem_identity'];source=e
        raw=label['metrics']['raw_J'];teacherH=label['metrics']['J']
    else:
        source=task['source']
        if sha(ROOT/source['path'])!=source['sha256']:raise ValueError('Existing local teacher')
        label=read(ROOT/source['path']);actions=label['actions'];expected=label['view_identity'];raw=label['H_result']['raw_J'];teacherH=label['teacher_J']
    if expected!=p.identity:raise ValueError('Same-problem teacher identity')
    computed=p.metrics([0,*actions,0])['J']
    if abs(computed-teacherH)>1e-7:raise ValueError('Existing teacher cost mismatch')
    sl=reference(p,tid)['metrics']['J']
    return dict(task_id=tid,physical_key=task['physical_key'],size=task['size'],FrozenSL_H_J=sl,strong_teacher_H_J=teacherH,strong_teacher_native_J=raw,
        gap=sl-teacherH,relative_gap=(sl-teacherH)/sl,teacher_source=source,no_new_teacher_or_H=True)

def rollout(pool,actor,s,branch,rep,budget):
    rid=f'{s["state_id"]}:{branch}:repeat{rep}';tick=budget.begin(rid);stats=Counter();internal=None
    try:
        p,evidence=pool.get(s['task_id'])
        from rc3_nav_value.r1.oracle import _instrument,_empty_internal
        internal=_empty_internal()
        def terminal(problem,raw):
            budget.check();budget.counts['H_calls']+=1
            with _instrument(internal):return finish(problem,raw,stats)
        prefix_tick=time.perf_counter();env=prefix_env(p,s['prefix_local'],terminal);prefix_s=time.perf_counter()-prefix_tick
        if p.identity!=s['problem_identity'] or fingerprint(p,env)!=s['fingerprint']:raise ValueError('Counterfactual prefix/state identity')
        masks=[];suffix=[];before=tensor_digest(actor.state_dict());e=encode(actor,p,budget);z=logits(actor,env,e,budget)
        rank=sorted(np.flatnonzero(env.mask()).tolist(),key=lambda j:(-float(z[j]),j))
        if [rank[0]+1,rank[1]+1]!=[s['top1'],s['top2']]:raise ValueError('Counterfactual action ranking changed')
        chosen=s['top1'] if branch=='baseline' else s['top2'];masks.append(packed(env.mask()));env.step(chosen);suffix.append(chosen)
        while not env.done:
            budget.check();z=logits(actor,env,e,budget);node=int(z.argmax())+1;masks.append(packed(env.mask()));env.step(node);suffix.append(node)
        r=env.result
        if before!=SL_TENSOR_SHA256 or tensor_digest(actor.state_dict())!=before:raise ValueError('Frozen Actor changed')
        rawplan=export(p,r['raw_route'],'R07_'+branch+'_raw');rawplan['validation']=validate(p,rawplan)
        if not rawplan['validation']['valid_model_plan'] or not r['plan']['validation']['valid_model_plan']:raise ValueError('Complete proxy constraint violation')
        if abs(math.fsum(env.rewards)+r['J'])>1e-7:raise ValueError('Complete H return mismatch')
        if branch=='baseline':
            old=reference(p,s['task_id'])
            if r['raw_route']!=old['raw_route'] or r['route']!=old['route'] or any(abs(r[k]-old['metrics'][k])>1e-7 for k in ('J','L','K','raw_J','raw_L','raw_K')):raise ValueError('Baseline FrozenSL/H not reproducible')
        record=dict(request_id=rid,run_id=RUN,scientific_identity=identity(),state_id=s['state_id'],task_id=s['task_id'],branch=branch,repeat=rep,
            problem_identity=p.identity,physical_key=s['physical_key'],size=s['size'],stage=s['stage'],progress=s['progress'],
            prefix_local=s['prefix_local'],state_fingerprint=s['fingerprint'],forced_action=chosen,forced_global_candidate=int(p.candidate_ids[chosen-1]),
            continuation='Exact frozenSL greedy after forced action',actor_tensor_sha256=before,H_identity=H_ID,source=evidence,
            suffix_actions_local=suffix,mask_bits_b64=masks,raw_route=r['raw_route'],route=r['route'],raw_plan=rawplan,plan=r['plan'],
            metrics={k:r[k] for k in ('J','L','K','raw_J','raw_L','raw_K')},H_s=r['H_s'],prefix_restore_s=prefix_s,counts=dict(stats),internal_search=internal,
            raw_semantics='Complete frozen-policy raw route before identical common H',valid=True)
        ref=budget.complete(rid,record,tick);return dict(**ref,metrics=record['metrics'],H_s=r['H_s'])
    except BaseException as ex:budget.fail(rid,ex);raise
    finally:
        if internal is not None:budget.internal(internal);budget.save()

def run(budget):
    pool=Pool(budget);actor=load_frozen_sl();actor.eval().requires_grad_(False)
    if tensor_digest(actor.state_dict())!=SL_TENSOR_SHA256:raise ValueError('Exact SL256 only')
    with torch.no_grad():
        states,replicates=select_states(pool,actor,budget)
        # Teacher values are read only AFTER the immutable task/state selection.
        gaps=[teacher_gap(pool,t) for t in read(HOME/'TASKS.json')['records']];put(HOME/'TEACHER_GAPS.json',dict(rows=gaps))
        results={}
        for i,s in enumerate(states):
            result=dict(state=s,baseline=rollout(pool,actor,s,'baseline',0,budget),alternative=rollout(pool,actor,s,'alternative',0,budget))
            results[s['state_id']]=result;put(HOME/'pairs'/(s['state_id'].replace(':','_')+'.json'),result)
            if (i+1)%12==0:state(status='ROLLOUT',complete_primary_states=i+1);print(dict(primary_states=i+1,requests=budget.counts['requests']),flush=True)
        repeatrows=[]
        for sid in replicates:
            s=results[sid]['state']
            for branch in ('baseline','alternative'):
                again=rollout(pool,actor,s,branch,1,budget);first=unarchive(results[sid][branch]['record']);second=unarchive(again['record'])
                keys=('raw_route','route','metrics','mask_bits_b64','suffix_actions_local','state_fingerprint')
                if any(first[k]!=second[k] for k in keys):raise ValueError('Actual duplicate same-state rollout is not deterministic')
                repeatrows.append(dict(state_id=sid,branch=branch,exact_actions_masks_raw_H_metrics=True,record=again))
        put(HOME/'REPRODUCIBILITY.json',dict(status='PASS',actual_repeated_requests=len(repeatrows),rows=repeatrows))
    budget.save();state(status='SCIENCE_COMPLETE',complete_primary_states=len(states),missing_states=len(read(HOME/'SELECTED_STATES.json')['missing']),closed=True)
