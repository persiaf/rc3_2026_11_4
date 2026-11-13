import time,math
from collections import Counter
import numpy as np
import torch
from rc3_d5_revision_20261112.data import Pool as OldPool
from rc3_global_local.environment import Environment
from rc3_global_local.terminal import finish,export
from rc3_global_local.validation import validate
from rc3_conservative_refinement.runtime import load_frozen_sl,tensor_digest,SL_TENSOR_SHA256
from rc3_r08_joint_decision_design_20261113.adapters import SavedEvidence,recorded_route
from rc3_r08_joint_decision_design_20261113.contracts import ProblemSnapshot,JointState
from .common import ROOT,HOME,D5,RUN,H_ID,read,put,identity,state
from .menu import build_menu
from .lower import solve

class Pool(OldPool):
    def __init__(self,budget):
        super().__init__(budget);ids={r['id'] for r in read(HOME/'TASKS.json')['records']}
        self.entries={k:v for k,v in self.entries.items() if k in ids};self.full={k:v for k,v in self.full.items() if k in ids};self.local={k:v for k,v in self.local.items() if k in ids};self.cases=[]

def snapshot(p,evidence):
    ix=np.ix_(p.task_ids,p.candidate_ids,np.arange(2));scene=p.parent.scene
    return ProblemSnapshot(p.identity,p.parent.identity,evidence['physical_key'],tuple(map(int,p.task_ids)),tuple(map(int,p.candidate_ids)),
        p.parent.F[ix],p.parent.A[ix],p.D,p.entry,p.exit,
        tuple(tuple(float(scene['candidates'][int(j)][k]) for k in ('x','y','theta','h')) for j in p.candidate_ids))

def certified_raw(p,raw,label,counts):
    tick=time.perf_counter();counts['raw_export_attempts']+=1;plan=export(p,raw,label);counts['raw_validation_attempts']+=1;plan['validation']=validate(p,plan)
    if not plan['validation']['valid_model_plan']:raise ValueError(plan['validation']['errors'])
    counts['raw_validation_pass']+=1;return plan,time.perf_counter()-tick

def anchor(pool,actor,task,budget,reader,refmeta):
    rid=task['id']+':anchor';tick=budget.begin(rid);internal=None;stats=Counter()
    try:
        prep=time.perf_counter();p,evidence=pool.get(task['id']);prepare_s=time.perf_counter()-prep
        old=reader.json(refmeta['path'],refmeta['sha256'])
        if old['problem_identity']!=p.identity or old['actor_tensor_sha256']!=SL_TENSOR_SHA256 or old['H_identity']!=H_ID:raise ValueError('Exact frozen reference source')
        from rc3_nav_value.r1.oracle import _instrument,_empty_internal
        internal=_empty_internal()
        def terminal(problem,raw):
            budget.counts['H_calls']+=1
            with _instrument(internal):return finish(problem,raw,stats)
        env=Environment(p,terminal);before=tensor_digest(actor.state_dict());encstart=time.perf_counter();encoded=actor.encode(p);budget.counts['encode_calls']+=1;budget.counts['encode_s']+=time.perf_counter()-encstart;actions=[]
        while not env.done:
            budget.check();scorestart=time.perf_counter();z=actor.score(env,encoded);budget.counts['score_calls']+=1;budget.counts['score_s']+=time.perf_counter()-scorestart
            mask=env.mask()
            if not torch.isfinite(z[torch.tensor(mask)]).all() or not torch.isneginf(z[torch.tensor(~mask)]).all():raise ValueError('FrozenSL mask/finite')
            node=int(z.argmax())+1;actions.append(node);env.step(node)
        r=env.result
        if tensor_digest(actor.state_dict())!=before or before!=SL_TENSOR_SHA256:raise ValueError('FrozenSL mutated')
        if r['raw_route']!=old['raw_route'] or r['route']!=old['route'] or any(abs(r[k]-old['metrics'][k])>1e-7 for k in ('J','L','K','raw_J','raw_L','raw_K')):raise ValueError('Actual anchor not exact saved SL256')
        if abs(math.fsum(env.rewards)+r['J'])>1e-7:raise ValueError('Anchor complete return')
        rawplan,rawcheck=certified_raw(p,r['raw_route'],'R08B_FrozenSL_raw',stats)
        record=dict(request_id=rid,run_id=RUN,purpose='reference',task_id=task['id'],size=task['size'],physical_key=task['physical_key'],problem_identity=p.identity,
            source=evidence,actor_tensor_sha256=before,H_identity=H_ID,raw_route=r['raw_route'],route=r['route'],raw_plan=rawplan,plan=r['plan'],
            metrics={k:r[k] for k in ('J','L','K','raw_J','raw_L','raw_K')},H_s=r['H_s'],input_prepare_s=prepare_s,raw_export_validate_s=rawcheck,actions=actions,
            counts=dict(stats),internal_search=internal,valid=True)
        ptr=budget.complete(rid,record,tick);typed=snapshot(p,evidence);s=JointState(typed,recorded_route(rawplan,rid),1,16,SL_TENSOR_SHA256)
        return p,s,record,ptr
    except BaseException as exc:budget.fail(rid,exc);raise
    finally:
        if internal is not None:budget.internal(internal);budget.save()

def edit_request(p,s,action,task,anchor_record,anchor_ptr,budget):
    rid=f"{task['id']}:edit{action['menu_index']:02}";tick=budget.begin(rid);internal=None;stats=Counter()
    try:
        from rc3_nav_value.r1.oracle import _instrument,_empty_internal
        internal=_empty_internal()
        with _instrument(internal):
            raw,detail=solve(p,s,action);rawplan,rawcheck=certified_raw(p,raw,'R08B_joint_raw',stats)
            budget.check();budget.counts['H_calls']+=1;r=finish(p,raw,stats)
        if not r['plan']['validation']['valid_model_plan']:raise ValueError('Final H invalid')
        record=dict(request_id=rid,run_id=RUN,purpose='joint_edit_opportunity',scientific_identity=identity(),task_id=task['id'],size=task['size'],physical_key=task['physical_key'],
            problem_identity=p.identity,H_identity=H_ID,actor_tensor_sha256=SL_TENSOR_SHA256,anchor=anchor_ptr,action=action,
            before_raw_metrics={k:anchor_record['metrics'][k] for k in ('raw_J','raw_L','raw_K')},before_H_metrics={k:anchor_record['metrics'][k] for k in ('J','L','K')},
            raw_route=r['raw_route'],route=r['route'],raw_plan=rawplan,plan=r['plan'],metrics={k:r[k] for k in ('J','L','K','raw_J','raw_L','raw_K')},
            lower=detail,H_s=r['H_s'],raw_export_validate_s=rawcheck,H_export_s=r['export_s'],H_validate_s=r['validation_s'],counts=dict(stats),internal_search=internal,
            fallback_used=False,opportunity_only=True,valid=True)
        return budget.complete(rid,record,tick)
    except BaseException as exc:budget.fail(rid,exc);raise
    finally:
        if internal is not None:budget.internal(internal);budget.save()

def run(budget):
    pool=Pool(budget);reader=SavedEvidence();refs=read(D5/'REFERENCE_MANIFEST.json')['records'];actor=load_frozen_sl().eval().requires_grad_(False)
    tasks=read(HOME/'TASKS.json')['records']
    if len(tasks)!=48 or len({t['physical_key'] for t in tasks})!=48:raise ValueError('Fixed48 distinct parent inputs')
    with torch.no_grad():
        for k,task in enumerate(tasks):
            state(status='RUNNING',current_task=task['id'],completed_tasks=k)
            p,s,base,bptr=anchor(pool,actor,task,budget,reader,refs[task['id']]);menu=build_menu(s,base['raw_plan']['assignments']);put(HOME/'menus'/(task['id']+'.json'),dict(task=task,problem_identity=p.identity,**menu))
            budget.counts['menu_build_calls']+=1;budget.counts['menu_build_s']+=menu['menu_build_s'];budget.counts['missing_menu_slots']+=menu['missing_slots'];budget.save();results=[]
            for action in menu['actions']:results.append(dict(action=action,**edit_request(p,s,action,task,base,bptr,budget)))
            put(HOME/'tasks'/(task['id']+'.json'),dict(task=task,anchor=bptr,menu_sha256=menu['menu_sha256'],menu_build_s=menu['menu_build_s'],results=results))
            print(dict(task=task['id'],complete_tasks=k+1,edits=len(results),requests=budget.counts['requests']),flush=True)
    if tensor_digest(actor.state_dict())!=SL_TENSOR_SHA256:raise ValueError('Final frozen Actor source')
    budget.save();state(status='SCIENCE_COMPLETE',closed=True,completed_tasks=48,C_D_E='NOT_RUN_NOT_AUTHORIZED')
