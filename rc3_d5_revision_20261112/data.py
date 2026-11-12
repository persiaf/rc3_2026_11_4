import copy,time,math
from collections import Counter
import numpy as np
from rc3_conservative_refinement.saved_inputs import SavedInputs,spec_key
from rc3_global_local.environment import Environment
from rc3_global_local.problem import array_hash
from .common import ROOT,HOME,H_ID,read,put,sha,gzput,gzread,state

class Pool:
    """Only registered training/development snapshots; immutable parameter-free cache."""
    def __init__(self,budget):
        self.budget=budget;self.inputs=SavedInputs();self.cache={};self.labels={};self.prepare_s=0.
        self.full={r['id']:r for r in read(HOME/'FULL_TASK_REGISTRATION.json')['records']}
        self.local={r['id']:r for r in read(HOME/'ORIGINAL_LOCAL_TEACHERS.json')['records']}
        self.cases=read(HOME/'REVISION_PROTOCOL.json')['evaluation']['development']
        self.entries={**self.full,**self.local,**{r['id']:r for r in self.cases}}
        for e in self.entries.values():self.inputs.allowed[spec_key(e['spec'])]=copy.deepcopy(e['spec'])
    def get(self,tid):
        self.budget.check()
        if tid not in self.entries:raise ValueError('Task absent from registered data revision')
        if tid in self.cache:self.budget.counts['immutable_input_cache_hits']+=1;return self.cache[tid]
        tick=time.perf_counter();entry=self.entries[tid];p,evidence=self.inputs.view(entry['spec'])
        if evidence['physical_key']!=entry['physical_key']:raise ValueError('Physical parent identity')
        if tid in self.full and p.parent.identity!=entry['expected_parent_identity']:raise ValueError('Full parent source identity')
        if p.C.flags.writeable or p.D.flags.writeable:raise ValueError('Input snapshot must be immutable')
        self.cache[tid]=(p,evidence);self.prepare_s+=time.perf_counter()-tick
        self.budget.counts['immutable_input_preparations']+=1;return self.cache[tid]
    def teacher(self,tid):
        p,evidence=self.get(tid)
        if tid not in self.labels:
            if tid in self.local:
                p0,actions,source=self.inputs.teacher(self.local[tid]['teacher_index'])
                if p0.identity!=p.identity:raise ValueError('Local teacher problem identity')
                self.labels[tid]=(actions,source['teacher_source'])
            elif tid in self.full:
                mf=read(HOME/'FULL_TEACHER_MANIFEST.json');e=mf['records'][tid]
                if mf['status']!='PASS' or not e['current_environment_replay_passed']:raise ValueError('Full teacher not admitted')
                if sha(ROOT/e['path'])!=e['sha256']:raise ValueError('Full teacher bytes')
                label=gzread(ROOT/e['path'])
                if label['problem_identity']!=p.identity or label['H_identity']!=H_ID:raise ValueError('Full teacher source')
                self.labels[tid]=(label['actions_local_nodes'],dict(path=e['path'],sha256=e['sha256'],method='Detour16+uniform_H'))
            else:raise ValueError('No teacher allowed for evaluation tasks')
        return p,self.labels[tid][0],self.labels[tid][1]

def replay_teacher(p,actions):
    # Validation replay invokes original mask/step, without another teacher/H.
    env=Environment(p,lambda problem,route:problem.metrics(route));steps=[]
    for node in actions:
        mask=env.mask()
        if not 1<=node<=len(mask) or not bool(mask[node-1]):raise ValueError('Teacher action illegal under current mask')
        before=int(env.remaining.sum());event=env.step(int(node))
        if int(env.remaining.sum())>=before:raise ValueError('Teacher action has no positive gain')
        steps.append(dict(node=int(node),legal=True,served=event['served_tasks_global'],remaining=len(event['remaining_tasks_global'])))
    if not env.done or env.remaining.any():raise ValueError('Full teacher leaves tasks uncovered')
    return steps

def run_data(budget):
    from rc3_global_local_impl_review.endpoint_baselines import run_endpoint
    from rc3_global_local.terminal import finish
    from rc3_nav_value.r1.oracle import _instrument,_empty_internal
    pool=Pool(budget);path=HOME/'FULL_TEACHER_MANIFEST.json'
    manifest=read(path) if path.exists() else dict(schema='d5-exact-full-teachers-r1',status='IN_PROGRESS',records={},selection_registration_sha256=sha(HOME/'FULL_TASK_REGISTRATION.json'))
    for tid,entry in pool.full.items():
        if tid in manifest['records']:
            e=manifest['records'][tid]
            if sha(ROOT/e['path'])!=e['sha256']:raise ValueError('Saved full teacher changed during resume')
            continue
        state(phase='DATA',status='GENERATING_FULL_TEACHERS',completed_full_teachers=len(manifest['records']),current_task=tid)
        native_id='teacher:'+tid+':native';tick=budget.begin_request('teacher',native_id);stats=Counter()
        try:
            prep=time.perf_counter();p,evidence=pool.get(tid);prepare=time.perf_counter()-prep
            budget.debit('teacher_native_attempts',limit=104);budget.counts['native_requests']+=1;budget.counts['new_teacher_requests']+=1;budget.save()
            native,info=run_endpoint(p,'detour16',time_limit_s=None,reject_fallback=True,counters=stats)
            raw=list(info['local_route'])
            rec=dict(request_id=native_id,purpose='teacher',task_id=tid,problem_identity=p.identity,spec=entry['spec'],physical_key=entry['physical_key'],
                plan=native,raw_route=raw,metrics=p.metrics(raw),counts=dict(stats),internal_search=info['internal_search'],
                raw_semantics='Detour16 native final before explicitly added common H; not a neural no-H route',prepare_s=prepare,source_pipeline=info['source_pipeline'])
            budget.internal(info['internal_search']);nref,ntime=budget.complete('teacher',native_id,rec,tick,'data')
        except BaseException as exc:budget.failed('teacher',native_id,exc);raise
        hid='teacher:'+tid+':uniformH';tick=budget.begin_request('teacher',hid);hstats=Counter();internal=_empty_internal()
        try:
            budget.debit('teacher_uniform_H_attempts',limit=104);budget.counts['full_H_calls']+=int(bool(raw[1:-1]));budget.save()
            with _instrument(internal):result=finish(p,raw,hstats)
            actions=[int(x) for x in result['route'][1:-1]];steps=replay_teacher(p,actions)
            if abs(result['J']-(result['L']+.75*result['K']))>1e-7:raise ValueError('Teacher units')
            label=dict(schema='d5-full-teacher-r1',task_id=tid,problem_identity=p.identity,parent_identity=p.parent.identity,
                physical_key=entry['physical_key'],spec=entry['spec'],candidate_ids=p.candidate_ids.tolist(),task_ids=p.task_ids.tolist(),
                F_sha256=array_hash(p.parent.F),A_sha256=array_hash(p.parent.A),D_sha256=array_hash(p.D),effective_F_sha256=array_hash(p.F),
                input_sources=evidence['input_source_files'],window=p.window,endpoints=dict(entry=p.entry,exit=p.exit),
                teacher_method='Original endpoint Detour16 horizon16 then frozen common H',H_identity=H_ID,
                source_binding_sha256=sha(HOME/'DATA_SOURCE_BINDING.json'),raw_route=raw,route=result['route'],raw_plan=native,plan=result['plan'],
                actions_local_nodes=actions,actions_global_candidates=result['plan']['visit_order_candidate_ids'],
                metrics={k:result[k] for k in ('J','L','K','raw_J','raw_L','raw_K')},environment_replay=steps,
                replay_complete=True,counts=dict(hstats),internal_search=internal,native_record=nref,native_request_s=ntime)
            href,htime=budget.complete('teacher',hid,dict(label,purpose='teacher',request_id=hid),tick,'data')
            target=HOME/'full_teachers'/(tid+'.json.gz');gzput(target,label)
            manifest['records'][tid]=dict(path=target.relative_to(ROOT).as_posix(),sha256=sha(target),problem_identity=p.identity,
                parent_identity=p.parent.identity,physical_key=entry['physical_key'],F_sha256=label['F_sha256'],A_sha256=label['A_sha256'],D_sha256=label['D_sha256'],
                current_environment_replay_passed=True,source='NEW_STRONG_TEACHER',native_request=nref,uniform_H_request=href)
            put(path,manifest,path.exists());budget.counts['full_teachers_completed']+=1;budget.internal(internal);budget.save()
            print(dict(phase='DATA',completed=len(manifest['records']),parents=len({e['physical_key'] for e in manifest['records'].values()}),wall_s=budget.elapsed()),flush=True)
        except BaseException as exc:budget.failed('teacher',hid,exc);raise
    if len(manifest['records'])!=96 or len({e['physical_key'] for e in manifest['records'].values()})!=96:raise ValueError('Full96 incomplete')
    manifest.update(status='PASS',full_teachers=96,physical_parents=96,new_teachers=96,reused_teachers=0,H_identity=H_ID)
    put(path,manifest,True)
    joint=dict(schema='d5-joint-352-r1',full_manifest=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path)),
        original_local_index=read(HOME/'ORIGINAL_LOCAL_TEACHERS.json')['index'],full=pool.full,local=pool.local,
        unique_full=96,unique_local=256,unique_total=352,old256_unchanged=True,validated_complete_full=96)
    put(HOME/'JOINT_DATA_INDEX.json',joint)
    state(phase='ENGINEERING',status='DATA_PASS',completed_full_teachers=96,full_physical_parents=96,next_action='Bind scientific sources and run targeted remaining-P0 acceptance')
