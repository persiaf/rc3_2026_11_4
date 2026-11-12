import os,time,copy,math,datetime
import torch
import numpy as np
from rc3_conservative_refinement.runtime import tensor_digest
from rc3_conservative_refinement.trainer import checkpoint_equal
from rc3_decoder_program_20261111.actor import ALLOW,export as actor_export
from .common import ROOT,HOME,RUN,read,put,sha,state,identity,gzput,gzread,unarchive,append
from .data import Pool,replay_teacher
from .runtime import new_actor,frozen_actor,policy_request,teacher_episode,reference_key,References,sample_generator,generator_text,restore_generator,recover_episode
from .training import Trainer
from . import evaluation

def fast_budget(budget):
    # Same immutable limits as Budget.check, parsed once rather than for every action.
    limits=read(HOME/'REVISION_PROTOCOL.json')['wall_limits'];pool='training' if budget.pool in ('data','reference','science') else budget.pool
    def check():
        e=budget.elapsed()
        if budget.base['wall']['total_s']+e>=345580 or budget.base['wall'].get(pool+'_s',0)+e>=limits[pool]-20:raise TimeoutError('Approved cumulative pool boundary')
        if os.environ.get('D5_JOB_DEADLINE') and time.time()>=float(os.environ['D5_JOB_DEADLINE'])-20:raise TimeoutError('Supervised job boundary')
    budget.check=check

def validate_joint(budget):
    pool=Pool(budget);rows=[]
    for tid in pool.local:
        p,actions,source=pool.teacher(tid);steps=replay_teacher(p,actions)
        budget.counts['existing_label_validation_replays']+=1
        rows.append(dict(task_id=tid,problem_identity=p.identity,source=source,actions=len(steps),all_current_masks_legal=True,complete=True))
    full=read(HOME/'FULL_TEACHER_MANIFEST.json')
    if full['status']!='PASS' or len(full['records'])!=96:raise ValueError('Complete teacher96 not admitted')
    for entry in full['records'].values():
        if sha(ROOT/entry['path'])!=entry['sha256'] or not entry['current_environment_replay_passed']:raise ValueError('Full teacher identity/replay')
    put(HOME/'JOINT_DATA_ADMISSION.json',dict(status='PASS',full=96,local=256,total=352,local_current_environment_replays=rows,
        full_replays_source=dict(path='FULL_TEACHER_MANIFEST.json',sha256=sha(HOME/'FULL_TEACHER_MANIFEST.json')),
        no_new_teacher_native_H_navigation_or_scene=True,old256_bytes_unchanged=True))
    budget.save();state(phase='ENGINEERING',status='JOINT352_ADMITTED',next_action='Targeted full8/resampled8 acceptance')

def acceptance(budget):
    if (HOME/'RECOVERY_ACCEPTANCE_PROTOCOL.json').exists():
        from .recover_acceptance import run
        return run(budget)
    if read(HOME/'JOINT_DATA_ADMISSION.json')['status']!='PASS':raise PermissionError('Joint352 labels not admitted')
    fast_budget(budget);pool=Pool(budget);seed=2026111101;ids=read(HOME/'schedules'/f'{seed}.json')['batches'][0]
    if read(HOME/'TARGETED_ACCEPTANCE_PROTOCOL.json')['new_request_cap']!=40:raise ValueError('Exact targeted protocol')
    start=time.perf_counter();loader=[]
    for tid in ids:
        p,actions,source=pool.teacher(tid);steps=replay_teacher(p,actions);loader.append(dict(task_id=tid,problem_identity=p.identity,actions=len(steps),source=source))
    input_setup=pool.prepare_s;reference_setup=time.perf_counter();frozen=frozen_actor(budget);reference_setup=time.perf_counter()-reference_setup
    mf=dict(schema='d5-frozenSL-reference-r1',status='IN_PROGRESS',records={},engineering_first_reference_s=0.,engineering_input_setup_s=input_setup,engineering_model_setup_s=reference_setup)
    refs={};ref_times=[]
    for slot,tid in enumerate(ids):
        out=policy_request(budget,pool,tid,f'acceptance:reference:{slot}',frozen,purpose='reference',folder='references/records',greedy=True,method='FrozenSL')
        path=HOME/'references'/(tid+'.json.gz');gzput(path,out['record']);p,_=pool.get(tid)
        mf['records'][tid]=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),key=reference_key(p),first_compute_s=out['full_request_s'],source='TARGETED_ACCEPTANCE_FIXED_SL')
        refs[tid]=out['record']['metrics']['J'];ref_times.append(out['full_request_s'])
    mf['engineering_first_reference_s']=sum(ref_times)+input_setup+reference_setup;put(HOME/'REFERENCE_MANIFEST.json',mf)
    actor=new_actor('D5-PG',budget);g=sample_generator(seed);trainer=Trainer(actor,budget,seed,g,True);folder=HOME/'acceptance/PG'
    trainer.save_current(folder,True);episodes=[];batch_start=time.perf_counter();before=tensor_digest(actor.state_dict())
    for slot,tid in enumerate(ids):
        ep=policy_request(budget,pool,tid,f'acceptance:PG:0:{slot}',actor,purpose='engineering',folder='acceptance/records',generator=g,seed=seed)['episode']
        ep['reference_J_H']=refs[tid];episodes.append(ep)
    pg_update=trainer.update(episodes);trainer.save_current(folder,True);batch_s=time.perf_counter()-batch_start
    if actor.frozen_digest()!=actor.initial_frozen_digest:raise ValueError('Complete batch changed frozen tensors')
    if not all(pg_update['gradient_norms'][n] is not None and math.isfinite(pg_update['gradient_norms'][n]) for n in ALLOW):raise ValueError('Complete batch gradient whitelist')
    slactor=new_actor('D5-SL',budget);st=Trainer(slactor,budget,seed,sample_generator(seed),True)
    sle=[teacher_episode(budget,pool,slactor,tid,f'acceptance:SL:{slot}') for slot,tid in enumerate(ids)]
    sl_update=st.update(sle)
    if slactor.frozen_digest()!=slactor.initial_frozen_digest:raise ValueError('Full/local supervised batch changed frozen tensors')
    # Actual restored sampling is compared, not just a reused captured trajectory.
    checkpoint=trainer.save_current(folder);clone=Trainer(new_actor('D5-PG',budget),budget,seed,sample_generator(seed),True);clone.restore_current(checkpoint)
    if not checkpoint_equal(trainer.optimizer.state_dict(),clone.optimizer.state_dict()):raise ValueError('Restored Adam')
    next_ids=read(HOME/'schedules'/f'{seed}.json')['batches'][1];original_next=[];restored_next=[]
    for slot,tid in enumerate(next_ids):
        if tid not in refs:
            out=policy_request(budget,pool,tid,f'acceptance:resume_reference:{slot}',frozen,purpose='reference',folder='references/records',greedy=True,method='FrozenSL')
            path=HOME/'references'/(tid+'.json.gz');gzput(path,out['record']);p,_=pool.get(tid)
            mf['records'][tid]=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),key=reference_key(p),first_compute_s=out['full_request_s'],source='TARGETED_ACCEPTANCE_FIXED_SL')
            refs[tid]=out['record']['metrics']['J'];mf['engineering_first_reference_s']+=out['full_request_s']
        a=policy_request(budget,pool,tid,f'acceptance:resume:original:{slot}',actor,purpose='engineering',folder='acceptance/records',generator=g,seed=seed)
        b=policy_request(budget,pool,tid,f'acceptance:resume:restored:{slot}',clone.actor,purpose='engineering',folder='acceptance/records',generator=clone.sampling,seed=seed)
        for key in ('actions','mask_bits_b64','raw_route','route','metrics','sample_log_prob'):
            if a['record'][key]!=b['record'][key]:raise ValueError('Resampled continuation differs: '+key)
        a['episode']['reference_J_H']=refs[tid];b['episode']['reference_J_H']=refs[tid];original_next.append(a['episode']);restored_next.append(b['episode'])
    put(HOME/'REFERENCE_MANIFEST.json',mf,True)
    if generator_text(g)!=generator_text(clone.sampling):raise ValueError('Sampling stream after actual continuation')
    trainer.update(original_next);clone.update(restored_next)
    if tensor_digest(actor.state_dict())!=tensor_digest(clone.actor.state_dict()) or not checkpoint_equal(trainer.optimizer.state_dict(),clone.optimizer.state_dict()):raise ValueError('Actual resumed next update mismatch')
    teacher_units=[dict(task_id=x['task_id'],J_H=-x['complete_return'],full_return=x['complete_return']) for x in episodes]
    estimated_reference=352*(sum(ref_times)/len(ref_times)+input_setup/len(ids))+reference_setup
    estimated_training=4096*batch_s+estimated_reference+352*input_setup/len(ids)
    passed=estimated_training<=28800
    put(HOME/'TARGETED_ACCEPTANCE.json',dict(status='PASS' if passed else 'BUDGET_THROUGHPUT_BLOCKED',
        source_identity=identity(),teacher_loader=loader,complete_batch_ids=ids,gradient_norms=pg_update['gradient_norms'],
        SL_gradient_norms=sl_update['gradient_norms'],PG_parameters_changed=before!=pg_update['after'],frozen_bitwise=True,
        restored_own_sampling_actions_masks_raw_H_probabilities_exact=True,restored_next_Adam_and_model_exact=True,
        actual_PG_full8_batch_including_update_checkpoint_s=batch_s,reference_request_times_s=ref_times,
        unique_input_prepare_s=input_setup,estimated_4096_with_reference_and_input_s=estimated_training,
        per_stream_limit_s=28800,units=teacher_units,Actor_attempts_new=4,new_request_cap=40,
        measured_PG_batch_size=8,SL_batch_size=8,resumed_sampling_batch_size=8,
        old_engineering_resources_inherited=True,engineering_models_forbidden_for_scientific_initialization=True))
    if passed:state(phase='REFERENCE',status='TARGETED_ACCEPTANCE_PASS',next_action='Complete exact immutable FrozenSL references, then P1')
    else:closeout('TECHNICAL_BUDGET_INCOMPLETE','Measured complete8 throughput does not fit approved8h stream; no science started')
    print(dict(phase='ACCEPTANCE',status='PASS' if passed else 'BLOCKED',batch_s=batch_s,estimated4096_s=estimated_training),flush=True)

def references(budget):
    fast_budget(budget);pool=Pool(budget);actor=frozen_actor(budget);mf=read(HOME/'REFERENCE_MANIFEST.json')
    for tid in list(pool.full)+list(pool.local):
        if tid in mf['records']:
            e=mf['records'][tid]
            if sha(ROOT/e['path'])!=e['sha256']:raise ValueError('Existing exact reference changed')
            continue
        out=policy_request(budget,pool,tid,'reference:'+tid,actor,purpose='reference',folder='references/records',greedy=True,method='FrozenSL')
        path=HOME/'references'/(tid+'.json.gz');gzput(path,out['record']);p,_=pool.get(tid)
        mf['records'][tid]=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),key=reference_key(p),first_compute_s=out['full_request_s'],source='REFERENCE_PRECOMPUTE')
        put(HOME/'REFERENCE_MANIFEST.json',mf,True)
        if len(mf['records'])%32==0:print(dict(phase='REFERENCE',completed=len(mf['records']),job_s=budget.elapsed()),flush=True)
    mf.update(status='PASS',unique_references=352,reference_job=budget.job)
    put(HOME/'REFERENCE_MANIFEST.json',mf,True);state(phase='P1',status='READY_FOR_FORMAL_P1',next_action='Fresh exactSL256 D5-PG4096 then same-schedule SL-U4096')

def reference_allocation():
    m=read(HOME/'REFERENCE_MANIFEST.json');r=read(HOME/'RESOURCE_ACCOUNTING.json')
    return float(r['jobs'][m['reference_job']]['wall_s'])+m['engineering_first_reference_s']

def closeout(code,reason,**extra):
    state(phase='P6',status=code,closed=True,stop_reason=reason,**extra)
    put(HOME/'NEXT_STEP_DECISION.json',dict(status=code,reason=reason,scientific_identity=identity(),automatic_new_experiments=False,**extra),(HOME/'NEXT_STEP_DECISION.json').exists())

def save_export(t,folder,label):
    path=folder/(label+'.actor.pt');entryfile=folder/(label+'.actor.json')
    if entryfile.exists():return read(entryfile)
    if path.exists():
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload['role']!=t.actor.role or payload['tensor_sha256']!=tensor_digest(t.actor.state_dict()) or payload['engineering_only']:raise ValueError('Uncommitted endpoint export differs')
        entry=dict(path=str(path),sha256=sha(path),role=t.actor.role,engineering_only=False)
    else:entry=actor_export(path,t.actor,engineering=False)
    entry['path']=path.relative_to(ROOT).as_posix();put(entryfile,entry);return entry

def save_C_candidate(t,folder):
    # Two alternating atomic slots preserve the pre/post budget boundary.
    slot=folder/f'SL_C_candidate_{t.cursor%2}.actor.pt';temp=slot.with_name(slot.name+'.pending')
    package=dict(schema='rc3-decoder-program-actor-v1',role=t.actor.role,model=copy.deepcopy(t.actor.state_dict()),base=t.actor.base_identity,
        revision=t.actor.revision,tensor_sha256=tensor_digest(t.actor.state_dict()),engineering_only=False,critic_included=False)
    with temp.open('wb') as f:torch.save(package,f);f.flush();os.fsync(f.fileno())
    os.replace(temp,slot)
    return dict(path=slot.relative_to(ROOT).as_posix(),sha256=sha(slot),role=t.actor.role,engineering_only=False,cursor=t.cursor)

def train_stream(budget,pool,role,seed,*,sl_time_target=None,until_C=False):
    schedule=read(HOME/'schedules'/f'{seed}.json')['batches'];base=HOME/'streams'/str(seed)/role;base.mkdir(parents=True,exist_ok=True)
    setup_start=time.perf_counter();setup_wall=time.time()
    gen=sample_generator(seed);t=Trainer(new_actor(role,budget),budget,seed,gen,False);latest=base/'LATEST.json'
    if latest.exists():t.restore_current(read(latest))
    else:t.save_current(base,True)
    setup_s=max(time.perf_counter()-setup_start,time.time()-setup_wall)
    pending_file=base/'ACTIVE_BATCH.json'
    if pending_file.exists():
        pending=read(pending_file)
        old_job=pending.get('job')
        if old_job and old_job!=budget.job:
            receipt=read(HOME/'controls'/old_job/'SUPERVISOR_RECEIPT.json')
            if not receipt['root_exit_confirmed'] or receipt['job_active_processes_at_end']!=0:raise RuntimeError('Old stream job not confirmed exited')
            charges=pending.get('recovery_training_s',{})
            if old_job not in charges:
                ended=datetime.datetime.fromisoformat(receipt['finish_utc'].replace('Z','+00:00')).timestamp()
                charges[old_job]=max(0.,ended-pending['batch_started_wall_s'])
                pending['recovery_training_s']=charges;put(pending_file,pending,True)
            if pending['op']>t.cursor:t.train_s+=sum(charges.values())
            elif pending['op']==t.cursor:
                timer=read(base/'TRAINING_TIME.json') if (base/'TRAINING_TIME.json').exists() else {}
                if timer.get('cursor')!=t.cursor or timer.get('checkpoint_sha256')!=read(latest)['sha256']:
                    t.train_s=max(t.train_s,pending['stream_before_s']+charges[old_job])
    refs=None if role=='D5-SL' else References(pool,budget)
    summaryfile=base/'STREAM_SUMMARY.json';summary=read(summaryfile) if summaryfile.exists() else dict(role=role,seed=seed,updates=0,reference_allocation_s=0 if refs is None else reference_allocation(),endpoints={},SL_C=None)
    # A crash at an evaluation point must finish that exact point before any
    # later optimizer opportunity; never silently skip a registered endpoint.
    for point in (1024,2048,4096):
        if point>t.cursor:continue
        e=summary['endpoints'].get(str(point))
        if e is None:
            f=base/f'step{point:05}.actor.json'
            if f.exists():e=read(f)
            elif t.cursor==point:e=save_export(t,base,f'step{point:05}')
            else:raise RuntimeError('Missing earlier fixed endpoint; cannot substitute later weights')
            summary['endpoints'][str(point)]=e;put(summaryfile,summary,summaryfile.exists())
        evaluation.evaluate(budget,pool,'D5-SL-U' if role=='D5-SL' else role,seed,point,entry=e)
    if summary.get('complete') and (role!='D5-SL' or not until_C or summary.get('C_complete')):return summary
    t.train_s+=setup_s
    if t.cursor==0 and '0' not in summary['endpoints']:
        summary['endpoints']['0']=save_export(t,base,'step00000');put(summaryfile,summary,summaryfile.exists())
    max_updates=16384 if role=='D5-SL' and until_C else 4096
    if sl_time_target is None and role=='D5-SL':raise ValueError('SL stream requires prior same-seed PG time target')
    activefile=base/'ACTIVE_BATCH.json';cstatefile=base/'SL_C_TIME_STATE.json';cstate=read(cstatefile) if cstatefile.exists() else dict(entry=summary.get('SL_C'),train_s=None)
    prior_C=cstate['entry']
    while t.cursor<max_updates:
        if role=='D5-SL' and t.cursor>=4096 and t.train_s>=sl_time_target:break
        if t.train_s+summary['reference_allocation_s']>=28780:raise TimeoutError('Per-stream8h cumulative boundary with mandatory closeout reserve')
        op=t.cursor+1;ids=schedule[(op-1)%4096];start=time.perf_counter();start_w=time.time();before_hash=tensor_digest(t.actor.state_dict())
        root_check=budget.check;prior_stream_s=t.train_s
        def stream_check():
            root_check()
            if prior_stream_s+summary['reference_allocation_s']+max(time.perf_counter()-start,time.time()-start_w)>=28780:raise TimeoutError('Current full batch reached remaining8h stream boundary')
        budget.check=stream_check
        active=read(activefile) if activefile.exists() else None
        if active and active['op']<=t.cursor:active=None
        if active and (active['op']!=op or active['actor_tensor_sha256']!=before_hash or active['task_ids']!=ids):raise ValueError('Active batch identity mismatch')
        if active is None:
            active=dict(op=op,actor_tensor_sha256=before_hash,task_ids=ids,episodes=[],sampling_before=generator_text(gen),recovery_training_s={})
        active.update(job=budget.job,batch_started_wall_s=time.time(),stream_before_s=t.train_s);put(activefile,active,activefile.exists())
        episodes=[]
        if role=='D5-SL':
            for slot,tid in enumerate(ids):episodes.append(teacher_episode(budget,pool,t.actor,tid,f'{seed}:{role}:{op}:{slot}'))
        else:
            for slot,tid in enumerate(ids):
                if slot<len(active['episodes']):
                    rec=unarchive(active['episodes'][slot]);ep=recover_episode(pool,rec,role);restore_generator(gen,rec['sampler_after']);budget.counts['recovered_on_policy_episodes']+=1
                else:
                    out=policy_request(budget,pool,tid,f'{seed}:{role}:{op}:{slot}',t.actor,purpose='train',folder=f'streams/{seed}/{role}',generator=gen,seed=seed)
                    ep=out['episode'];active['episodes'].append(out['ref']);put(activefile,active,True)
                ep['reference_J_H']=refs.get(tid);episodes.append(ep)
        # Actual replayed optimizer attempts beyond a committed cursor are charged;
        # data collection is not repeated when the source-matched episode survived.
        prior_time=t.train_s;t.update(episodes);t.train_s=prior_time+max(time.perf_counter()-start,time.time()-start_w)
        cp=t.save_current(base,keep=t.cursor%128==0)
        candidate=save_C_candidate(t,base) if role=='D5-SL' and prior_time<=sl_time_target else None
        t.train_s=prior_time+max(time.perf_counter()-start,time.time()-start_w)
        if candidate is not None and t.train_s<=sl_time_target:
            prior_C=candidate;cstate=dict(entry=candidate,train_s=t.train_s);put(cstatefile,cstate,cstatefile.exists())
        # Persist timer including checkpoint cost; tensor/Adam state is unchanged.
        put(base/'TRAINING_TIME.json',dict(cursor=t.cursor,checkpoint_sha256=cp['sha256'],train_s=t.train_s,reference_allocation_s=summary['reference_allocation_s'],
            boundary='Preparation,own collection/teacher replay,backward,clip,Adam,LATEST checkpoint and C candidate export. Timing receipt own write is research logging, in pool wall.'), (base/'TRAINING_TIME.json').exists())
        summary.update(updates=t.cursor,train_s=t.train_s,training_compute_s=t.train_s+summary['reference_allocation_s'],SL_C=prior_C)
        budget.check=root_check
        if t.cursor in (1024,2048,4096):
            entry=save_export(t,base,f'step{t.cursor:05}');summary['endpoints'][str(t.cursor)]=entry;put(summaryfile,summary,summaryfile.exists())
            if role!='D5-SL' or t.cursor<=4096:evaluation.evaluate(budget,pool,'D5-SL-U' if role=='D5-SL' else role,seed,t.cursor,actor=t.actor)
        if t.cursor%32==0:
            phase='P2' if seed==2026111101 and (role=='Full-PG' or until_C) else 'P1' if seed==2026111101 else 'P3'
            put(summaryfile,summary,summaryfile.exists());state(phase=phase,
                status='TRAINING',seed=seed,role=role,update=t.cursor,train_s=t.train_s,training_compute_s=summary['training_compute_s'])
            print(dict(phase='TRAIN',seed=seed,role=role,update=t.cursor,train_s=round(t.train_s,3),compute_s=round(summary['training_compute_s'],3)),flush=True)
    summary.update(complete=t.cursor>=4096,updates=t.cursor,train_s=t.train_s,training_compute_s=t.train_s+summary['reference_allocation_s'],SL_C=prior_C)
    if role=='D5-SL':
        if prior_C is not None:
            # C selection follows elapsed training time only, not quality.
            ctime=cstate['train_s']
            summary['compute_ratio']=ctime/sl_time_target;summary['compute_matched']=abs(ctime/sl_time_target-1)<=.02
        else:summary.update(compute_ratio=None,compute_matched=False)
        summary['C_complete']=t.train_s>=sl_time_target or t.cursor>=16384
    put(summaryfile,summary,summaryfile.exists());return summary

def pipeline(budget):
    fast_budget(budget);pool=Pool(budget);base=evaluation.baselines(budget,pool);seed=2026111101
    if not read(HOME/'STATUS.json').get('P1_complete'):state(phase='P1',status='TRAINING_FIRST_SEED',scientific_identity=identity())
    pg=train_stream(budget,pool,'D5-PG',seed)
    if not pg['complete']:raise RuntimeError('P1 PG fixed4096 incomplete')
    sl=train_stream(budget,pool,'D5-SL',seed,sl_time_target=pg['training_compute_s'])
    pgr=evaluation.evaluate(budget,pool,'D5-PG',seed,4096,entry=pg['endpoints']['4096'])
    slr=evaluation.evaluate(budget,pool,'D5-SL-U',seed,4096,entry=sl['endpoints']['4096'])
    gate=evaluation.p1_gate(pgr,slr,base['FrozenSL']);put(HOME/'P1_DECISION.json',gate,(HOME/'P1_DECISION.json').exists())
    put(HOME/'P1_ROUTE_CHANGES.json',dict(rows=evaluation.route_changes(pgr,base['FrozenSL'])),(HOME/'P1_ROUTE_CHANGES.json').exists())
    state(P1_complete=True,P1_gate=gate['status'],P1_PG_updates=4096,P1_SL_U_updates=4096)
    if not gate['passed']:closeout(gate['status'],'Fixed4096 P1 task-level screening gate not passed',P1_complete=True);return
    from . import later_stages
    later_stages.advance(budget,pool,base,pg,sl)
