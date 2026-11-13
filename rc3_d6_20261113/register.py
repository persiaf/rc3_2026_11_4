"""Freeze D6 mathematics, all inputs and sources before new arithmetic/optimizers."""
from pathlib import Path
import json,hashlib,datetime,subprocess,ast,time
from .common import ROOT,HOME,D5,BASE,RUN,H_ID,read,put,sha

def main():
    if (HOME/'PROTOCOL.json').exists():raise FileExistsError('No reset of D6 registration')
    origin=datetime.datetime(2026,11,13,1,20,tzinfo=datetime.timezone.utc).timestamp()
    protocol=dict(schema='rc3-D6-causal-tail-v1',run_id=RUN,base_commit=BASE,origin_unix_s=origin,deadline_unix_s=origin+7200,
        origin_scope='Conservative timestamp before first task read; includes preparation,diagnostics,training,evaluation,audit,review,Git and waiting; no reset on resume',
        request_limit=4920,actor_optimizer_limit=768,teacher_replay_limit=2048,critic_updates=0,new_scene_navigation_teacher=0,
        planned_requests=dict(train=4096,eval=792,total=4888,technical_same_input_reserve=32),engineering_optimizer=0,
        roles=dict(A='Exact D5 full-trajectory scalar-cost PG',B='D6 causal remaining-cost PG',SL='Matched five-tensor continuing SL'),
        network='Original D5 five tensors25152/65088; all other original tensors frozen; no architecture/input change',
        allowlist=['key.weight','dynamic.weight','nav.weight','query.weight','query.bias'],
        initialization='Exact legal SL256 independently cloned; never D5 endpoints/engineering/diagnostic weights',
        optimizer=dict(algorithm='Adam',lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0,clip_grad_norm=1.,batch=8,epochs=1,entropy=0),
        A_loss='mean_episode((J_H-J_FrozenSL_same_problem_H).detach()*sum_t(logpi_t))',
        B_loss='mean_episode(sum_t((J_H-raw_prefix_cost_before_t-q_t*J_FrozenSL_same_problem_H).detach()*logpi_t))',
        q='remaining_before_t / initial_task_count; only division by initial_task_count, no other return/task normalization',
        SL_loss='mean_episode(mean_action(masked CE))',gamma=1,H_identity=H_ID,mask_cost_and_endpoints='Unchanged',
        sampling=dict(seed=2026111101,convention='Exact D5 SHA256-derived torch.Generator start for each independent A/B instance; separate on-policy draws and independently saved RNG; no trajectory sharing'),
        schedule='First256 batches of D5 seed2026111101: each4 full10x10 +4 local; identical task order for A/B/SL',
        endpoints=[0,128,256],primary_endpoint=256,step0='Actual three zero-update exports; quality reused from current exact FrozenSL128 grid; no extra step0 requests or separate timing',
        evaluation='66 exposed SEL8+DEV3, six methods freshly run at128 and256; native-final and neural raw/H separate; no fair deployment speed claim from unequal caches',
        quality_gate=dict(HJ_parent_equal_gain_vs_FrozenSL_min=.005,HJ_parent_equal_gain_vs_matched_SL_min=.005,
            SEL_and_DEV_each_nonnegative_vs_both=True,all66_proxy_valid=True,B_vs_A='Report paired difference; no substitution for the two required controls',
            endpoint128='Fixed descriptive screen only; no early weight selection or tuning',endpoint256='Sole primary decision; no extension or seed changes'),
        cost_validation=dict(trajectories=32768,raw_terminal_exit=True,H_correction_once=True,float64_abs_tolerance=1e-7,
            no_new_H=True,old_trajectory_use='Statistics/decomposition only, never new Actor optimizer data'),
        stop='Cost/source/mask/finite failure stops before subsequent optimization; preserve consumed resources. Missing fixed256 is technical INCOMPLETE, not scientific failure. Main gate failure ends recipe.',
        recovery='All streams checkpoint model/Adam/globalRNG/sampling/cursor/own active episodes/resources; actual process resume at128. Ambiguous uncheckpointed optimizer attempt stops, no free repeat.',
        forbidden=['original60 semantics','oldS3 semantics','confirmation/final retained sets','fixed32','A1/Critic reopen','new teacher','new scene','new navigation','main merge','history rewrite'])
    put(HOME/'PROTOCOL.json',protocol)
    schedule=read(D5/'schedules/2026111101.json');batches=schedule['batches'][:256]
    if len(batches)!=256 or any(len(b)!=8 or sum(t.startswith('F') for t in b)!=4 for b in batches):raise ValueError('D5 first256 schedule')
    put(HOME/'COMMON_SCHEDULE.json',dict(source='rc3_d5_revision_20261112/schedules/2026111101.json',source_sha256=sha(D5/'schedules/2026111101.json'),batches=batches,
        total_slots=2048,unique_tasks=len({t for batch in batches for t in batch}),same_for=['A','B','SL']))
    paths=['FULL_TASK_REGISTRATION.json','ORIGINAL_LOCAL_TEACHERS.json','FULL_TEACHER_MANIFEST.json','JOINT_DATA_INDEX.json','REFERENCE_MANIFEST.json','JOINT_DATA_ADMISSION.json','SCIENTIFIC_SOURCE_BINDING_R04B.json','RESOURCE_ACCOUNTING.json',
        'streams/2026111101/D5-PG/REQUESTS.jsonl.gz','streams/2026111101/D5-PG/REQUEST_POINTERS.jsonl','streams/2026111101/D5-PG/UPDATES.jsonl.gz']
    inputs={str((D5/n).relative_to(ROOT)).replace('\\','/'):sha(D5/n) for n in paths}
    archive='rc3_d5_revision_20261112/streams/2026111101/D5-PG/REQUESTS.jsonl.gz'
    pointer=subprocess.check_output(['git','show',BASE+':'+archive],cwd=ROOT).decode()
    expected=next(x.split(':',1)[1] for x in pointer.splitlines() if x.startswith('oid sha256:'))
    if inputs[archive]!=expected:raise ValueError('Actual saved32768 archive differs from baseline LFS')
    put(HOME/'REGISTERED_INPUTS.json',dict(files_sha256=inputs,old_D5_archive_LFS_oid=expected,
        old_D5_status='COMPLETE4096/4096,NO_DEVELOPMENT_SIGNAL,P2-P5NOT_RUN; no old source or state modifications',
        new_data_generated=False,training_reference='352 exact frozen D5 problem/H/SL keyed records; no new reference request'))
    b=read(D5/'SCIENTIFIC_SOURCE_BINDING_R04B.json')
    for p in HOME.glob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'));b['source_sha256'][p.relative_to(ROOT).as_posix()]=sha(p)
    art={str((HOME/n).relative_to(ROOT)).replace('\\','/'):sha(HOME/n) for n in ('PROTOCOL.json','COMMON_SCHEDULE.json','REGISTERED_INPUTS.json')}
    art.update(inputs);b['artifact_sha256'].update(art)
    stable=dict(run_id=RUN,base_commit=BASE,protocol_sha256=sha(HOME/'PROTOCOL.json'),sources={k:v for k,v in b['source_sha256'].items() if k.startswith(HOME.name+'/')},inputs=art)
    ident=hashlib.sha256(json.dumps(stable,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    put(HOME/'SCIENTIFIC_IDENTITY.json',dict(identity=ident,**stable))
    b.update(schema='rc3-D6-bound-execution-v1',base_commit=BASE,run_id=RUN,revision_run_id=RUN,scientific_identity=ident,D6_artifacts=art,
        source_role='D6 execution; old D5 data/source identities inherited verbatim, no old execution authorization reused')
    put(HOME/'SOURCE_BINDING.json',b)
    put(HOME/'AUTHORIZATION.json',dict(authorized=True,authority='Current explicit RC3-D6 user task',run_id=RUN,
        source_binding_sha256=sha(HOME/'SOURCE_BINDING.json'),protocol_sha256=sha(HOME/'PROTOCOL.json'),scientific_identity=ident,
        stages=['old_saved_credit_diagnostic','A/B/SL256','current66x6x2evaluation','saved_output_audit','independent_review','codex_branch_publication'],no_other_stages=True))
    put(HOME/'RESOURCE_ACCOUNTING.json',dict(schema='D6-new-budget-with-old-ledger-preserved',limits=dict(requests=4920,actor=768,teacher_replay=2048,wall_s=7200),
        counts=dict(requests=0,requests_completed=0,requests_failed=0,actor_optimizer_attempts=0,actor_optimizer_completed=0,teacher_replays=0,
            critic_optimizer_attempts=0,new_scenes=0,navigation_solves=0,new_teacher_requests=0,original60_reads=0,oldS3_reads=0,confirmation_requests=0,final_requests=0),
        wall=dict(total_s=time.time()-origin),jobs={},prior_D5=dict(path='rc3_d5_revision_20261112/RESOURCE_ACCOUNTING.json',sha256=inputs['rc3_d5_revision_20261112/RESOURCE_ACCOUNTING.json'],
            requests=33954,Actor=8202,not_reset_or_overwritten=True,not_part_of_new_D6_caps=True)))
    put(HOME/'STATUS.json',dict(run_id=RUN,scientific_identity=ident,status='REGISTERED_NOT_RUN',closed=False,phase='diagnose',old_D5_unchanged=True))
    print(dict(status='REGISTERED_BEFORE_NEW_COMPUTATION',identity=ident,elapsed_in_new7200s=time.time()-origin))

if __name__=='__main__':main()
