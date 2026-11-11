"""One preregistration for engineering and science; accepted Job supervisor."""
import time
START_M,START_W=time.monotonic(),time.time()
import argparse
import ast
import hashlib
import json
import subprocess
import sys
from .common import HOME,ROOT,BASE,TABLE,ROLES,METHODS,read,put,sha

def register():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==BASE
    if (HOME/'SOURCE_BINDING.json').exists():raise FileExistsError('No repeated registration/reset')
    old=read(ROOT/'rc3_r04_m2_20261111/attempts/02/SOURCE_BINDING.json')
    sources=dict(old['source_sha256'])
    for p in HOME.glob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'));sources[p.relative_to(ROOT).as_posix()]=sha(p)
    artifacts={}
    def admit(path,expected):
        if path in artifacts and artifacts[path]!=expected:raise ValueError('Conflicting frozen artifact')
        artifacts[path]=expected
    refs=read(ROOT/'rc3_integrated_rppo/DATA_REFERENCES.json')
    index=read(ROOT/'rc3_integrated_rppo/runs/rc3_integrated_rppo_v1_20261110_01/inputs/INDEX.json')
    for path,v in refs['source_files'].items():admit(path,v['sha256'])
    for e in list(refs['bank'].values())+list(index.values()):
        for v in e['files'].values():admit(v['path'],v['sha256'])
    for v in refs['teachers']+[refs['teacher_pool_index']]:admit(v['path'],v['sha256'])
    for path,v in old['artifact_sha256'].items():
        if path.endswith('.pt') and ('/S1/update_0256.pt' in path or '/SL256_0256_' in path):admit(path,v)
    for name in ['rc3_integrated_rppo/DATA_SPLITS.json','rc3_integrated_rppo/DATA_REFERENCES.json',
        'rc3_integrated_rppo/runs/rc3_integrated_rppo_v1_20261110_01/inputs/INDEX.json',
        'rc3_navigation_compat_20261106/SOURCE_BINDING_UPDATED.json','rc3_navigation_compat_20261106/LEGACY_TEACHER_REPLAY_MANIFEST.json',
        'rc3_global_local_execution_v3/REGISTRY.json','rc3_global_local_execution_v3/EXECUTION_SOURCE_BINDING.json',
        'rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/seed_11622026/S1/update_0256.pt.json',
        'rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/seed_11622026/S1/update_0256.pt.commit.json',
        'rc3_r04_m1_recovery_20261111/engineering/ENGINEERING_TEST_RESULTS.json',TABLE]:
        payload=subprocess.check_output(['git','show',BASE+':'+name],cwd=ROOT)
        expected=hashlib.sha256(payload).hexdigest();assert sha(ROOT/name)==expected;admit(name,expected)
    legacy=read(ROOT/'rc3_navigation_compat_20261106/LEGACY_TEACHER_REPLAY_MANIFEST.json')
    # Exact replay helper may read the manifest's own original source/science anchors.
    for k,v in old['artifact_sha256'].items():
        if k not in artifacts and not k.endswith('.pt') and '/requests/' not in k:artifacts[k]=v
    schedule_name='rc3_conservative_refinement/integration_20261110/SCHEDULE_R03.json'
    raw=(ROOT/schedule_name).read_bytes();assert hashlib.sha256(raw).hexdigest()=='2a6fd2d65a17fd510dd9837a82a293cb616803e0006165e432882f1ea3cd90a0'
    previous=json.loads(raw)
    schedule=dict(seed=11622026,training=previous['training'][:128],evaluation=previous['evaluation'],physical_groups=previous['physical_groups'],
        parent_schedule_source=dict(path=schedule_name,sha256=sha(ROOT/schedule_name)))
    assert len(schedule['evaluation'])==66 and len(schedule['training'])==128
    put(HOME/'SCHEDULE.json',schedule)
    protocol=dict(schema='rc3-frozen-reference-online-PG-protocol-v1',base_commit=BASE,
        fixed_research_goal='large complex parent DRL training; frozen Actor rapidly completes variable-size local constrained task routes',
        experiment='frozen-reference-online-PG-11622026-v1',seed=11622026,
        frozen_reference=dict(path=TABLE,sha256=sha(ROOT/TABLE),fit_parent_ids=read(ROOT/TABLE)['parent_ids'],
            condition=read(ROOT/TABLE)['condition'],unknown='fit-global',value='-b_L-0.75*b_K+b_H; ALREADY normalized, NEVER divide n again',
            update='NEVER',role='action-independent training return reference only; absent from Actor-only deployment'),
        actor=dict(base_path='rc3_integrated_rppo/runs/rc3_integrated_rppo_v1_20261110_01/actors/11622026/SL256_0256_1817f58aa650.pt',
            base_sha256='60f5aa4e59983e92d565f0599823b710965cf2c96a041e7eb483d23a8eb0556c',
            schema='relative_pre_tanh_centered_v1',input_dim=206,hidden=[64,64],distances_enabled=True,amplitude=.25,zero_final_layer=True,
            initialization='same runtime._seed(11622026,A1_residual_init) fresh hidden/net tensors; no M1/M2/R03/oldM3 residual weights'),
        update=dict(gamma=1.,gae_lambda=1.,A_sample='detach(G_complete-b_frozen(state))',
            online_own_current_policy=True,off_policy_sharing=False,actor_candidates_per_opportunity=1,PPO_epochs=1,
            actor_lr=3e-5,betas=[.9,.999],eps=1e-8,gradient_clip=1.,ratio_clip=.1,
            beta_base_KL=.02,entropy_term='unchanged:none',advantage_scaling='same prior per-batch all-state RMS+1e-8; raw unscaled A recorded; no advantage mean subtraction',
            reduction='RL state_SUM then episode_MEAN; matched SL state_MEAN CE then episode_MEAN',critic='NONE'),
        guard=dict(max_step_kl=.001,max_step_probability_change=.02,max_base_total_variation=.1,consecutive_rejections_stop=16),
        schedule_sha256=sha(HOME/'SCHEDULE.json'),roles=list(ROLES),evaluation_methods=list(METHODS),endpoints=[64,128],
        sampling_stream='common random numbers: int(SHA256(11622026|frozen-reference-PG|opportunity|slot)[:15],16), then unchanged runtime._seed(seed,action_sampling); independent current-policy sampling per arm, no trajectory sharing',
        caps=dict(engineering=dict(requests=24,actor_optimizer_attempts=16,critic_optimizer_attempts=0,wall_s=300),
            science=dict(requests=2328,actor_optimizer_attempts=384,critic_optimizer_attempts=0,wall_s=7200)),
        scientific_counts=dict(online_rollouts=1536,endpoint_evaluation_requests=792,teacher_replays=768,actor_opportunities=384,
            per_RL_opportunity={'10x10':3,'4x3':1,'6x6':1,'8x8':1},per_SL_opportunity=6),
        engineering=dict(actual_requests=19,planned_Actor_attempts=6,models_discarded=True,
            tests=['zero correction exact logits and complete raw/H SL routes','frozen component table units/fallback/detach',
                'both current-policy RL real updates and matched SL','registered forced rejection AFTER actual candidate/real guard, exact Actor+Adam rollback',
                'actual checkpoint model/Adam/RNG/cursor/usedIDs/query resource exact restore and next-step equivalence',
                'strict Actor-only production schema/role/stage, complete route export equivalence'],
            tolerance=dict(logits_atol=1e-6,route_cost_atol=1e-7,rollback_and_restore='bitwise'),no_neural_critic=True),
        quality_gate='PG-cond saving>=0.5% AND physical-parent paired bootstrap95% lower>0 versus FrozenSL AND continued_SL_A1, THEN PG-global, separately raw_J and H_J and both exposed cohorts; each local size raw/H point saving>=0 versus FrozenSL/continuedSL; all routes valid',
        inference=dict(unit='physical parent; average case costs within parent first',bootstrap=20000,generator='PCG64',seed=11652026,quantile='linear',
            cohorts='8 exposed trained-parent new-task selection;3 exposed development parents; descriptive, not confirmatory'),
        stop=['source/unit/nonfinite/invalid-current-policy/input/constraint failure: stop preserve; no resampling',
            'resource cap or16 consecutive guard rejections: stop paired run, no refill',
            'always fixed64/128 if healthy; neither endpoint has valid route gain: end candidate; never256',
            'even if screen passes, no automatic further training or final test'],
        H_identity='5db67cacc1128b6143c7a04b9b84a70c37bdc0ab917a0eebae0b8aca132c4d14',
        native='original endpoint Fixed3/Detour16 search/H; no extra common H; native final route is raw comparison convention, every H pre/postprocess route separately retained',
        timing='entire request including saved input/navigation restoration, source check, cold deployment load, provider/features/actions/native/H/export/validation/evidence and separate IO; all caches counted; feature stage separately',
        boundaries='No new scenes/navigation solve/teachers; original60/oldS3/final tests0; M1G3FAIL and M2deleteCB unchanged; historical active design pointers preserved',
        old_scientific_identity_excludes_current_authorization=True)
    put(HOME/'PROTOCOL.json',protocol)
    for n in ('PROTOCOL.json','SCHEDULE.json'):admit((HOME/n).relative_to(ROOT).as_posix(),sha(HOME/n))
    binding=dict(schema='rc3-frozen-reference-PG-source-binding-v1',source_sha256=sources,artifact_sha256=artifacts,
        environment=old['environment'],protocol_sha256=sha(HOME/'PROTOCOL.json'),schedule_sha256=sha(HOME/'SCHEDULE.json'),
        base_commit=BASE,H_identity=protocol['H_identity'])
    put(HOME/'SOURCE_BINDING.json',binding)
    put(HOME/'PROTOCOL_APPROVAL.json',dict(approved=True,authority='explicit current user task',protocol_sha256=sha(HOME/'PROTOCOL.json'),source_binding_sha256=sha(HOME/'SOURCE_BINDING.json')))
    for stage in ('engineering','science'):
        put(HOME/(stage.upper()+'_AUTHORIZATION.json'),dict(authorized=True,authority='explicit current user task includes integration/real transaction and bounded scientific execution',
            stage=stage,source_binding_sha256=sha(HOME/'SOURCE_BINDING.json'),protocol_sha256=sha(HOME/'PROTOCOL.json'),
            prerequisite='engineering CHECKS.json status PASS' if stage=='science' else 'implementation review GO',caps=protocol['caps'][stage]))
    put(HOME/'IMPLEMENTATION_REVIEW.json',dict(status='GO',source_binding_sha256=sha(HOME/'SOURCE_BINDING.json'),scope='new frozen-reference Actor-only training; old G3 remainsFAIL',explicit_stage_authorization_separate=True))
    put(HOME/'REGISTRATION_RECEIPT.json',dict(wall_s=max(time.monotonic()-START_M,time.time()-START_W),status='BEFORE_FIRST_ACTUAL_CALL',
        protocol_sha256=sha(HOME/'PROTOCOL.json'),source_binding_sha256=sha(HOME/'SOURCE_BINDING.json'),actor_optimizer_attempts=0,complete_requests=0))
    print('One-time protocol/source/schedule/resource registration COMPLETE')

def launch(stage):
    binding=read(HOME/'SOURCE_BINDING.json')
    if stage=='science' and read(HOME/'engineering/CHECKS.json')['status']!='PASS':raise RuntimeError('Real engineering acceptance required')
    for name in ('__init__.py','supervisor.py'):
        path=ROOT/'rc3_r04_m1_recovery_20261111'/name
        if sha(path)!=binding['source_sha256'][path.relative_to(ROOT).as_posix()]:raise ValueError('Accepted supervisor source')
    from rc3_r04_m1_recovery_20261111.supervisor import launch_supervised
    if (HOME/(stage+'_control')).exists():raise FileExistsError('No duplicate stage execution')
    cap=read(HOME/'PROTOCOL.json')['caps'][stage]['wall_s'];prior=read(HOME/'REGISTRATION_RECEIPT.json')['wall_s'] if stage=='engineering' else 0.
    receipt=launch_supervised([str(ROOT/'rc3_learning/.venv/Scripts/python.exe'),'-B','-X','utf8','-m',HOME.name+'.worker',stage],
        HOME/(stage+'_control'),'frozen-reference-PG-20261111-'+stage+'-01',wall_limit_s=cap-prior-2,
        closeout_reserve_s=15,termination_tolerance_s=2,cwd=ROOT,entry_monotonic=START_M,entry_wall=START_W,
        worker_heartbeat_path=HOME/stage/'WORKER_HEARTBEAT.json')
    elapsed=max(time.monotonic()-START_M,time.time()-START_W)
    put(HOME/(stage.upper()+'_ENTRY_CONSUMPTION.json'),dict(status=receipt['status'],wall_s=elapsed,prior_registration_s=prior,
        stage_cumulative_wall_s=prior+elapsed,root_exit_confirmed=receipt['root_exit_confirmed'],active_processes_at_end=receipt['job_active_processes_at_end']))
    print(json.dumps(dict(stage=stage,status=receipt['status'],wall_s=prior+elapsed)))
    raise SystemExit(0 if receipt['status']=='COMPLETE' else 1)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('register','engineering','science'));a=p.parse_args()
    register() if a.action=='register' else launch(a.action)
