import time
START_M,START_W=time.monotonic(),time.time()
import argparse
import ast
import json
import subprocess
from .common import HOME,ROOT,BASE,RATES,TABLE,read,put,sha

def register():
    if (HOME/'SOURCE_BINDING.json').exists():raise FileExistsError('One task registration; no resource reset')
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==BASE
    prior=read(ROOT/'rc3_frozen_reference_pg_20261111/execution_v2/SOURCE_BINDING.json')
    sources=dict(prior['source_sha256']);artifacts=dict(prior['artifact_sha256'])
    for path in sorted(HOME.glob('*.py')):
        ast.parse(path.read_text(encoding='utf-8'));sources[path.relative_to(ROOT).as_posix()]=sha(path)
    # All reused production modules bind to the exact user-specified baseline.
    for path in sorted((ROOT/'rc3_frozen_reference_pg_20261111').glob('*.py')):
        name=path.relative_to(ROOT).as_posix();payload=subprocess.check_output(['git','show',BASE+':'+name],cwd=ROOT)
        import hashlib
        expected=hashlib.sha256(payload).hexdigest();assert sha(path)==expected
        sources[name]=expected
    previous=read(ROOT/'rc3_frozen_reference_pg_20261111/SCHEDULE.json')
    schedule=dict(seed=11622026,training=previous['training'][:64],
        evaluation=[c for c in previous['evaluation'] if c['group']=='independent_parent_exposed_development'],
        teacher_pool_order=list(range(256)),teacher_pool_source='rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/teachers256/POOL_INDEX.json',
        teacher_pool_sha256='98c1e7335fced56589d6d978d1eb3c9a895590c065413c8746d77c03a2044a1a')
    assert len(schedule['evaluation'])==18 and len(schedule['training'])==64
    assert sha(ROOT/schedule['teacher_pool_source'])==schedule['teacher_pool_sha256']
    put(HOME/'SCHEDULE.json',schedule)
    protocol=dict(schema='actor-positive-and-conditional-step-scale-v1',base_commit=BASE,seed=11622026,
        fixed_research_goal='large complex parent DRL; frozen Actor quickly completes local full constrained tasks',
        positive_selection=dict(pool='fixed original256 saved training teacher labels/actions/views; no new teacher or alternate task',
            state_restoration='accurate frozenSL256 plus exact saved prefixes, stop before final teacher env.step, no terminal/H/planning/cost labels',
            eligible='teacher legal and different from SL argmax; construct actual206-64-64-1 float32 A1 network weights with strict teacher margin>=1e-4, abs(delta)<=.25, FULL legal-mask original guard',
            witness='two-hinge one-bump actual A1 network at a stable candidate-feature coordinate;48 scalar bisections, zero optimizer; no gap-only substitution',
            witness_scope='per-state actual-network existence; not a shared witness-weight fit proof; actual CE trials fit all selected states jointly',
            deterministic_order='SL teacher rank2 FIRST, then SHA256 canonical(teacher index,view identity,prefix,state index); first32 verified witnesses',
            max_states=32,minimum_interpretable_states=16,insufficient='report exact count, do not generate/replace tasks; online NOT_RUN',
            selection_seal='write selected data/feature identities and FEASIBLE_RANK_WITNESSES before FIRST actual CE optimizer call'),
        positive_rates=list(RATES),positive_updates=dict(per_rate=128,total_max=384,loss='CE state mean over SAME fixed selected states in SAME order every step',
            Adam=dict(betas=[.9,.999],eps=1e-8,weight_decay=0.,gradient_clip=1.),
            start='same accurateSL256-derived fresh A1 zero output and identical hidden tensors for all rates',
            success='at least16 witnessed states; exact step128 strict target-top1 margin>1e-5 fraction>=80%, accepted>=ceil(.9*128), finite+mask+amplitude+all applied guards satisfied',
            no_midpoint_or_best_weight_selection=True,diagnostic_weights_forbidden_as_online_start=True),
        actor=dict(input=206,hidden=[64,64],distances='ON',correction='relative_pre_tanh_centered_v1',amplitude=.25,zero_output=True,
            base_path='rc3_integrated_rppo/runs/rc3_integrated_rppo_v1_20261110_01/actors/11622026/SL256_0256_1817f58aa650.pt',
            base_sha256='60f5aa4e59983e92d565f0599823b710965cf2c96a041e7eb483d23a8eb0556c'),
        guard=dict(max_step_kl=.001,max_step_probability_change=.02,max_base_total_variation=.1,consecutive_rejection_limit=16),
        caps=dict(positive=dict(requests=0,actor_optimizer_attempts=384,critic_optimizer_attempts=0,wall_s=600),
            online=dict(requests=984,actor_optimizer_attempts=192,critic_optimizer_attempts=0,wall_s=3600)),
        trigger='ONLY original3e-5 FAIL and a new rate PASS: take MINIMUM passed new rate; originalPASS or allFAIL/insufficient => online NOT_RUN; NEVER select from complete-route evaluation',
        online=dict(arms={'A':'B-cond lr3e-5','B':'B-cond calibrated minimum passing LR','C':'continuedSL-A1 calibrated LR'},
            fresh_initialization='new identical SL256 plus zeroA1; NO diagnostic weights loaded',opportunities_per_arm=64,
            RL_per_opportunity={'10x10':3,'4x3':1,'6x6':1,'8x8':1},online_rollouts=768,teacher_replays=384,
            endpoints=[32,64],evaluation_cases=18,methods=['A','B','C','FrozenSL','Fixed3','Detour16'],evaluation_requests=216,
            update='same prior Trainer: complete gamma=lambda1 detach(G-Bcond), per-batch RMS, one candidate, .1 ratio clip, .02 baseKL, no entropy change; CE control same route-mean reduction',
            random_stream='int(SHA256 canonical([actor-positive-scale,11622026,op,slot])[:15],16) then frozen runtime action seed transform; CRN, each arm OWN current-policy fresh calls, no sharing trajectories',
            table='exact frozen M2 B-cond, already normalized, no repeated division by n, no update, absent from deployed Actor',
            quality='B saves>=.5% vs BOTH FrozenSL andC in complete raw_J and H_J, parent-cluster descriptive95% lower>0; no per-size point regression vs either. BvsA reported, not replacement for strong controls.',
            inference='3 physical DEV parents, case costs averaged within parent; paired20000 PCG64seed11652026 linear quantile; exposed single-seed mechanism screen only',
            timing='current complete request, saved input/source/nav restore, cold model, features/policy/native/H/validation/archive/index; actual first-choice, ratio/KL/TV and all failures',
            native='original Fixed3/Detour16 full pipeline; final native route is native raw convention, not a new H ablation; full internal-search counters retained'),
        table_path=TABLE,table_sha256=sha(ROOT/TABLE),schedule_sha256=sha(HOME/'SCHEDULE.json'),
        H_identity='5db67cacc1128b6143c7a04b9b84a70c37bdc0ab917a0eebae0b8aca132c4d14',
        stop='source/input/nonfinite failure stops stage and preserves consumption; no task/seed replacement.16 consecutive rejections stop configuration. No automatic step/feature/guard/amplitude extension.',
        historical_boundary='PG-global/PG-cond128 STOP and all negative results frozen; no256, no Critic fitting/new reference; original60/oldS3/final test/Word untouched',
        wall_scope='all active registration/restore/witness/fit/new-output audit charged; code authorship/prose/Git maintenance separate. Failed entries retained. No old million-row CSV audit.')
    put(HOME/'PROTOCOL.json',protocol)
    for name in ('PROTOCOL.json','SCHEDULE.json'):artifacts[(HOME/name).relative_to(ROOT).as_posix()]=sha(HOME/name)
    binding=dict(schema='actor-positive-scale-source-v1',base_commit=BASE,source_sha256=sources,artifact_sha256=artifacts,
        environment=prior['environment'],protocol_sha256=sha(HOME/'PROTOCOL.json'),schedule_sha256=sha(HOME/'SCHEDULE.json'),table_sha256=sha(ROOT/TABLE))
    put(HOME/'SOURCE_BINDING.json',binding)
    put(HOME/'PROTOCOL_APPROVAL.json',dict(approved=True,authority='explicit current user task',protocol_sha256=sha(HOME/'PROTOCOL.json'),source_binding_sha256=sha(HOME/'SOURCE_BINDING.json')))
    put(HOME/'IMPLEMENTATION_REVIEW.json',dict(status='GO',scope='known-training-state positive control then strictly conditional online LR comparison; no old stage reopened',source_binding_sha256=sha(HOME/'SOURCE_BINDING.json')))
    for stage in ('positive','online'):
        put(HOME/(stage.upper()+'_AUTHORIZATION.json'),dict(authorized=True,authority='explicit user task',stage=stage,
            prerequisite='TRIGGER_DECISION.triggered true' if stage=='online' else 'registered actual A1 witnesses before CE',source_binding_sha256=sha(HOME/'SOURCE_BINDING.json'),caps=protocol['caps'][stage]))
    put(HOME/'REGISTRATION_RECEIPT.json',dict(status='BOUND_BEFORE_FIRST_NEW_NUMERICAL_COMPUTATION',wall_s=max(time.monotonic()-START_M,time.time()-START_W),
        optimizer_attempts=0,requests=0,protocol_sha256=sha(HOME/'PROTOCOL.json'),source_binding_sha256=sha(HOME/'SOURCE_BINDING.json')))
    print('Protocol, exact source, deterministic state-selection rule/pool and both stage budgets registered')

def launch(stage):
    if (HOME/(stage+'_control')).exists():raise FileExistsError('No repeated stage or resource reset')
    if stage=='online':
        d=read(HOME/'TRIGGER_DECISION.json')
        if not d['triggered']:raise RuntimeError('Online NOT_RUN: positive trigger false')
    binding=read(HOME/'SOURCE_BINDING.json')
    for name in ('__init__.py','supervisor.py'):
        p=ROOT/'rc3_r04_m1_recovery_20261111'/name;assert sha(p)==binding['source_sha256'][p.relative_to(ROOT).as_posix()]
    from rc3_r04_m1_recovery_20261111.supervisor import launch_supervised
    prior=read(HOME/'REGISTRATION_RECEIPT.json')['wall_s'] if stage=='positive' else 0.
    cap=read(HOME/'PROTOCOL.json')['caps'][stage]['wall_s']
    receipt=launch_supervised([str(ROOT/'rc3_learning/.venv/Scripts/python.exe'),'-B','-X','utf8','-m',HOME.name+'.worker',stage],
        HOME/(stage+'_control'),'actor-positive-scale-20261111-'+stage+'-01',wall_limit_s=cap-prior-2,
        closeout_reserve_s=15,termination_tolerance_s=2,cwd=ROOT,entry_monotonic=START_M,entry_wall=START_W,worker_heartbeat_path=HOME/stage/'WORKER_HEARTBEAT.json')
    elapsed=max(time.monotonic()-START_M,time.time()-START_W)
    put(HOME/(stage.upper()+'_ENTRY_CONSUMPTION.json'),dict(status=receipt['status'],wall_s=elapsed,prior_registration_s=prior,stage_cumulative_wall_s=elapsed+prior,
        root_exit_confirmed=receipt['root_exit_confirmed'],active_processes_at_end=receipt['job_active_processes_at_end']))
    print(dict(stage=stage,status=receipt['status'],wall_s=elapsed+prior))
    raise SystemExit(0 if receipt['status']=='COMPLETE' else 1)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('register','positive','online'));a=p.parse_args()
    register() if a.action=='register' else launch(a.action)
