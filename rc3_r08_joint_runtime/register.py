"""Freeze B implementation, task identities and original approved gates."""
import time,hashlib,json,datetime
from .common import ROOT,HOME,RUN,BASE,H_ID,put,read,sha

def main():
    if (HOME/'PROTOCOL.json').exists():raise RuntimeError('Existing B identity; no reset')
    origin=datetime.datetime(2026,11,13,7,16,30,tzinfo=datetime.timezone.utc).timestamp()
    docs=['R08_MASTER_PLAN.md','HIERARCHICAL_DRL_DESIGN.md','JOINT_OPTIMIZATION_MODEL.md','CODE_REFACTOR_PLAN.md']
    frozen={f'rc3_r08_joint_decision_design_20261113/{n}':sha(ROOT/'rc3_r08_joint_decision_design_20261113'/n) for n in docs}
    tasks=read(ROOT/'rc3_r07_20261113/TASKS.json');put(HOME/'TASKS.json',tasks)
    protocol=dict(schema='R08-B-registered-joint-opportunity-v1',run_id=RUN,base_commit=BASE,stage='B_ONLY',
        origin_unix_s=origin,deadline_unix_s=origin+1800,wall_scope='All implementation,registration,tests,restore,real requests,H,validation,analysis,review,Git and waiting',
        normal_requests=816,max_requests=900,Actor_optimizer=0,Critic_optimizer=0,new_teacher_scene_navigation=0,
        tasks='Exact R07 48 and order unchanged,32 full16 local6/5/5; all fixed parents unique',frozen_design_sha256=frozen,H_identity=H_ID,
        anchor='Fresh exact frozenSL greedy complete rollout plus original H; require exact saved raw/H reproduction',
        menu=dict(drop_blocks=8,replacement_pool=8,per_block_actions=2,total_actions=16,adjacent_first=True,
            U_empty='Deletion-only, as registered conditional enumeration is for nonempty U; do not fill missing slots',
            cancel='drop/add intersection cancelled before dedup/ranking/arity',
            order='Alternating sorted adjacent and coverage-overlap nonadjacent classes; net-add count, add-ID tuple, drop-ID tuple tie',
            dedup='Per-block first2 then global exact net-edit dedup; do not backfill duplicates',
            source='Frozen raw incumbent; no prospective route cost or H used in menu ordering'),
        lower=dict(insertions_max=2,add_order='Global ID ascending',position='Finite minimum navigation increment then earliest position',two_opt_passes=8,cleanup='One registered normalize after2opt',extra_joint_search=False),
        metrics=dict(material_relative=.01,tie_absolute=1e-7,prefixes=[1,2,4,8,16],gain='(anchor_H_J-edit_H_J)/max(anchor_H_J,1e-7)',
            missing='Missing menu slots/tasks contribute zero opportunity, fixed48 denominator',
            attribution='Net descriptor arity after cancellation for original gate; also report actual raw/H configuration deltas, no causal interaction claim'),
        gate=dict(full_parents_min=8,local_parents_min=4,task_material_fraction_min=.30,full_material_fraction_min=.25,local_material_fraction_min=.25,
            oracle_parent_mean_min=.015,material_edit_net_joint_fraction_min=.5,all_valid=True,
            throughput=dict(D_requests=22428,gradient_checkpoint_reserve_s=384*5,setup_analysis_delivery_reserve_s=3000,D_wall_cap_s=21600,
                estimator='max(mean full request time,p95 edit time)*22428 +1920+3000; include menu allocation and measured source/model setup, conservative engineering planning estimate not actual training throughput')),
        future_C_D_E='NOT_AUTHORIZED_NOT_RUN',technical_errors='Stop on identity,input,real constraint,source/limit or implementation error; never replace case/action',delivery_reserve_s=300)
    put(HOME/'PROTOCOL.json',protocol)
    prior=read(ROOT/'rc3_r07_20261113/SOURCE_BINDING.json')
    bind=dict(schema='R08-B-execution-source-v1',run_id=RUN,base_commit=BASE,R08_B_stage='B_ONLY',H_identity=H_ID,
        source_sha256=prior['source_sha256'],artifact_sha256=prior['artifact_sha256'],environment=prior['environment'],
        prior_producer_envelope_sha256=sha(ROOT/'rc3_r07_20261113/SOURCE_BINDING.json'),
        source_role='R08-B execution only; prior maps describe unchanged producer dependencies, not old scientific identities or authorizations')
    bind['source_sha256'].update({p.relative_to(ROOT).as_posix():sha(p) for p in HOME.glob('*.py')})
    for f in ('__init__.py','contracts.py','adapters.py'):bind['source_sha256']['rc3_r08_joint_decision_design_20261113/'+f]=sha(ROOT/'rc3_r08_joint_decision_design_20261113'/f)
    bind['R08-B_artifacts']={**frozen,'rc3_r07_20261113/TASKS.json':sha(ROOT/'rc3_r07_20261113/TASKS.json'),
        'rc3_r08_joint_runtime/TASKS.json':sha(HOME/'TASKS.json')}
    put(HOME/'SOURCE_BINDING.json',bind)
    ident=hashlib.sha256(json.dumps(dict(protocol=sha(HOME/'PROTOCOL.json'),source=sha(HOME/'SOURCE_BINDING.json'),tasks=sha(HOME/'TASKS.json')),sort_keys=True).encode()).hexdigest()
    put(HOME/'SCIENTIFIC_IDENTITY.json',dict(identity=ident,run_id=RUN,authorization_SHA_not_in_identity=True))
    put(HOME/'AUTHORIZATION.json',dict(authorized=True,scope='Explicit current user R08-B only request; C/D/E unauthorized',source_binding_sha256=sha(HOME/'SOURCE_BINDING.json'),protocol_sha256=sha(HOME/'PROTOCOL.json'),registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    put(HOME/'STATUS.json',dict(run_id=RUN,status='REGISTERED_BEFORE_FIRST_NEW_REQUEST',stage='B',closed=False,C_D_E='NOT_RUN_NOT_AUTHORIZED'))
    put(HOME/'RESOURCE_ACCOUNTING.json',dict(counts=dict(requests=0,requests_completed=0,requests_failed=0,actor_optimizer_attempts=0,critic_optimizer_attempts=0,new_teacher_requests=0,new_scenes=0,navigation_solves=0,native_requests=0,H_calls=0,original60_reads=0,oldS3_task_reads=0,confirmation_final_reads=0),wall_s=time.time()-origin,old_ledgers_not_reset=True))
    print(dict(status='REGISTERED',identity=ident,wall_s=time.time()-origin,remaining_s=origin+1800-time.time()),flush=True)

if __name__=='__main__':main()
