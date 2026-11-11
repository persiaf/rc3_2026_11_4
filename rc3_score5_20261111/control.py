import time
START_M,START_W=time.monotonic(),time.time()
import ast
import argparse
import subprocess
from pathlib import Path
from .common import HOME,ROOT,BASE,read,put,sha,admit

def register():
    if (HOME/'SOURCE_BINDING.json').exists():raise FileExistsError('No registration reset')
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==BASE
    prior=read(ROOT/'rc3_frozen_reference_pg_20261111/execution_v2/SOURCE_BINDING.json')
    sources=dict(prior['source_sha256']);artifacts=dict(prior['artifact_sha256'])
    for p in HOME.glob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'));sources[p.relative_to(ROOT).as_posix()]=sha(p)
    previous=read(ROOT/'rc3_frozen_reference_pg_20261111/SCHEDULE.json')
    schedule=dict(seed=11622026,training=previous['training'][:128],evaluation=previous['evaluation'],engineering=[previous['training'][0]['specs'][j] for j in (0,3,4,5)],
        distribution='PG128x(3 full10x10+1 each4x3/6x6/8x8),96 training physical layouts. SL768 existing labels, local-only256 pool over96 layouts, opportunity/parameters matched but not identical task distribution/FLOPs.',
        independent_retained_test='NOT_AUTHORIZED')
    put(HOME/'SCHEDULE.json',schedule)
    protocol=dict(schema='score5-frozen-representation-complete-return-PG-v1',base_commit=BASE,experiment_id='score5_original_actor_v1_20261111',
        status='ENGINEERING_AUTHORIZED_SCIENCE_PROPOSED_NOT_AUTHORIZED',fixed_goal='large parent DRL training; freeze Actor for complete local constrained planning',
        allowlist=['key.weight','dynamic.weight','nav.weight','query.weight','query.bias'],trainable_scalars=25152,frozen_scalars=39936,
        frozen=['global_net','task_in','candidate_in','to_task','to_candidate','task_norm','candidate_norm','prefix_GRU'],
        source='Exact legally bound SL256_0256_1817f58aa650.pt, SHA60f5aa4e59983e92d565f0599823b710965cf2c96a041e7eb483d23a8eb0556c; no fixed32 diagnostic weights',
        original_score='Original64-wide GlobalLocalPolicy encode/query/key/dynamic/nav/prefix/10*tanh/mask. No detached206 residual input, no additional scoring network',
        gradient_cache='Frozen representation/no_grad and frozen prefix history detached; grad-enabled keys recomputed from current key.weight every score. No trainable graph cache. Numerical inference keys only within exact model/problem/revision/tensor-version/epoch. Every update/rollback/restore invalidates cache; fresh encode per replay/request.',
        problem='Unchanged task/window/candidate/nav source, legal positive-gain mask, coverage, endpoints,J=L+.75K,frozenH and greedy export',
        return_rule='gamma=lambda1 complete Environment reward-to-go, terminal exit and H credit included; G normalized once by initial n exactly like prior cost helper; report unnormalized fullJ/L/K',
        PG=dict(reference='Existing frozen B-cond fit-only33-parent M2 table; observable size/remaining progress only, unknown->fit-global. Already normalized; no neuralCritic/no update/no deployment selection.',
            loss='-episode_MEAN state_SUM logpi(action)*detach((G-Bcond)/batch_RMS); no TD,entropy,old base-KL penalty,extra epoch or ratio-clip mechanism',
            on_policy='Current own Actor independently samples each complete episode; source tensor hash checked on replay; no updated/off-policy trajectory sharing'),
        SL=dict(loss='episode_MEAN state_MEAN teacher CE on existing6-label schedule',matching='same5 tensors/initialization/optimizer/stability/update opportunities; labels local-only, PG50%full; NOT identical data distribution,FLOPs or full-supervision control'),
        stability=dict(Adam=dict(lr=3e-5,betas=[.9,.999],eps=1e-8,weight_decay=0),gradient_clip=1.,one_candidate_per_opportunity=True,
            actual_incremental_KL_max=.001,actual_incremental_max_probability_change=.02,reference='PREVIOUS policy on current batch, same forPG/SL',
            no_added_residual_amplitude=True,no_permanent_SL_TV_bound=True,no_base_KL_regularization=True,
            reject='Exact5 tensors+Adam restore, used episode IDs/cursor/cost remain charged; caches invalid even after restored bytes',stop_after_consecutive_rejects=16),
        caps=dict(engineering=dict(requests=24,actor_optimizer_attempts=12,critic_optimizer_attempts=0,wall_s=600),science=dict(requests=1428,actor_optimizer_attempts=256,critic_optimizer_attempts=0,wall_s=7200)),
        engineering=dict(planned_requests=14,planned_Actor_attempts=5,synthetic_or_teacher_Top1_quality_gate=False,
            checks=['initial logits/mask/raw+H routes equalSL in4sizes','all5 gradients finite nonzero/frozen unchanged','optimizer whitelist/foreign instance rejection',
                'no_grad cache re-enters grad and two separate backwards','realPG and existing-label SL update','stale update/rollback cache rejected',
                'exact5weight+Adam rollback after real candidate','restore model/Adam/RNG/cursor/query-resource and exact next update','Actor-only frozen export outputs/roles/stages equal'],
            numerical_tolerances='Initial/evaluation logits/mask/routes/tensors exact; collection versus recomputed current logits atol1e-6,returns1e-7; frozen parameters exact; gradients finite norm>0 in all5',
            engineering_weights_forbidden_as_scientific_initialization=True),
        science=dict(authorized=False,seed=11622026,PG_opportunities=128,SL_opportunities=128,PG_episodes=768,existing_label_replays=768,evaluation_endpoints=[64,128],
            cases_per_endpoint=66,methods=['FrozenSL','Score5-SL','Score5-PG','Fixed3','Detour16'],evaluation_requests=660,
            exposure='8 training-parent selection layouts/48 cases plus3 exposed DEV layouts/18 cases; NOT independent generalization',
            quality='At fixed128: PG vs BOTH FrozenSL and5tensorSL rawJ andHJ parent-equal saving>=.5% with saved20k parent-bootstrap95%lower>0 in each cohort; no per-size point regression; allcomplete validity/success.64 is fixed endpoint, no best checkpoint selection.',
            metrics=['raw/H J,L,K','success rate with allattempt denominator','full input/source/nav/model/features/actions/H/export/validation/I/O time,p50,p95','raw/H stop-set change,full-sequence change,same-set order change'],
            native='Fixed3/Detour16 original full pipeline, no extra H; native raw is final native return, not preH ablation',
            timing='Same endpoint contemporaneous methods, rotated order; no oldtiming substitute, filesystem cache not controlled, not final paired deployment speed confirmation',
            stop='source/input/nonfinite/overspend stops with no sample replacement;16 paired consecutive rejections stops; fixed128 ends, no256. Quality NOT_MET => stopconfiguration.',
            gain_followup='Only if complete-task increment appears, CONSIDER separate same-budget fullActorPG attribution. Not approved/implemented/executed in this protocol.',
            independent_retained_eval=False,new_scenes=0,new_navigation=0,new_teacher=0,explicit_common_H_max=1164,native_requests=264),
        H_identity='5db67cacc1128b6143c7a04b9b84a70c37bdc0ab917a0eebae0b8aca132c4d14',
        history='S2,GLv4,GLv5 already ran SL-initialized allActor training; five-tensor frozen-representation scope is this candidate difference.53-row table/fixed32 closure and all old Actor/Trainer/protocol/results remain.',
        wall_scope='Engineering registration/source/input/actual requests/updates/checks charged; metadata prep failure preserved. Code authorship/protocol prose/Git maintenance separate. Science is NOT_RUN; no budget is consumed or stageactivated for it.')
    put(HOME/'PROTOCOL.json',protocol)
    for p in [HOME/'PROTOCOL.json',HOME/'SCHEDULE.json',HOME/'SAVED_TEACHER_METADATA.json',HOME/'DATA_PREPARATION_RECEIPT.json']:
        artifacts[p.relative_to(ROOT).as_posix()]=sha(p)
    put(HOME/'SOURCE_BINDING.json',dict(schema='score5-exact-source-v1',base_commit=BASE,source_sha256=sources,artifact_sha256=artifacts,environment=prior['environment'],protocol_sha256=sha(HOME/'PROTOCOL.json'),schedule_sha256=sha(HOME/'SCHEDULE.json')))
    put(HOME/'IMPLEMENTATION_REVIEW.json',dict(status='GO',scope='Necessary registered engineering only; no formal science execution authorization',source_binding_sha256=sha(HOME/'SOURCE_BINDING.json')))
    fields=dict(source_binding_sha256=sha(HOME/'SOURCE_BINDING.json'),protocol_sha256=sha(HOME/'PROTOCOL.json'),schedule_sha256=sha(HOME/'SCHEDULE.json'))
    put(HOME/'ENGINEERING_AUTHORIZATION.json',dict(authorized=True,stage='engineering',authority='Explicit current user instruction for necessary bounded engineering',**fields))
    put(HOME/'SCIENCE_AUTHORIZATION.json',dict(authorized=False,stage='science',status='NOT_AUTHORIZED',**fields))
    prep=read(HOME/'DATA_PREPARATION_RECEIPT.json')
    elapsed=max(time.monotonic()-START_M,time.time()-START_W)
    put(HOME/'REGISTRATION_RECEIPT.json',dict(status='BOUND_BEFORE_FIRST_MODEL_REQUEST_OR_OPTIMIZER',registration_s=elapsed,
        metadata_prep_s=prep['second_metadata_registration']['elapsed_s']+prep['first_metadata_inspection']['tool_wall_upper_s'],
        charged_wall_s=elapsed+prep['second_metadata_registration']['elapsed_s']+prep['first_metadata_inspection']['tool_wall_upper_s'],Actor_attempts=0,complete_requests=0))
    print('Five-tensor candidate, engineering limits, source/data and proposed1428-request route protocol frozen')

def launch(stage,authorization):
    admit(stage,authorization)
    if (HOME/(stage+'_control')).exists():raise FileExistsError('No repeated entry/resource reset')
    b=read(HOME/'SOURCE_BINDING.json')
    for name in ['__init__.py','supervisor.py']:
        p=ROOT/'rc3_r04_m1_recovery_20261111'/name;assert sha(p)==b['source_sha256'][p.relative_to(ROOT).as_posix()]
    from rc3_r04_m1_recovery_20261111.supervisor import launch_supervised
    prior=read(HOME/'REGISTRATION_RECEIPT.json')['charged_wall_s'] if stage=='engineering' else 0.;cap=read(HOME/'PROTOCOL.json')['caps'][stage]['wall_s']
    cmd=[str(ROOT/'rc3_learning/.venv/Scripts/python.exe'),'-B','-X','utf8','-m',HOME.name+'.worker',stage]
    if authorization:cmd+=['--authorization',str(Path(authorization).resolve())]
    receipt=launch_supervised(cmd,HOME/(stage+'_control'),'score5-20261111-'+stage+'-01',wall_limit_s=cap-prior-2,closeout_reserve_s=15,termination_tolerance_s=2,cwd=ROOT,
        entry_monotonic=START_M,entry_wall=START_W,worker_heartbeat_path=HOME/stage/'WORKER_HEARTBEAT.json')
    elapsed=max(time.monotonic()-START_M,time.time()-START_W)
    put(HOME/(stage.upper()+'_ENTRY_CONSUMPTION.json'),dict(status=receipt['status'],entry_wall_s=elapsed,prior_charged_s=prior,cumulative_wall_s=elapsed+prior,
        root_exit_confirmed=receipt['root_exit_confirmed'],active_processes_at_end=receipt['job_active_processes_at_end']))
    print(dict(status=receipt['status'],stage=stage,cumulative_wall_s=elapsed+prior));raise SystemExit(0 if receipt['status']=='COMPLETE' else 1)
if __name__=='__main__':
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','engineering','science','status']);p.add_argument('--authorization');a=p.parse_args()
    if a.action=='register':register()
    elif a.action=='status':print(dict(engineering=(HOME/'engineering/CHECKS.json').exists(),science_authorized=read(HOME/'SCIENCE_AUTHORIZATION.json')['authorized']))
    else:launch(a.action,a.authorization)
