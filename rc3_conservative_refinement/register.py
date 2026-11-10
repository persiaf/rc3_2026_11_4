"""Create immutable engineering registration or a concrete unapproved protocol.

This command never creates an approval receipt, trains, or generates tasks.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch

from .guard import GuardLimits
from .runtime import SL_PATH, SL_SHA256, UPSTREAM
from .saved_inputs import ANCHORS, SavedInputs
from .storage import ROOT, digest, read, write_new
from .trainer import PPOConfig

HOME=Path(__file__).resolve().parent
STAGE=HOME/'integration_20261110'
H_ID='5db67cacc1128b6143c7a04b9b84a70c37bdc0ab917a0eebae0b8aca132c4d14'


def environment():
    return dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,
                platform=platform.platform(),device='cpu',threads=1,deterministic_algorithms=True)


def engineering(revision):
    STAGE.mkdir(parents=True,exist_ok=True)
    name=f'SOURCE_BINDING_{revision}.json'
    tracked=subprocess.run(['git','ls-files','--','*.py'],cwd=ROOT,check=True,capture_output=True,text=True).stdout.splitlines()
    paths=set(tracked)|{p.relative_to(ROOT).as_posix() for p in HOME.glob('*.py')}
    sources={p:digest(ROOT/p) for p in sorted(paths)}
    artifacts=dict(ANCHORS)
    artifacts[SL_PATH]=SL_SHA256
    for key,sha_key in [('path','checkpoint_sha256'),('sidecar_path','sidecar_sha256'),('commit_path','commit_sha256')]:
        artifacts[UPSTREAM[key]]=UPSTREAM[sha_key]
    for path in ('rc3_conservative_refinement/SOURCE_BINDING.json',
                 'rc3_conservative_refinement/PROTOCOL_CANDIDATE.json',
                 'rc3_integrated_rppo/PROTOCOL_SMALL256.json',
                 'rc3_integrated_rppo/EXECUTION_PROTOCOL.json',
                 'rc3_global_local_execution_v3/ACTIVE_EXECUTION_STATUS.json', '.gitattributes'):
        artifacts[path]=digest(ROOT/path)
    for path,sha in artifacts.items():
        if digest(ROOT/path)!=sha:
            raise ValueError('Historical artifact hash differs: '+path)
    binding=dict(schema='rc3-conservative-real-integration-source-v1',revision=revision,
                 baseline='3f7507611b85dac757f126555258f32e0888dcb2',H_identity=H_ID,
                 source_sha256=sources,artifacts=artifacts,environment=environment(),
                 old_training_entries_invoked=False,original60_semantic_access=0)
    write_new(STAGE/name,binding)
    prior=[];used={};wall=0.
    for p in sorted(STAGE.glob('engineering_*/RESOURCE_ACCOUNTING.json')):
        r=read(p);journal=p.parent/'events.jsonl'
        prior.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=digest(p),
                          ledger_path=journal.relative_to(ROOT).as_posix(),ledger_sha256=digest(journal)))
        import json
        events=[json.loads(line) for line in journal.read_bytes().splitlines()]
        wall+=sum(e['elapsed_s'] for e in events if e['event']=='session_close')
        for k,n in r['counts'].items():
            used[k]=used.get(k,0)+n
    caps={k:limit-used.get(k,0) for k,limit in dict(requests=32,actor_optimizer_attempts=16,critic_optimizer_attempts=16,teacher_episodes=2).items()}
    caps['wall_s']=max(0.,1800-wall)
    plan=dict(schema='rc3-real-integration-engineering-plan-v1',status='REGISTERED_ENGINEERING_ONLY',revision=revision,
              base_commit=binding['baseline'],source_binding_path=(STAGE/name).relative_to(ROOT).as_posix(),
              source_binding_sha256=digest(STAGE/name),output=(STAGE/f'engineering_{revision}').relative_to(ROOT).as_posix(),
              prior_attempts=prior,caps=caps,seed=11622026,environment=environment(),
              cases='SavedInputs.engineering_cases: first saved10x10 plus development0/2/4',
              planned_requests=20,planned_actor_calls=5,planned_critic_calls=4,planned_teacher_replays=1,
              tolerances=dict(zero_logits='torch.equal',zero_routes='exact IDs and scalar metrics equality',
                              real_feature_summary_m=1e-10,continuous_resume='exact tensors/Adam/RNG/cursor/route',
                              old_batched_logits_atol=1e-6,raw_dense_rewards_atol=1e-7),
              tests=['all synthetic tests with Adam/SGD.step forbidden','4real inputs xON/OFF/plainSL zero equivalence',
                     'real accepted Adam and checkpoint roundtrip','real continuous versus restored second update',
                     'separate zero-limit strict rejection fixture with exact head/Adam rollback',
                     'actor-only cold deployment complete raw/H route equality',
                     'one real saved-label matchedSL guarded step','one native Fixed3 and one native Detour16 complete validation'],
              formal_training_authorized=False,formal_optimizer_cap=0,
              sealed=dict(original60=True,S3=True,unapproved_final=True))
    write_new(STAGE/f'ENGINEERING_PLAN_{revision}.json',plan)
    print('REGISTERED '+str(STAGE/f'ENGINEERING_PLAN_{revision}.json'))


def proposal(revision):
    plan_path=STAGE/f'ENGINEERING_PLAN_{revision}.json';plan=read(plan_path)
    folder=ROOT/plan['output'];resource=read(folder/'RESOURCE_ACCOUNTING.json')
    if resource['status']!='PASS':
        raise ValueError('Cannot propose execution from failed real engineering')
    inputs=SavedInputs()
    schedule=[dict(opportunity=i,specs=inputs.training(11622026,i),teacher_indices=inputs.teacher_indices(11622026,i)) for i in range(1,257)]
    schedule_path=STAGE/f'SCHEDULE_{revision}.json'
    write_new(schedule_path,dict(seed=11622026,training=schedule,evaluation=inputs.evaluation_cases,
                                 physical_groups=inputs.physical_groups,source_anchors=ANCHORS))
    protocol=dict(schema='rc3-conservative-paired-formal-protocol-v1',status='PENDING_EXPLICIT_PARAMETER_APPROVAL',
                  experiment='conservative_paired_11622026_v1',baseline='3f7507611b85dac757f126555258f32e0888dcb2',
                  source_binding_path=plan['source_binding_path'],source_binding_sha256=plan['source_binding_sha256'],
                  engineering_gate=dict(resource_path=(folder/'RESOURCE_ACCOUNTING.json').relative_to(ROOT).as_posix(),
                                        resource_sha256=digest(folder/'RESOURCE_ACCOUNTING.json'),
                                        checks_path=(folder/'CHECKS.json').relative_to(ROOT).as_posix(),checks_sha256=digest(folder/'CHECKS.json')),
                  schedule_path=schedule_path.relative_to(ROOT).as_posix(),schedule_sha256=digest(schedule_path),
                  seed=11622026,environment=environment(),H_identity=H_ID,
                  actor=dict(input_dim=206,hidden=[64,64],amplitude=.25,zero_last_layer=True,
                             frozen_SL_path=SL_PATH,frozen_SL_sha256=SL_SHA256,
                             on_off_difference='Only new mean/max summary inputs zeroed in OFF'),
                  critic=dict(input_dim=268,hidden=[128,128],outputs=['raw_remaining_L','raw_remaining_K','terminal_H_credit'],
                              initialization_seed=11622026,training_only=True),
                  ppo=asdict(PPOConfig()),guard=asdict(GuardLimits()),
                  opportunities=256,checkpoints=[64,128,256],
                  warm=dict(episodes_per_condition=24,updates_per_condition=32,
                            own_zero_policy_trajectories=True,actor_updates=0,steps_per6episode_batch=8),
                  training=dict(episodes_per_opportunity=6,full10x10=3,local4x3=1,local6x6=1,local8x8=1,
                                actor_candidates_per_opportunity=1,critic_updates_per_opportunity=1,
                                gamma=1.,gae_lambda=1.,advantage='Full G - frozen V_old; one RMS across valid states, no centering',
                                no_post_update_off_policy_sharing=True),
                  matched_SL=dict(input_dim=206,distances_enabled=False,teacher_labels_per_opportunity=6,
                                  loss='Mean route CE, then episode mean',same_guard=True,
                                  same_initialization=True,opportunities=256,
                                  match_scope='Opportunity/Adam/correction/guard budget; accepted updates and teacher inputs may differ; not FLOPs-matched'),
                  evaluation=dict(cases=66,physical_parents=11,trained_parents=8,independent_parents=3,
                                  all_exposed_development=True,methods=['FrozenSL','continued_SL','ON','OFF','Fixed3','Detour16'],
                                  repeats_by_endpoint={'64':1,'128':1,'256':3},requests=1980,
                                  metric_primary='Unnormalized full H-J AND raw-J; L/K separately; all valid',
                                  timing='Actual saved-input full request, cold model load, new provider construction, policy,H,raw/H validation; IO separate; Critic diagnostic excluded from deployment',
                                  navigation='Already-saved obstacle-aware D and paths loaded and hash-checked per request, no fresh navigation solve',
                                  statistics='Physical parent clusters, relative savings after parent case means, percentile bootstrap20000 PCG64seed11652026; descriptive single-seed study',
                                  primary_success='ON improves FrozenSL and continued_SL with lower descriptive95% CI>0 in both cohorts; ONvsOFF also positive; one seed/3 exposed independent parents cannot establish stable deployment generalization',
                                  deployment_reference='Detour cost degradation upper95<=.005; median ratio<=.7,p95<=1; no Fixed3 simultaneous dominance; limited development evidence only'),
                  stop=dict(consecutive_guard_rejections=16,no_threshold_or_lr_relaxation=True,
                            no_gain='At two consecutive registered endpoints both ON/OFF mean gain vs FrozenSL<=0: stop configuration',
                            critic='At two endpoints either Critic state-pooled value MSE >1.25 x fixed warm RTG scalar-mean baseline: stop and retain failure; separate component means for calibration',
                            infeasible='Stop/preserve case and failure; no resampling or safety changes',
                            extra_training='Forbidden beyond256; early stops not refilled',
                            time_cost='Report full-request regression; Actor-only forward cannot substitute'),
                  caps=dict(requests=5100,actor_optimizer_attempts=768,critic_optimizer_attempts=576,
                            teacher_episodes=1536,wall_s=18000),
                  output=(STAGE/'runs'/'conservative_paired_11622026_v1').relative_to(ROOT).as_posix(),
                  authorization_required=['Independent exact protocol approval','Separate stage execution receipt with same caps'],
                  old_authorizations_reused=False,original60_access=0,S3_access=0,final_unmaterialized_inputs_access=0)
    write_new(STAGE/f'PROTOCOL_DRAFT_{revision}.json',protocol)
    print('PENDING_REVIEW '+str(STAGE/f'PROTOCOL_DRAFT_{revision}.json'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['engineering','proposal'])
    p.add_argument('--revision',default='R01')
    args=p.parse_args()
    if not args.revision.startswith('R') or not args.revision[1:].isdigit():
        raise ValueError('Use an explicit R-number revision')
    (engineering if args.action=='engineering' else proposal)(args.revision)


if __name__=='__main__':
    main()
