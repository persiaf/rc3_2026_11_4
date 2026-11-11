import time
START_M,START_W=time.monotonic(),time.time()
import ast
import json
import subprocess
from .common import HOME,ROOT,BASE,DATA,read,put,sha

def register():
    if (HOME/'SOURCE_BINDING.json').exists():raise FileExistsError('One registration; no resource reset')
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==BASE
    prior=read(ROOT/'rc3_actor_positive_scale_20261111/SOURCE_BINDING.json')
    sources=dict(prior['source_sha256']);artifacts=dict(prior['artifact_sha256'])
    for p in HOME.glob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'));sources[p.relative_to(ROOT).as_posix()]=sha(p)
    for name in [DATA,'rc3_actor_positive_scale_20261111/FEASIBLE_RANK_WITNESSES.json',
        'rc3_actor_positive_scale_20261111/POSITIVE_CONTROL_SUMMARY.json','rc3_actor_positive_scale_20261111/RESOURCE_ACCOUNTING.json',
        'rc3_actor_positive_scale_20261111/UPDATE_SCALE_RESULTS.csv']:
        artifacts[name]=sha(ROOT/name)
    inventory=read(ROOT/'rc3_actor_positive_scale_20261111/LFS_PUBLICATION_EXPECTED.json')['objects']
    assert artifacts[DATA]==next(o['oid_sha256'] for o in inventory if o['path']==DATA)
    protocol=dict(schema='shared-a1-witness-conditional-rank-v1',base_commit=BASE,seed=11622026,
        fixed_goal='Large complex parent DRL training; frozen Actor quickly completes local complete constrained tasks',
        data_path=DATA,data_sha256=artifacts[DATA],states='EXACT saved32 in original order, teacher/features/candidate order/mask unchanged; no pool scan or reselection',
        architecture=[206,64,64,1],amplitude=.25,distances='ON',old_witness_scope='Separate per-state weights; no old shared32-state witness',
        construction=dict(bumps_max=32,first_hidden_max=64,second_hidden_max=64,
            projections='Fixed206 coordinate axes +128 L2-normalized Rademacher +128 L2-normalized Gaussian vectors, PCG64seed11622026, float32 weights; no extra inputs/stateID/query lookup',
            candidate_count=462,attempts_max=8,choice='For each distinct exact teacher feature, stable sort candidates by decreasing minimum distinct-row projection separation divided by max(1+abs(X)@abs(p)); attempt k uses kth candidate clipped to last available for every bump; a group with zero choices fails construction; no pool changes',
            normalized_separation_min=2e-6,half_width='min distinct projection gap/3',duplicate_policy='Exact teacher feature groups merge; all occurrences and near-neighbor distances reported; no dropping conflicts',
            scalar_scale='Per-target isolated float32 spike minimal scale for margin2e-4,48 fixed bisections on[0,64*legal_count]; max over exact duplicate target group, divide by actual minimum positive float32 target bump peak; no optimizer',
            success='ONE actual float32 A1 parameter set simultaneously all32 strict margin>=1e-4, original full-legal base->witness guard, mask, finite and abs(delta)<=.25000001; then independent NumPy probability/A1 arithmetic',
            failure='No shared witness found within this construction only; not a proof shared network impossible',diagnostic_only=True),
        rank=dict(prerequisite='JOINT_FEASIBILITY.status PASS only',initial='EXACT sealed original zero-output initial_correction; never old fitted or constructed weights',
            learning_rate=3e-4,Adam=dict(betas=[.9,.999],eps=1e-8,weight_decay=0.,gradient_clip=1.),
            loss='mean32(relu(1e-4+max_other_legal_logit-teacher_logit))',order='Same fixed32 every step',attempts_max=128,
            success='EXACT step128 strict target margin>1e-5 fraction>=80%; accepted>=ceil(.9*128)=116; finite, mask, amplitude, all actually applied incremental guards',
            CE_reference='Accurate frozen original3e-4 CE step128:14/32,128 accepted; read saved results, no CE retrain',
            records='Every state margin/residual/probability change, full candidate+applied logits, rejected state+constraint, candidate model+Adam hashes; exact repeated candidates explicitly linked',
            endpoint_guard_semantics='Candidate reports compare previous applied->candidate, baseTV toSL. Descriptive applied-endpoint KL/probability columns compare SL->endpoint; these are not incremental step-guard failures.',
            diagnostic_only=True),
        guard=dict(max_step_kl=.001,max_step_probability_change=.02,max_base_total_variation=.1,consecutive_rejection_limit=16),
        caps=dict(diagnostic=dict(requests=0,actor_optimizer_attempts=128,critic_optimizer_attempts=0,wall_s=650)),
        total_wall_cap_s=900,review_wall_reserve_s=250,pre_registration_source_read_upper_s=10,
        wall_scope='Source reading/registration, real provenance, frozen-data load, construction/forwards/fitting/new-output audits and complete independent review charged; code authorship/prose/Git-LFS maintenance separate. No historical huge CSV audit.',
        stop='Source/input/nonfinite error stops and preserves failed consumption;16 consecutive rejections stop rank config; no repeat entries/refill/extra LR/steps/guard/amplitude. No online DRL trigger.',
        frozen_history='Old3 LR FAIL,323 attempts,online NOT_RUN and resource bytes preserved. All diagnostic weights forbidden formal initialization/deployment. original60/oldS3/userWord untouched.')
    put(HOME/'PROTOCOL.json',protocol);artifacts[(HOME/'PROTOCOL.json').relative_to(ROOT).as_posix()]=sha(HOME/'PROTOCOL.json')
    binding=dict(schema='shared-a1-rank-source-v1',source_sha256=sources,artifact_sha256=artifacts,
        environment=prior['environment'],protocol_sha256=sha(HOME/'PROTOCOL.json'),base_commit=BASE)
    put(HOME/'SOURCE_BINDING.json',binding)
    for name in ['PROTOCOL_APPROVAL.json','DIAGNOSTIC_AUTHORIZATION.json']:
        put(HOME/name,dict(approved=True,authorized=True,authority='Explicit current user task',
            source_binding_sha256=sha(HOME/'SOURCE_BINDING.json'),scope='Shared construction then strictly conditional rank diagnostic; no online/route requests'))
    put(HOME/'REGISTRATION_RECEIPT.json',dict(status='FROZEN_BEFORE_FIRST_SCIENTIFIC_COMPUTATION',
        registration_s=max(time.monotonic()-START_M,time.time()-START_W),
        charged_wall_s=max(time.monotonic()-START_M,time.time()-START_W)+10.,actor_attempts=0,requests=0))
    print('Source, exact sealed data, projections/attempts/tolerances, original success standards and900s cumulative budget registered')

def launch():
    if (HOME/'diagnostic_control').exists():raise FileExistsError('No repeated entry or resource reset')
    binding=read(HOME/'SOURCE_BINDING.json')
    for name in ['__init__.py','supervisor.py']:
        p=ROOT/'rc3_r04_m1_recovery_20261111'/name;assert sha(p)==binding['source_sha256'][p.relative_to(ROOT).as_posix()]
    from rc3_r04_m1_recovery_20261111.supervisor import launch_supervised
    prior=read(HOME/'REGISTRATION_RECEIPT.json')['charged_wall_s']
    receipt=launch_supervised([str(ROOT/'rc3_learning/.venv/Scripts/python.exe'),'-B','-X','utf8','-m',HOME.name+'.worker'],
        HOME/'diagnostic_control','shared-a1-rank-20261111-diagnostic-01',wall_limit_s=650-prior-2,
        closeout_reserve_s=15,termination_tolerance_s=2,cwd=ROOT,entry_monotonic=START_M,entry_wall=START_W,
        worker_heartbeat_path=HOME/'diagnostic/WORKER_HEARTBEAT.json')
    elapsed=max(time.monotonic()-START_M,time.time()-START_W)
    put(HOME/'ENTRY_CONSUMPTION.json',dict(status=receipt['status'],entry_wall_s=elapsed,prior_registration_source_read_charged_s=prior,
        core_charged_wall_s=elapsed+prior,root_exit_confirmed=receipt['root_exit_confirmed'],active_processes_at_end=receipt['job_active_processes_at_end']))
    print(dict(status=receipt['status'],core_charged_wall_s=elapsed+prior));raise SystemExit(0 if receipt['status']=='COMPLETE' else 1)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run']);a=p.parse_args()
    register() if a.action=='register' else launch()
