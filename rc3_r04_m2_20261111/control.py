"""Register exact sources before computation; reuse accepted Windows Job supervisor."""
import time
START_M, START_W = time.monotonic(), time.time()
import argparse
import ast
import json
import subprocess
import sys
from .common import HOME, ROOT, OLD, BASE, RUN_ID, read, put, sha

def git_bytes(path):
    return subprocess.check_output(['git', 'show', BASE+':'+path], cwd=ROOT)

def register():
    if (HOME/'SOURCE_BINDING.json').exists(): raise FileExistsError('Registered once; no resource reset')
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == BASE
    previous = read(OLD/'SOURCE_BINDING.json')
    assert sha(OLD/'SOURCE_BINDING.json') == '8806070b0c8937ceb6fb8e44bf2b1f1c788905ac4c1ad6b6945d837235a78dbf'
    sources = dict(previous['source_sha256'])
    for directory in (ROOT/'rc3_r04_m1_real_20261111', HOME):
        for p in sorted(directory.glob('*.py')):
            ast.parse(p.read_text(encoding='utf-8-sig'))
            relative = p.relative_to(ROOT).as_posix()
            actual = sha(p)
            if relative in sources: assert sources[relative] == actual
            sources[relative] = actual
    artifacts = dict(previous['artifact_sha256'])
    input_names = ['SOURCE_BINDING.json', 'RESTORED_INPUT_INDEX.json', 'REAL_PREFLIGHT.json',
        'M1_DATA_AND_SOURCE_IDENTITY.json', 'M1_EXECUTION_PROTOCOL.json', 'BATCH_SCHEDULE.json',
        'STAGE_AUTHORIZATION.json', 'M1_GATE_DECISION.json', 'CHECKPOINT_INTERFACE_RESULTS.json',
        'M1_RESTORED_DATASET.pt', 'models/C1_step0128.export.pt', 'FIT_ONLY_REFERENCES.json']
    identities = {}
    import hashlib
    for name in input_names:
        p = OLD/name; relative = p.relative_to(ROOT).as_posix(); payload = git_bytes(relative)
        if payload.startswith(b'version https://git-lfs.github.com/spec/v1'):
            expected = next(line.split(':')[1] for line in payload.decode().splitlines() if line.startswith('oid sha256:'))
        else: expected = hashlib.sha256(payload).hexdigest()
        assert sha(p) == expected, relative
        artifacts[relative] = expected; identities[name] = dict(path=relative, sha256=expected, bytes=p.stat().st_size)
    resource_path = ROOT/'rc3_r04_m1_real_20261111/RESOURCE_ACCOUNTING.json'
    assert resource_path.read_bytes() == git_bytes(resource_path.relative_to(ROOT).as_posix())
    old_resource = read(resource_path); put(HOME/'OLD_RESOURCE_SNAPSHOT.json', old_resource)
    protocol = dict(schema='rc3-r04-m2-protocol-v1', base_commit=BASE,
        purpose='deletion test: fit-only initial conditional statistics versus learned residual',
        fixed_research_goal='large-parent DRL; frozen policy rapidly produces complete variable-local-task routes',
        scope='exposed holdout DEVELOPMENT MECHANISM SCREEN; no independent generalization or online route claim',
        systems=['B-global0', 'B-cond0', 'CG0', 'CB0', 'CG128', 'CB128', 'C1_fixed128'],
        inputs=identities, seed=11622026, network=[268, 128, 128, 3],
        runtime_inputs='same268 saved observable features plus width,height for anchor; no targets',
        condition='size x min(3,max(0,floor(4*(1-clamp(v[256],0,1)))))',
        condition_estimator='33 fit parents equal / trajectories equal / within-stratum states mean',
        unknown_condition='fit-global; no resampling', normalization='fixed initial taskcount n; no empirical scale',
        positive_link='softplus(invsoftplus(max(b,1e-6))+u).clamp_min(1e-6)',
        inverse_softplus='b+log(-expm1(-b))', zero_value_floor=1e-6,
        H='signed bH+uH', value='-(L/n+0.75*K/n-H/n); composed from SAME components',
        initialization='identical seeded hidden/net tensors, ZERO output weight and bias; fit-only anchors external to net',
        optimizer=dict(name='Adam', lr=.0001, betas=[.9,.999], eps=1e-8, weight_decay=0., gradient_clip=1.),
        fitting='0.5 sum component normalized squared error; state_SUM per trajectory then episode_MEAN; no extra Value loss',
        schedule_path='rc3_r04_m1_real_20261111/attempts/02/BATCH_SCHEDULE.json',
        schedule='first128 of existing common fixed schedule; 8 distinct fit parents per batch; no shuffle/search',
        endpoints=[0,128], per_arm_steps=128, C1='exact saved step128 export; no retraining',
        caps=dict(critic_optimizer_attempts=256, actor_optimizer_attempts=0, wall_s=1800,
            new_scenes=0, navigation_requests=0, teacher_requests=0, native_requests=0, H_requests=0,
            original60_reads=0, oldS3_reads=0),
        engineering='zero-update new link/runtime tests only; reuse accepted supervisor tests and real Torch verifier',
        engineering_tolerance=dict(atol=2e-7, rtol=1e-6), numerical_audit_tolerance=dict(atol=2e-10, rtol=2e-10),
        inference='both state_pooled and parent_equal: within each parent pool its states',
        bootstrap=dict(unit='physical_parent', paired=True, draws=20000, generator='PCG64', seed=11622026, quantile='linear'),
        main_pairs=[['CB128','B-cond0'],['CB128','CG128'],['CG128','B-global0'],['CB128','C1_fixed128'],['CG128','C1_fixed128']],
        continuation_standard=dict(CB='ALL: in BOTH aggregations Value ratio<=0.95 AND paired95%upper<1 vs BOTH B-cond0 and CG128; each normalized component ratio<=1.10 vs both; each size>=3 parents and Value ratio<=1 vs both',
            CG_diagnostic='Value ratio<=0.95 AND paired95%upper<1 vs B-global0 in BOTH aggregations; diagnostic only, no automatic promotion',
            C1='mandatory same-budget reference; beating C1 alone never establishes learned residual value',
            failed_CB='STOP this candidate; no steps/conditions/network additions, no jointPPO/R03/oldM3'),
        M1=dict(G3='FAIL', step256='NOT_RUN', old_resources='preserved unchanged'),
        wall_accounting='active registration, engineering/imports/data hash/load, fit/evaluate/save/recheck and independent scientific review; code/prose/Git excluded; all failures retained',
        old_resource=dict(path=resource_path.relative_to(ROOT).as_posix(), sha256=sha(resource_path),
            wall_s=old_resource['cumulative_all_research_accounted_wall_s'], critic_attempts=old_resource['counts']['critic_optimizer_attempts']),
        execution_design='separate M2 authorized by current user task; historical v2/v3 active pointers unchanged',
        H_identity='5db67cacc1128b6143c7a04b9b84a70c37bdc0ab917a0eebae0b8aca132c4d14')
    put(HOME/'M2_PROTOCOL.json', protocol)
    artifacts[(HOME/'M2_PROTOCOL.json').relative_to(ROOT).as_posix()] = sha(HOME/'M2_PROTOCOL.json')
    accepted = ROOT/'rc3_r04_m1_recovery_20261111/engineering/ENGINEERING_TEST_RESULTS.json'
    assert sha(accepted) == previous['accepted_engineering_receipt']['sha256']
    artifacts[accepted.relative_to(ROOT).as_posix()] = sha(accepted)
    binding = dict(schema='rc3-r04-m2-exact-source-binding-v1', base_commit=BASE, source_sha256=sources,
        artifact_sha256=artifacts, environment=previous['environment'], protocol_sha256=sha(HOME/'M2_PROTOCOL.json'),
        prior_binding_sha256=sha(OLD/'SOURCE_BINDING.json'), inputs=identities,
        accepted_engineering_receipt=dict(path=accepted.relative_to(ROOT).as_posix(), sha256=sha(accepted)),
        scientific_identity_excludes_downstream_authorization=True)
    put(HOME/'SOURCE_BINDING.json', binding)
    put(HOME/'IMPLEMENTATION_REVIEW.json', dict(status='GO', source_binding_sha256=sha(HOME/'SOURCE_BINDING.json'),
        scope='offline critic only; environment/reward/mask/endpoint/H unchanged; no actor or planner imports required',
        authorization_is_separate=True))
    put(HOME/'STAGE_AUTHORIZATION.json', dict(authorized=True, authority='explicit user R04-M2 task', run_id=RUN_ID,
        source_binding_sha256=sha(HOME/'SOURCE_BINDING.json'), protocol_sha256=sha(HOME/'M2_PROTOCOL.json'),
        max_critic_optimizer_attempts=256, per_arm=128, actor_optimizer_attempts=0, task_wall_cap_s=1800))
    put(HOME/'REGISTRATION_RECEIPT.json', dict(status='REGISTERED_BEFORE_FIRST_NUMERICAL_COMPUTATION',
        wall_s=max(time.monotonic()-START_M,time.time()-START_W), optimizer_attempts=0,
        source_binding_sha256=sha(HOME/'SOURCE_BINDING.json'), protocol_sha256=sha(HOME/'M2_PROTOCOL.json')))
    print('M2 exact protocol, sources, data, thresholds and separate authorization REGISTERED')

def launch():
    if (HOME/'ENTRY_CONSUMPTION.json').exists() or (HOME/'control').exists(): raise FileExistsError('No automatic duplicate entry')
    binding = read(HOME/'SOURCE_BINDING.json')
    # Verify the accepted supervisor itself before invoking it, without importing Torch.
    for name in ('__init__.py','supervisor.py'):
        p = ROOT/'rc3_r04_m1_recovery_20261111'/name
        assert sha(p) == binding['source_sha256'][p.relative_to(ROOT).as_posix()]
    from rc3_r04_m1_recovery_20261111.supervisor import launch_supervised
    prior = read(HOME/'REGISTRATION_RECEIPT.json')['wall_s']
    receipt = launch_supervised([sys.executable, '-m', HOME.name+'.worker'], HOME/'control', RUN_ID,
        wall_limit_s=1790-prior, closeout_reserve_s=20, termination_tolerance_s=5,
        cwd=ROOT, entry_monotonic=START_M, entry_wall=START_W, worker_heartbeat_path=HOME/'WORKER_HEARTBEAT.json')
    wall = max(time.monotonic()-START_M,time.time()-START_W)
    put(HOME/'ENTRY_CONSUMPTION.json', dict(status=receipt['status'], wall_s=wall,
        registration_wall_s=prior, cumulative_new_task_wall_s=prior+wall,
        old_consumption_preserved_s=read(HOME/'M2_PROTOCOL.json')['old_resource']['wall_s'],
        root_exit_confirmed=receipt['root_exit_confirmed'], job_active_processes_at_end=receipt['job_active_processes_at_end']))
    print(json.dumps(receipt, ensure_ascii=False))
    raise SystemExit(0 if receipt['status']=='COMPLETE' else 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('register','launch'))
    globals()[parser.parse_args().action]()
