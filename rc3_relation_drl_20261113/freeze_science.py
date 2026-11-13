"""Freeze M2 identities and an untrained independent random Actor; never train.

One-shot preparation only. No dataset construction, model forward, optimizer,
teacher query, H, or scientific stage launch is performed here.
"""
from .common import ROOT, HOME, H_ID, read, put, sha, verify_binding
from .register_engineering import local_dependencies


def ref(path):
    path = ROOT / path
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path))


def main():
    if (HOME/'M2_PROTOCOL.json').exists() or (HOME/'M2_RANDOM_INIT.pt').exists():
        raise RuntimeError('Scientific registration already exists; preserve it and register any revision separately')
    engineering = verify_binding(HOME/'ENGINEERING_SOURCE_BINDING.json')
    result = read(HOME/'engineering/M1_FINAL_RESULT.json')
    if result['status'] != 'ENGINEERING_PASS':
        raise ValueError('Completed M1 required for this registration')
    from .runtime import setup, export_actor
    from .actor import RelationActor
    setup(2026111301, 'cuda')
    identity = 'relation_navcover_m2_v1_20261113'
    actor = RelationActor().to('cuda')
    initial = export_actor(actor, HOME/'M2_RANDOM_INIT.pt', False, identity)
    put(HOME/'M2_RANDOM_INIT.json', dict(schema='relation-scientific-random-init-v1', scientific_id=identity,
        seed=2026111301, role='UNTRAINED_RANDOM_ONLY', export=initial, config=actor.config,
        parameters=sum(p.numel() for p in actor.parameters()), parameter_tensors=len(list(actor.parameters())),
        model_forwards=0, optimizer_updates=0, engineering_state_loaded=False,
        future_SL512='Produced only by authorized init stage and bound by its immutable COMPLETE.json; not yet available'))
    del actor
    from .evaluate import METHODS
    cases = read(HOME/'M2_EVALUATION_CASES.json')
    grid = []
    for endpoint in (0, 512, 2048):
        for population, ids in (('LOCAL66', [x['id'] for x in cases['LOCAL66']]),
                                ('FULL12', cases['FULL12_ids'])):
            for tid in ids:
                grid += [dict(endpoint=endpoint, method=method, population=population, case_id=tid) for method in METHODS]
    assert len(grid) == 1404
    put(HOME/'M2_EVALUATION_GRID.json', dict(schema='relation-M2-evaluation-grid-v1', requests=grid,
        normal_count=1404, additional_technical_allowance=596, automatic_retry=False))
    files = dict(engineering['files'])
    files.update(local_dependencies([HOME/x for x in ('m2.py', 'evaluate.py', 'freeze_science.py', 'register_data.py', 'test_protocol.py')]))
    from rc3_conservative_refinement.saved_inputs import ANCHORS
    from rc3_conservative_refinement.runtime import SL_PATH, SL_SHA256, UPSTREAM
    metadata = ['rc3_d5_revision_20261112/'+n for n in ('FULL_TASK_REGISTRATION.json', 'ORIGINAL_LOCAL_TEACHERS.json',
                'REVISION_PROTOCOL.json', 'FULL_TEACHER_MANIFEST.json', 'REFERENCE_MANIFEST.json', 'JOINT_DATA_INDEX.json', 'JOINT_DATA_ADMISSION.json')]
    metadata += list(ANCHORS)
    metadata += [SL_PATH, UPSTREAM['path'], UPSTREAM['sidecar_path'], UPSTREAM['commit_path']]
    metadata += ['rc3_relation_drl_20261113/'+n for n in ('ENVIRONMENT.json', 'M2_DATA_SCHEDULE.json', 'M2_EVALUATION_CASES.json',
                 'M2_EVALUATION_GRID.json', 'M2_RANDOM_INIT.pt', 'M2_RANDOM_INIT.json', 'engineering/M1_FINAL_RESULT.json')]
    for path in metadata:
        files[path] = sha(ROOT/path)
    if files[SL_PATH] != SL_SHA256 or any(files[path] != expected for path, expected in ANCHORS.items()):
        raise ValueError('Historical source/data/model anchor changed')
    binding = dict(schema='relation-scientific-source-v1', scientific_id=identity,
        base_commit=engineering['base_commit'], files=dict(sorted(files.items())), H_identity=H_ID,
        input_semantics='Hash-pinned original SavedInputs per-source files and teacher/ref manifests; no source regeneration',
        authorization_sha_in_scientific_identity=False)
    put(HOME/'SCIENTIFIC_SOURCE_BINDING.json', binding)
    from .m2 import EXPECTED_LIMITS
    protocol = dict(schema='relation-M2-protocol-v1', status='FROZEN_NOT_AUTHORIZED', scientific_id=identity,
        source_binding=ref('rc3_relation_drl_20261113/SCIENTIFIC_SOURCE_BINDING.json'),
        schedule=ref('rc3_relation_drl_20261113/M2_DATA_SCHEDULE.json'),
        evaluation_cases=ref('rc3_relation_drl_20261113/M2_EVALUATION_CASES.json'),
        evaluation_grid=ref('rc3_relation_drl_20261113/M2_EVALUATION_GRID.json'),
        runtime_environment=ref('rc3_relation_drl_20261113/ENVIRONMENT.json'),
        initialization=dict(seed=2026111301, random_export=initial, steps=512, batch=8,
            paired_start='Exact init COMPLETE.json export; identical weights and independently fresh Adam for each stream'),
        sampling_seed=2026111303, device='cuda', reference_floor=1.0,
        m1_receipt=ref('rc3_relation_drl_20261113/engineering/M1_FINAL_RESULT.json'), limits=EXPECTED_LIMITS,
        mechanism='Explicit true navigation D and dynamic remaining-task/arm coverage E relations only',
        retained=dict(H_identity=H_ID, objective='J = L + 0.75 K', navigation='original undirected RC3',
                      mask='original positive coverage', validation='original complete independent raw/H validation'),
        optimizer=dict(name='Adam', lr=1e-4, betas=[.9,.999], eps=1e-8, weight_decay=0, grad_clip_norm=1., entropy=0),
        training=dict(RL_steps=2048, SL_steps=2048, tasks_per_update=4, trajectories_per_task=4,
                      baseline='same-task leave-one-out', loss='mean_B_G ((J_H - mean_other_J_H)/max(saved_FrozenSL_J_H,1)) * sum_logprob',
                      online_trajectories=32768, gradient_full_replays=32768, continued_SL_full_replays=8192,
                      SL_loss='mean_task negative mean_action logprob', empty_training_case='technical failure; no substitution'),
        evaluation=dict(endpoints=[0,512,2048], methods=list(METHODS), LOCAL66=66, FULL12=12,
                        full_ids=cases['FULL12_ids'], populations_separate=True, all_exposed=True,
                        aggregation='within-parent mean J, relative reference gain, then equal mean over parents',
                        cold_setup='reported separately from resident full-request latency; each request includes output persistence'),
        gates=dict(endpoint_512='Stop if either population overall gain vs RelationInit < -0.03',
            endpoint_2048=dict(LOCAL66_vs_FrozenSL_min=.01, LOCAL66_vs_RelationInit_min=.005, LOCAL66_vs_RelationSL_min=.005,
                LOCAL66_SEL_DEV_each_scale_vs_Init_and_SL_min=0., FULL12_vs_FrozenSL_Init_SL_min=0., all_complete_valid=True),
            incomplete='INSUFFICIENT; never success-subset averages', failure='NO_DEVELOPMENT_SIGNAL at fixed full endpoint; no extension'),
        recovery=dict(boundary='Atomically committed STATE.json after each update or complete endpoint',
            checkpoint='model, Adam, CPU/CUDA/Python/NumPy RNG, cursor, request IDs, counts, wall; two inactive-slot writes',
            sampling='per-request deterministic hash-derived generator from base/update/slot/trajectory',
            paid_uncommitted='TECHNICAL_INCOMPLETE; retain debit, no automatic replay',
            identity='canonical directory, same original stage authorization and cumulative resource ledger'),
        scope=dict(M2_executed=False, protocol_approved=False, stage_authorized=False, scientific_updates=0,
                   engineering_models_allowed=False, MATLAB_data_or_weights_allowed=False,
                   future_seeds_time_matched_SL_ablations='Only new protocols after main scientific PASS'))
    put(HOME/'M2_PROTOCOL.json', protocol)
    put(HOME/'M2_FREEZE_RECEIPT.json', dict(status='FROZEN_NOT_AUTHORIZED', protocol=ref('rc3_relation_drl_20261113/M2_PROTOCOL.json'),
        scientific_id=identity, source_binding=protocol['source_binding'], random_initialization=initial,
        preparation=dict(random_model_constructions=1, model_forwards=0, optimizer_updates=0, complete_requests=0, new_H=0),
        next='Stop at M1; exact protocol approval and independent init-stage authorization required'))
    print(dict(status=protocol['status'], scientific_id=identity, source_files=len(files), protocol_sha256=sha(HOME/'M2_PROTOCOL.json'), initial=initial))


if __name__ == '__main__':
    main()
