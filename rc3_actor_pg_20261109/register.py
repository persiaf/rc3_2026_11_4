"""One preregistration, before exports, requests or new optimizer updates."""
import copy
import time
import subprocess
import numpy as np
from rc3_global_local_trainer.data import TrainingRegistry,TrainingSchedule
from rc3_global_local_trainer.rng import NamedStreams
from rc3_global_local.problem import array_hash
from .common import *


def register():
    began=time.perf_counter()
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,encoding='utf8').strip()==BASE,'Wrong frozen baseline')
    registry=TrainingRegistry(ROOT/'rc3_global_local_execution_v3/REGISTRY.json')
    schedule=TrainingSchedule(registry,NamedStreams(OLD_PROTOCOL,SEED))
    specs=[schedule.batch(u) for u in range(128)]
    input_refs={};cases=[]
    for spec in [s for b in specs for s in b]+list(registry.development_specs()):
        folder=OLD_RUN/'data/task_draws'/f"{spec['map_seed']}_{spec['task_seed']}_A{spec['A_condition']}"
        for filename in ('receipt.json','scene.json','arrays.npz'):
            p=folder/filename;require(p.is_file(),'Prespecified saved input missing, no replacement: '+str(p))
            input_refs[label(p)]=dict(sha256=digest(p),bytes=p.stat().st_size)
        receipt=read(folder/'receipt.json');key=receipt['map_key']
        nav=OLD_RUN/'data/navigation'/f'{key}.npz';nr=nav.with_suffix('.json')
        for p in (nav,nr):
            require(p.is_file(),'Saved navigation missing')
            input_refs[label(p)]=dict(sha256=digest(p),bytes=p.stat().st_size)
        if spec['role']=='development':
            with np.load(folder/'arrays.npz',allow_pickle=False) as z:F,A=z['F'],z['A']
            with np.load(nav,allow_pickle=False) as z:D=z['costs']
            files={'receipt':folder/'receipt.json','scene':folder/'scene.json','arrays':folder/'arrays.npz','navigation':nav,'navigation_receipt':nr}
            cases.append(dict(id=f"DEV_{len(cases):03}",group='existing_development_only',size=[spec['width'],spec['height']],
                endpoint=spec['endpoints'],physical_key=key,spec=spec,files={n:dict(path=label(p),sha256=digest(p)) for n,p in files.items()},
                original_parent_identity=receipt['parent_identity'],F_sha256=array_hash(F),A_sha256=array_hash(A),D_sha256=array_hash(D)))
    require(len(cases)==18 and {(tuple(c['size']),c['endpoint']) for c in cases[:6]}==
        {(s,e) for s in ((4,3),(6,6),(8,8)) for e in ('closed_depot','open_first_last_connected')},'Fixed six DEV coverage differs')
    old6=read(ROOT/'rc3_s3_evaluator/ENGINEERING_REGISTRATION.json')['cases']
    require(all({k:c[k] for k in ('id','size','endpoint','physical_key','files','original_parent_identity','F_sha256','A_sha256','D_sha256')}==
                {k:o[k] for k in ('id','size','endpoint','physical_key','files','original_parent_identity','F_sha256','A_sha256','D_sha256')} for c,o in zip(cases[:6],old6)),
            'Six cases differ from fixed old engineering selection')
    write_new(HOME/'INPUT_REGISTRY.json',dict(training_batches=specs,development_cases=cases,source_files=input_refs,
        source='Exactly first128 original seed11622026 S2 schedules; all tasks already saved, read-only; no outcome-based selection',
        data_rng_protocol=OLD_PROTOCOL,first_six_deployment_cases=[c['id'] for c in cases[:6]]))
    qs=[];rng=np.random.Generator(np.random.PCG64(11642026))
    for mode in ('D0','D1','D2'):
        for c in cases[:6]:
            for kind in ('cold','hot'):
                for repeat in range(3):
                    for method in rng.permutation(METHODS).tolist():
                        q=dict(seed=SEED,mode=mode,case_id=c['id'],method=method,kind=kind,repeat=repeat)
                        qs.append(dict(q,id=identity(q)))
    require(len(qs)==756 and len({q['id'] for q in qs})==756,'Deployment request count')
    write_new(HOME/'DEPLOYMENT_GRID.json',dict(order_seed=11642026,requests=qs))
    protocol=dict(schema='rc3-actor-lifecycle-and-matched-policy-gradient-screen-v1',experiment_id=EXPERIMENT,baseline=BASE,
        goal='Large complex parent-scene DRL training; frozen complete constraint-valid rapid variable-size local planning; supervision auxiliary only',
        independent_new_experiment=True,old_S3_results_and_gaps_frozen=True,old_budgets_reused=False,old60='SEALED',
        original_v3_protocol_sha256=OLD_PROTOCOL,registry_sha256=REGISTRY,H_identity=H_ID,
        deployment=dict(seed=SEED,cases=[c['id'] for c in cases[:6]],methods=list(METHODS),modes=['D0','D1','D2'],
            cache=['cold','hot'],repeats=3,complete_requests_cap=756,wall_s=1800,optimizer_updates=0,
            export_models='All3seeds x untrained/SL256/continuedSL512/RL512, exact tensors and minimal architecture/source only',
            lifecycle='D0 full checkpoint each neural request; D1 actor-only each neural request; D2 four actor-only models loaded once before requests. D2 caches parameters only, never task encoding/history/route.',
            order='Fixed D0,D1,D2; six fixed old DEV cases; cold then hot;3 repeats; PCG64 seed11642026 method permutations. Sequential lifecycle order is explicit, not a randomized causal speed benchmark.',
            timing='Record process phase wall and each mode startup separately. Requests include parsing/source checks, view/nav, model access, fresh encode, greedy/native, H, export, validation, artifact write/hash. Hot prewarm navigation outside request timer but inside1800 wall. Source/export/preparation and all IO charged inside phase; no free dynamic encoding.',
            equivalence='Exact parameter name/shape/dtype/value/tensor hash for12 exports; exact raw/final route and J/L/K for all same case/method across mode/cache/repeat; no quantization/pruning/distillation'),
        mechanism=dict(seed=SEED,updates_each=128,batch=4,common_schedule_sha256=identity(specs),
            start='Own seed SL256 exact fixed tensor; A/B separate meta materialization and fresh Adam, no inherited moments',
            A='Frozen original Trainer primary + forced_low + forced_high; weights1/.25/.25 normalized1.5; primary first logprob included, forced-first logprobs excluded; detachedLOO',
            B='Three independently sampled complete policy trajectories, first action logprobs included; weights1/1/1 normalized3; detachedLOO',
            confound='B jointly changes forced-start sampling and mixture weights as explicitly requested; not an isolated first-action-only causal ablation',
            trajectories_per_update=12,terminal_requests_each_cap=1536,terminal_requests_training_cap=3072,wall_s_each=1800,
            optimizer=dict(algorithm='Adam',lr=1e-4,betas=[.9,.999],eps=1e-8,weight_decay=0.,clip_global_norm=1.,gamma=1.,entropy_weight=0.,advantage_standardization=False),
            sample_mix='Exactly64 full and64 local batches; cyclic local4x3/6x6/8x8, same parent/task/window/A/endpoints across A/B',
            rng='Both data schedule fixed to original protocol/seed first128. A original primary/forced_low/forced_high roles; B policy_0/1/2 distinct generators; no shared mutable trajectories or reseeding within rows.',
            terminal='Original full H + export + validation, original bounded128 exact input/H/raw-route LRU; every request including hit predebited against1536. No free teacher or native queries.',
            checkpoint='Initial and every32 full model/Adam/RNG/cache/ledger; every update full before/after clip gradients and resulting parameters; all actions/masks/logprob/rewards recorded',
            forbidden=['PPO','entropy loss','KL loss','advantage normalization','new network','reward/lr/clip/batch/sample changes']),
        fixed_endpoint_dev=dict(cases=[c['id'] for c in cases],models=['SL256','A128','B128'],requests_cap=54,wall_s=600,
            pipeline='Actor-only per-request frozen greedy + common H, original full saved parent geometry, saved navigation sources; raw and postH J/L/K, complete plans and validation',
            aggregation='Three true development parent layouts, within-parent mean six cases, then equal parent-relative gains; size/endpoint and per-case tables too. Bootstrap20000 parents seed11632026; intervals descriptive, not3seeds or18independent layouts.',
            screen='Report B vs A AND B vs shared SL256 raw/postH. A positive candidate requires postH parent-mean improvement on both and bootstrap95 lower>0, all feasible/numerically valid. Otherwise REVISE; this is exploratory single-seed DEV triage, not S3 confirmation or convergence proof.'),
        total_mechanism_terminal_cap=3126,
        engineering=dict(real_optimizer_cap=0,real_model_inference_cap=0,navigation_native_H_teacher_cap=0,wall_s=120,
            scope='Synthetic tensor/schema/cache/likelihood/gradient tests only; no full scientific model forward or optimizer'),
        stop='Any invalid source, infeasible prescribed task, nonfinite value, failed validation/IO or cap stops affected phase with evidence. No retry/reseed/replacement or budget reset. New experiment fixed output directory and STARTED reject implicit reruns.',
        authority='Operationalization of current user instruction explicitly authorizing implementation, tests and bounded experiments; not claiming user pre-read generated source SHA; separate source-bound technical attestation before first science')
    write_new(HOME/'PROTOCOL.json',protocol)
    write_new(HOME/'AUTHORIZATION.json',dict(schema='rc3-current-user-task-execution-authority-v1',authorized_by='user',experiment_id=EXPERIMENT,
        protocol_sha256=digest(HOME/'PROTOCOL.json'),basis='Current user message: 保持DRL研究目标，完成推理制品提速实验与匹配策略梯度小试',
        user_scope_quotes=['在同一任务中完成实现、必要单测和有界实验。','本轮作为独立新实验，不沿用旧实验剩余额度。'],
        phases=['DEPLOYMENT','A128','B128','DEV54'],old_budget_reused=False,generated_hash_pre_read_by_user=False,
        source_technical_acceptance='Separate SOURCE_BINDING.json and ENGINEERING_VERIFICATION.json, before any science',utc=now()))
    side=read(ROOT/read(ROOT/'rc3_s3_evaluator/S3_MODEL_INDEX.json')['models'][str(SEED)]['SL256']['sidecar_path'])
    binding={k:v for k,v in side['binding'].items() if k not in ('optimizer_seed','run','phase')}
    write_new(HOME/'READ_ONLY_DATA_SOURCE_BINDING.json',dict(binding=binding,
        scope='Data interpretation provenance only. It is not old-stage execution authority; new experiment authorization is AUTHORIZATION.json. Provider is read_only and cannot generate or modify old inputs.'))
    old_manifest=read(ROOT/'project_management/GIT_MANIFEST.json')
    frozen={n:v for n,v in old_manifest['files'].items() if n.startswith('rc3_') and not n.startswith('rc3_actor_pg_20261109/')}
    write_new(HOME/'FROZEN_PRIOR_EVIDENCE.json',dict(baseline=BASE,files=frozen,old_S3_failures_not_refilled=True))
    write_new(HOME/'REGISTRATION_ACCOUNTING.json',dict(utc=now(),wall_s=time.perf_counter()-began,input_source_files=len(input_refs),
        frozen_prior_entries=len(frozen),new_task_navigation_model_native_H_teacher_optimizer=0))
    print(dict(cases=18,deploy_requests=len(qs),training_specs=512,input_files=len(input_refs)))


def freeze():
    sources=read(ROOT/'rc3_s3_connectivity_20261109/SOURCE_BINDING.json')['source_sha256'].copy()
    sources.update({label(p):digest(p) for p in HOME.glob('*.py')})
    from rc3_s3_evaluator.provenance import environment
    write_new(HOME/'SOURCE_BINDING.json',dict(schema='rc3-independent-actor-pg-source-binding-v1',utc=now(),
        source_sha256=sources,protocol_sha256=digest(HOME/'PROTOCOL.json'),registry_sha256=REGISTRY,H_identity=H_ID,
        input_registry_sha256=digest(HOME/'INPUT_REGISTRY.json'),deployment_grid_sha256=digest(HOME/'DEPLOYMENT_GRID.json'),
        authorization_sha256=digest(HOME/'AUTHORIZATION.json'),environment=environment(),old_scientific_sources_unchanged=True))
    print(digest(HOME/'SOURCE_BINDING.json'))


if __name__=='__main__':
    import sys
    {'register':register,'freeze':freeze}[sys.argv[1]]()
