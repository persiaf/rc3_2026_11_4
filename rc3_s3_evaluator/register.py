"""Static registration only. Never constructs targets, policies or routes."""
import argparse
import numpy as np
from .common import *
from .provenance import environment, verify_training_anchors, ModelCatalog
from .journal import grid
from rc3_global_local.problem import array_hash
from rc3_global_local_trainer.data import TrainingRegistry


def sources():
    names=set(read(ROOT/'rc3_navigation_compat_20261106/SOURCE_BINDING_UPDATED.json')['source_sha256'])
    for package in ('rc3_s3_evaluator','rc3_baseline','rc3_constructive','rc3_fast','rc3_global_local',
                    'rc3_global_local_impl_review','rc3_nav_value','rc3_v3','rc3_v4','rc3_improve'):
        names.update(label(p) for p in (ROOT/package).rglob('*.py'))
    return {n:digest(project_file(n)) for n in sorted(names)}


def static_registration():
    verify_training_anchors()
    path=HERE/'S3_MODEL_INDEX.json'
    if not path.exists():
        lineage_path=ROOT/'rc3_s2_results_20261107/WEIGHT_LINEAGE.json'
        status_path=ROOT/'rc3_s2_results_20261107/STAGE_STATUS.json'
        lineage,status=read(lineage_path),read(status_path)
        models={};upstream={label(lineage_path):digest(lineage_path),label(status_path):digest(status_path)};complete={}
        for s in SEEDS:
            seed=lineage['seeds'][str(s)]
            require(seed['status']=='COMPLETE','Incomplete S2')
            models[str(s)]={
                'untrained':seed['untrained_reference'], 'SL256':seed['frozen_SL_reference'],
                'continuedSL512':seed['branches']['continued_SL']['fixed_endpoint'],
                'RL512':seed['branches']['RL']['fixed_endpoint']}
            stage=next(x for x in status['seeds'] if x['seed']==s)
            require(digest(project_file(stage['stage_receipt']))==stage['stage_receipt_sha256'],'S2 COMPLETE changed')
            complete[str(s)]=stage['stage_receipt_sha256'];upstream[stage['stage_receipt']]=stage['stage_receipt_sha256']
            for name in ('fork_receipt',):upstream[seed[name]]=seed[name+'_sha256']
        write_new(path,dict(schema='rc3-s3-fixed-model-index-v1',experiment_id=EXPERIMENT,
            protocol_sha256=PROTOCOL,training_source_sha256=TRAIN_SOURCE,models=models,upstream_sha256=upstream,S2_complete_sha256=complete))
    ModelCatalog()
    i=1
    while (HERE/f'UNIT_TEST_REGISTRATION_{i:02}.json').exists():i+=1
    write_new(HERE/f'UNIT_TEST_REGISTRATION_{i:02}.json',dict(status='REGISTERED_BEFORE_TESTS',utc=now(),
        source_sha256=sources(),wall_s_max=300,test_methods_max=64,synthetic_scalar_validations_max=64,
        real_complete_requests_max=0,real_model_inferences_max=0,native_max=0,H_max=0,
        optimizer_updates_max=0,new_teacher_queries_max=0,S3_target_generation_max=0,
        scope='Hand-authored geometry/scalar and full synthetic statistics grid; checkpoint files are hashed only. Temp fixtures are not scientific models.'))


def register_probes():
    verify_training_anchors();ModelCatalog()
    registry=TrainingRegistry(ROOT/'rc3_global_local_execution_v3/REGISTRY.json')
    data=ROOT/f'rc3_global_local_execution_v3/runs/{EXPERIMENT}/data'
    cases=[]
    for spec in list(registry.development_specs())[:6]:
        folder=data/'task_draws'/f"{spec['map_seed']}_{spec['task_seed']}_A{spec['A_condition']}"
        rec=read(folder/'receipt.json');nav=data/'navigation'/(rec['map_key']+'.npz')
        with np.load(folder/'arrays.npz',allow_pickle=False) as a:
            Fh,Ah=array_hash(a['F']),array_hash(a['A'])
        with np.load(nav,allow_pickle=False) as n:Dh=array_hash(n['costs'])
        files={k:dict(path=label(p),sha256=digest(p)) for k,p in
            dict(receipt=folder/'receipt.json',scene=folder/'scene.json',arrays=folder/'arrays.npz',
                 navigation=nav,navigation_receipt=nav.with_suffix('.json')).items()}
        cases.append(dict(id=f"DEV_{spec['case']:03}",group='engineering_development_only',size=spec['size'],
            endpoint=spec['endpoints'],physical_key=rec['map_key'],spec=spec,files=files,
            original_parent_identity=rec['parent_identity'],F_sha256=Fh,A_sha256=Ah,D_sha256=Dh))
    code=dict(schema='rc3-s3-code-binding-v1',protocol_sha256=PROTOCOL,registry_sha256=REGISTRY,
        training_source_sha256=TRAIN_SOURCE,H_identity=H_ID,environment=environment(),
        source_sha256=sources(),model_index_sha256=digest(HERE/'S3_MODEL_INDEX.json'),
        purpose='Frozen S3 evaluator; not a training source revision or S3 execution authorization')
    write_new(HERE/'S3_CODE_BINDING.json',code)
    requests=grid(cases,SEEDS[0],True)
    write_new(HERE/'ENGINEERING_REQUEST_GRID.json',requests)
    write_new(HERE/'ENGINEERING_REGISTRATION.json',dict(scope='S3_ENGINEERING_ONLY',status='REGISTERED_BEFORE_FIRST_REQUEST',
        utc=now(),experiment_id=EXPERIMENT,engineering_id=ENGINEERING_ID,model_seed=SEEDS[0],
        code_binding_sha256=digest(HERE/'S3_CODE_BINDING.json'),complete_requests_max=126,wall_s_max=1800,
        formal_S3_authorized=False,target_access='SAVED_DEV_ONLY',optimizer_updates_max=0,new_teacher_queries_max=0,
        new_task_generation_max=0,request_grid_sha256=identity(requests),cases=cases,
        expected_calls=dict(native=36,greedy=90,common_H_invocations=72,cold_navigation=84,hot_navigation_prime=2,hot_hits=42),
        planned_closed_boundary_pause_after_requests=21,
        authority='Current user task authorizes bounded engineering acceptance only; formal S3 targets and original60 remain sealed.',
        timing_scope='Complete requests include parsing, source checks, navigation, F/A, extraction, model load, all encodings, construction, H, export, validation and artifact IO. Hot primes navigation only; prime and supervision/load/restore IO charged to outer 1800s.'))
    write_new(HERE/'PRE_EXECUTION_REVIEW.json',dict(judgment='GO_ENGINEERING_ONLY',utc=now(),
        protocol_changes=False,model_reward_H_changes=False,formal_S3=False,source_binding_sha256=digest(HERE/'S3_CODE_BINDING.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('static','probes'));a=p.parse_args()
    (static_registration if a.action=='static' else register_probes)()
