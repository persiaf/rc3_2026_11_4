"""New task draws on fixed DEV geometry; old training and teachers read-only."""
from types import SimpleNamespace
import time
import numpy as np
from .common import *
from rc3_fast.planner import map_key
from rc3_global_local.problem import array_hash
from rc3_global_local_impl_review.runtime_guards import snapshot_parent
from rc3_s3_connectivity_20261109.scope import scoped_view


def saved_provider(cases):
    from rc3_s3_evaluator.inputs import SavedDevelopmentInputs
    return SavedDevelopmentInputs(SimpleNamespace(scope='engineering',registration=dict(cases=cases)),HOME/'runs')


def training_provider(clock,ledger,log):
    from rc3_global_local_trainer.approval import ExecutionPermit
    from rc3_global_local_trainer.data import TrainingRegistry,SceneProvider
    b=read(HOME/'READ_ONLY_DATA_SOURCE_BINDING.json')['binding'];reg=ROOT/'rc3_global_local_execution_v3/REGISTRY.json'
    permit=ExecutionPermit(HOME/'PROTOCOL.json',reg,digest(HOME/'PROTOCOL.json'),REGISTRY,b,
        ('read_only_data_provenance',),digest(HOME/'AUTHORIZATION.json'),digest(HOME/'AUTHORIZATION.json'))
    provider=SceneProvider(TrainingRegistry(reg),OLD_RUN/'data',ledger,permit,record=log,read_only=True)
    provider.check=clock.check
    return provider


class SavedNavigation:
    def __init__(self,parents,log,clock):
        self.parents={p['physical_key']:p for p in parents};self.cache={};self.log=log;self.clock=clock

    def get(self,scene,kind,meter):
        from rc3_global_local_trainer.navigation_compat import load_navigation_variants,select_navigation
        self.clock.check();key=map_key(scene);source=self.parents[key];path=ROOT/source['navigation_file']
        require(digest(path)==source['navigation_sha256'] and digest(path.with_suffix('.json'))==source['navigation_receipt_sha256'],'Frozen navigation bytes changed')
        if key not in self.cache:
            with meter.span('saved_navigation_load'):
                self.cache[key]=select_navigation(load_navigation_variants(scene,path),scene)
                meter.add('saved_navigation_artifact_loads')
        else:meter.add('saved_navigation_only_hits')
        return self.cache[key]


def prepare_bank(clock,log):
    """No policy or route evaluation; all72 cases fixed before any optimizer."""
    from rc3_constructive.problem import resample_tasks
    from rc3_fast.solver import vectorized_feasibility
    from rc3_global_local.run_s0 import make_allow
    declared=read(HOME/'INPUT_REGISTRY.json');parents=declared['new_development_parents']
    pool=SavedNavigation(parents,log,clock);records=[];draws={}
    for case in declared['new_development']:
        clock.check();spec=case['spec'];key=(spec['parent'],spec['draw']);meter=Meter(log,clock.check)
        if key not in draws:
            src=parents[spec['parent']];path=ROOT/src['scene_file']
            require(digest(path)==src['scene_sha256'],'Geometry source changed')
            with meter.span('new_task_generation_and_FA'):
                meter.add('new_task_draws');scene=resample_tasks(read(path),spec['task_seed'])
                require(map_key(scene)==case['physical_key'],'New tasks changed physical geometry')
                F,_,_=vectorized_feasibility(scene);A=make_allow(F,spec['A_condition']);meter.add('proxy_FA_generations')
            nav=pool.get(scene,'saved_navigation',meter);parent=snapshot_parent(scene,F,A,navigation=nav)
            folder=HOME/'runs/SCIENCE/new_inputs'/f'parent{key[0]}_draw{key[1]}'
            write_new(folder/'scene.json',scene)
            with (folder/'arrays.npz').open('xb') as f:np.savez_compressed(f,F=F,A=A)
            receipt=dict(task_seed=spec['task_seed'],A_condition=spec['A_condition'],physical_key=case['physical_key'],
                scene_sha256=digest(folder/'scene.json'),arrays_sha256=digest(folder/'arrays.npz'),F_sha256=array_hash(F),A_sha256=array_hash(A),
                parent_identity=parent.identity,navigation_content_sha256=nav.content_sha256,navigation_source=src,
                protocol_sha256=digest(HOME/'PROTOCOL.json'),source_binding_sha256=digest(HOME/'SOURCE_BINDING.json'))
            write_new(folder/'RECEIPT.json',receipt);draws[key]=(parent,folder,receipt)
        parent,folder,receipt=draws[key]
        view,evidence=scoped_view(parent,spec)
        require(len(view.task_ids)>0,'Empty prescribed DEV target; no replacement')
        records.append(dict(case,view_identity=view.identity,scope_evidence=evidence,
            files={n:dict(path=label(folder/f),sha256=digest(folder/f)) for n,f in [('scene','scene.json'),('arrays','arrays.npz'),('receipt','RECEIPT.json')]}))
        log(dict(event='new_DEV_input_frozen',case_id=case['id'],view_identity=view.identity,task_ids=view.task_ids.tolist(),
            candidate_ids=view.candidate_ids.tolist(),scope_evidence=evidence,counts=meter.counts,seconds=meter.seconds))
    require(len(records)==72 and len(draws)==12,'Wrong frozen bank')
    write_new(HOME/'runs/SCIENCE/NEW_TASK_BANK_FROZEN.json',dict(cases=records,physical_layouts=12,task_draws=12,
        frozen_before_optimizer=True,no_policy_evaluation=True,old_S3_and60_task_access=0,utc=now()))
    return records


class NewDevelopmentInputs:
    def __init__(self,cases):self.by_id={c['id']:c for c in cases}

    def load(self,case,meter):
        require(self.by_id.get(case['id'])==case,'Unregistered DEV case')
        with meter.span('new_DEV_input_load_and_verify'):
            paths={k:ROOT/v['path'] for k,v in case['files'].items()}
            require(all(digest(paths[k])==v['sha256'] for k,v in case['files'].items()),'New DEV input modified')
            scene=read(paths['scene']);receipt=read(paths['receipt'])
            with np.load(paths['arrays'],allow_pickle=False) as z:F,A=z['F'].copy(),z['A'].copy()
            require(array_hash(F)==receipt['F_sha256'] and array_hash(A)==receipt['A_sha256'],'F/A receipt mismatch')
            require(scene['task_seed']==case['spec']['task_seed'] and map_key(scene)==case['physical_key'],'Task/geometry mismatch')
            meter.add('fresh_task_input_reads')
        return scene,F,A,case['spec'],dict(input_kind='GL_v5_new_task_on_training_geometry',
            input_source_sha256={k:v['sha256'] for k,v in case['files'].items()},expected_view_identity=case['view_identity'])
