"""Zero-science authorization regression. UNIT_ONLY fixtures grant no authority.

Run only after the root agent freezes an engineering execution binding.
Byte fixtures exercise lineage prechecks, never tensor restoration or models.
"""
import argparse
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

from . import authority as auth
from . import runner as adapter
from rc3_global_local_trainer.data import TrainingRegistry
from rc3_global_local_trainer.runner import TrainingRun, StageClock

ROOT, HERE = auth.ROOT, Path(__file__).resolve().parent
OUT = None
JOURNAL = None
ASSERTIONS = 0
FORBIDDEN = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8', newline='\n')


def event(kind, **row):
    with JOURNAL.open('a', encoding='utf8', newline='\n') as stream:
        stream.write(json.dumps(dict(event=kind, **row), ensure_ascii=False, allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())


def check(value, message='Assertion failed'):
    global ASSERTIONS
    ASSERTIONS += 1
    if not value: raise AssertionError(message)


def rejects(fn, name, types=(auth.AdmissionError, RuntimeError, ValueError, FileExistsError)):
    try: fn()
    except types as exc:
        event('expected_rejection', case=name, type=type(exc).__name__, message=str(exc))
        return True
    raise AssertionError('Invalid request was accepted: '+name)


def fixture_binding(name):
    value=auth.science_binding(sha(auth.PROTOCOL),sha(auth.REGISTRY),{'UNIT_ONLY':'not-production-source'},
        'UNIT_ONLY_H',name,'UNIT_ONLY_PROTOCOL_APPROVAL','UNIT_ONLY_MANIFEST')
    value['engineering_fixture']=True
    return value


def artifact(folder, name='UNIT_ONLY.artifact'):
    path=Path(folder)/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(b'UNIT_ONLY_NON_EXECUTABLE_LINEAGE_BYTES\n')
    return path


def upstream(chain, stage, seed):
    return {k:chain.completed(k)[1] for k in chain.required_predecessors(stage,seed) if k.split('/')[0]!=stage}


def complete(chain, stage, seed, auth_id, resources=None):
    entry=chain.admit(stage,seed,auth_id,upstream(chain,stage,seed))
    f=artifact(chain.folder,stage.replace('-','_')+'_'+str(seed)+'.unit')
    row=chain.close(entry,[f],resources or {'UNIT_ONLY_simulated_used':1},complete=True)
    return row


def protocol_case():
    p=auth.read(auth.PROTOCOL);r=auth.read(auth.REGISTRY)
    check(p['status']=='PENDING_USER_APPROVAL_NOT_ACTIVE_NOT_AUTHORIZED')
    check(p['approval_state']['user_protocol_approved'] is False)
    check(p['seeds_sha256']==sha(auth.REGISTRY))
    check(len(r['S1_teachers'])==256 and len(r['S2_train'])==96 and len(r['S3'])==108)
    check(all(len(x['task_seeds'])==8 for x in r['S2_train']))
    check(tuple(r['policy_seeds'])==auth.SEEDS)
    pointer=auth.read(ROOT/'rc3_global_local/FUTURE_DESIGN_STATUS.json')
    check(pointer['active_protocol']=='S2_S3_protocol_v2.json')
    check(sha(auth.REGISTRY)==sha(ROOT/'rc3_global_local_impl_review/PROPOSED_SEEDS_V3.json'))
    return dict(candidate_pending=True,active_design='v2',teachers=256,bank=768,S3_declarations=108,target_tasks_generated=0)


def gates_case():
    before=copy.deepcopy(FORBIDDEN)
    for stage,seed in [('S1-A',None),('S1-B',auth.SEEDS[0]),('S2',auth.SEEDS[0]),('S3',None),('S1-B',999),('S1-B',True),('S1-A',auth.SEEDS[0])]:
        rejects(lambda:auth.require_stage_authority(auth.PROTOCOL,auth.REGISTRY,None,None,stage,seed,'UNIT_ONLY'),stage+str(seed))
    rejects(lambda:auth.require_stage_authority(ROOT/'rc3_global_local/S2_S3_protocol_v2.json',auth.REGISTRY,None,None,'S1-A',None,'UNIT_ONLY'),'old protocol')
    rejects(lambda:auth.output_directory(HERE/'runs'/'nested'/'UNIT_ONLY','UNIT_ONLY'),'nested duplicate run')
    rejects(lambda:auth.output_directory(ROOT/'rc3_global_local','UNIT_ONLY'),'old evidence output')
    canonical=HERE/'runs'/'UNIT_ONLY_NOT_CREATED'
    check(auth.output_directory(canonical,'UNIT_ONLY_NOT_CREATED')==canonical.resolve())
    for stage in ['S1-A','S1-B','S2']:
        args=['--stage',stage,'--experiment-id','UNIT_ONLY_NOT_CREATED','--output',str(canonical)]
        if stage!='S1-A':args+=['--seed',str(auth.SEEDS[0])]
        check(adapter.main(args)==2)
    check(not canonical.exists());check(before==FORBIDDEN)
    rejects(lambda:adapter.StageRun(SimpleNamespace(),OUT/'not_a_formal_run'),'permit type')
    return dict(production_missing_approval_rejected=True,canonical_single_output=True,science_calls=0)


def chain_case():
    binding=fixture_binding('UNIT_CHAIN');chain=auth.StageChain(OUT/'chain',binding)
    a=complete(chain,'S1-A',None,'UNIT_AUTH_A',{'UNIT_ONLY_calls':512,'UNIT_ONLY_elapsed_s':150.})
    a_hash=chain.completed('S1-A')[1]
    rejects(lambda:chain.admit('S2',auth.SEEDS[0],'UNIT_AUTH_S2',{}),'S2 before all S1')
    rejects(lambda:chain.admit('S1-B',auth.SEEDS[1],'UNIT_AUTH_B',{'S1-A':a_hash}),'fixed seed order')
    for seed in auth.SEEDS:complete(chain,'S1-B',seed,'UNIT_AUTH_B',{'UNIT_ONLY_optimizer_cursor':256})
    parent=upstream(chain,'S2',auth.SEEDS[0]);check(len(parent)==4)
    rejects(lambda:chain.admit('S2',auth.SEEDS[0],'UNIT_AUTH_S2',dict(parent,**{'S1-A':'wrong'})),'wrong parent hash')
    for seed in auth.SEEDS:complete(chain,'S2',seed,'UNIT_AUTH_S2',{'UNIT_ONLY_optimizer_cursor':1024})
    check(chain.completed('S1-A')[0]==a)
    check(chain.completed('S1-A')[1]==a_hash)
    ids=[auth.read(chain.path(k,'AUTHORIZATION.json'))['authorization_sha256'] for k in ['S1-A','S1-B/'+str(auth.SEEDS[0]),'S2/'+str(auth.SEEDS[0])]]
    check(len(set(ids))==3)
    check(all(auth.read(chain.path(k,'COMPLETE.json'))['science_binding']==binding for k in ['S1-A','S1-B/'+str(auth.SEEDS[0]),'S2/'+str(auth.SEEDS[0])]))
    rejects(lambda:chain.admit('S1-A',None,'NEW_AUTH_RESET',{}),'completed stage rerun')
    return dict(new_downstream_authorization=True,same_science_identity=True,all_three_S1_endpoints_required=True,fixed_seed_order=True,no_real_execution=True)


def resume_case():
    chain=auth.StageChain(OUT/'resume',fixture_binding('UNIT_RESUME'));complete(chain,'S1-A',None,'UNIT_A')
    parent=upstream(chain,'S1-B',auth.SEEDS[0]);ad=chain.admit('S1-B',auth.SEEDS[0],'UNIT_B',parent)
    start=chain.path(ad['key'],'START.json');before=start.read_bytes()
    restored=chain.admit('S1-B',auth.SEEDS[0],'UNIT_B',parent,resume=True)
    check(restored==ad and start.read_bytes()==before)
    rejects(lambda:chain.admit('S1-B',auth.SEEDS[0],'DIFFERENT_AUTH',parent,resume=True),'same stage changed authorization')
    rejects(lambda:chain.admit('S1-B',auth.SEEDS[0],'UNIT_B',parent),'unannounced restart')
    spent={'UNIT_ONLY_queries':7,'UNIT_ONLY_elapsed_s':31.25}
    chain.close(ad,[],spent,complete=False,error={'type':'UNIT_ONLY_SIMULATED_FAILURE'})
    check(auth.read(chain.path(ad['key'],'FAILED.json'))['cumulative_stage_resources']==spent)
    rejects(lambda:chain.admit('S1-B',auth.SEEDS[0],'UNIT_B',parent,resume=True),'failed restart')
    # Resume arithmetic adapter delegates to the separately verified frozen
    # loader. This unit patches only that parent to isolate the new overhead.
    clock=StageClock(100);run=object.__new__(adapter.StageTrainingRun);run.admission_overhead_s=2.5
    def frozen_resume(*args):clock.before_s=31.25
    with patch.object(TrainingRun,'_resume',frozen_resume):
        run._resume(None,None,None,clock,None)
    check(clock.before_s==33.75)
    return dict(same_authorization_resume=True,failed_or_new_authorization_reset_rejected=True,prior_stage_seconds=31.25,new_admission_seconds=2.5,resumed_before_seconds=33.75,clock_test='isolated adapter arithmetic, not new model resume')


def parent_completion_mutation_case():
    """Child completion must bind exact current parent receipts, recursively."""
    chain=auth.StageChain(OUT/'parent_completion_mutation',fixture_binding('UNIT_PARENT_MUTATION'))
    complete(chain,'S1-A',None,'UNIT_A')
    for seed in auth.SEEDS:complete(chain,'S1-B',seed,'UNIT_B')
    parent_map=upstream(chain,'S2',auth.SEEDS[0])
    mutations=[]
    for parent,child in [('S1-A',auth.stage_key('S1-B',auth.SEEDS[0])),
                         (auth.stage_key('S1-B',auth.SEEDS[0]),auth.stage_key('S1-B',auth.SEEDS[1]))]:
        f=chain.path(parent,'COMPLETE.json');original=f.read_bytes();row=auth.read(f)
        row['cumulative_stage_resources']['UNIT_ONLY_tampered_resource']=123
        save(f,row)
        name='parent_resource_'+str(len(mutations))
        save(OUT/'parent_completion_mutation_evidence'/(name+'.json'),row)
        rejects(lambda:chain.completed(child),name+' child stale receipt')
        rejects(lambda:chain.admit('S2',auth.SEEDS[0],'UNIT_S2',parent_map),name+' S2 stale ancestry')
        f.write_bytes(original)
        check(chain.completed(child)[0]['stage_key']==child)
        mutations.append(dict(parent=parent,child=child,type='resource_only_parent_change'))
    child=auth.stage_key('S1-B',auth.SEEDS[1]);f=chain.path(child,'COMPLETE.json');original=f.read_bytes()
    for mode in ['missing','extra']:
        row=json.loads(original)
        if mode=='missing':row['predecessors'].pop(next(iter(row['predecessors'])))
        else:row['predecessors']['S1-B/99999999']='UNIT_ONLY_NONDECLARED_PARENT'
        save(f,row)
        save(OUT/'parent_completion_mutation_evidence'/('parent_keys_'+mode+'.json'),row)
        rejects(lambda:chain.completed(child),mode+' exact parent key set')
        rejects(lambda:chain.admit('S2',auth.SEEDS[0],'UNIT_S2',parent_map),mode+' S2 parent set')
        f.write_bytes(original)
        check(chain.completed(child)[0]['stage_key']==child)
        mutations.append(dict(child=child,type=mode+'_parent_key'))
    check(upstream(chain,'S2',auth.SEEDS[0])==parent_map)
    admission=chain.admit('S2',auth.SEEDS[0],'UNIT_S2',parent_map)
    check(admission['predecessors']==parent_map)
    chain.close(admission,[artifact(chain.folder,'S2_RESTORED_ACCEPTED.unit')],{'UNIT_ONLY_simulated_used':1},complete=True)
    return dict(resource_only_parent_mutations_rejected=2,missing_and_extra_parent_sets_rejected=True,
        S2_rejects_stale_ancestry=True,original_byte_restoration_accepts=True,
        negative_mutation_fixtures=mutations,real_stage_execution=False)


def make_pool(folder,binding):
    pool=folder/'teachers256';pool.mkdir(parents=True)
    rows=[];specs=list(TrainingRegistry(auth.REGISTRY).teacher_specs())
    for i,spec in enumerate(specs):
        identity='UNIT_ONLY_VIEW_'+str(i)
        plan=dict(status='complete',problem_identity=identity,validation={'valid_model_plan':True},target_task_ids_global=[i],objective_length_plus_stops_m=2.)
        hp=dict(plan,objective_length_plus_stops_m=1.)
        label=dict(fixture_scope='UNIT_ONLY_METADATA_NOT_A_TEACHER',index=i,registry_sha256=binding['registry_sha256'],spec=spec,view_identity=identity,actions=[1],teacher_J=1.,native_plan=plan,H_result=dict(J=1.,raw_J=2.,route=[0,1,0],plan=hp))
        f=pool/f'label_{i:03}.json';save(f,label);rows.append(dict(index=i,file=f.name,sha256=sha(f)))
    ledger=dict(native_teacher_attempts=256,native_teacher_completed=256,terminal={'teacher_H':dict(terminal_requests=256,terminal_attempts=256,terminal_completed=256,terminal_failed=0)})
    save(pool/'POOL_INDEX.json',dict(schema='rc3-shared-teacher-pool-v1',fixture_scope='UNIT_ONLY',count=256,registry_sha256=binding['registry_sha256'],source_binding=binding,labels=rows,ledger=ledger))
    return pool


def pool_case():
    folder=OUT/'pool';binding=fixture_binding('UNIT_POOL');pool=make_pool(folder,binding)
    check(len(adapter.teacher_artifacts(folder,binding))==257)
    f=pool/'label_000.json';original=auth.read(f);index=auth.read(pool/'POOL_INDEX.json')
    def mutation(transform,name):
        row=copy.deepcopy(original);transform(row);save(f,row)
        changed=copy.deepcopy(index);changed['labels'][0]['sha256']=sha(f);save(pool/'POOL_INDEX.json',changed)
        rejects(lambda:adapter.teacher_artifacts(folder,binding),name)
        save(f,original);save(pool/'POOL_INDEX.json',index)
    mutation(lambda r:r['native_plan'].pop('validation'),'missing native certificate')
    mutation(lambda r:r['H_result']['plan']['validation'].update(valid_model_plan=False),'invalid H certificate')
    mutation(lambda r:r.update(actions=[2]),'route label mismatch')
    mutation(lambda r:r['spec'].update(A_condition=99),'teacher spec mismatch')
    mutation(lambda r:r['H_result'].update(J=100),'H cost mismatch')
    f.write_bytes(f.read_bytes()+b' ');rejects(lambda:adapter.teacher_artifacts(folder,binding),'changed pool bytes');save(f,original)
    changed=copy.deepcopy(index);changed['ledger']['native_teacher_attempts']=257;save(pool/'POOL_INDEX.json',changed)
    rejects(lambda:adapter.teacher_artifacts(folder,binding),'teacher budget excess');save(pool/'POOL_INDEX.json',index)
    check(len(adapter.teacher_artifacts(folder,binding))==257)
    return dict(complete_structural_records=256,wrong_spec_route_H_validation_hash_budget_rejected=True,teacher_queries=0,scope='Metadata only; no fixture is a real teacher or formal input')


def make_checkpoint(folder,seed,phase,cursor,binding):
    base=folder/f'seed_{seed}'/phase;base.mkdir(parents=True,exist_ok=True)
    f=base/('untrained.pt' if cursor==0 else f'update_{cursor:04}.pt');f.write_bytes(b'UNIT_ONLY_NOT_A_TORCH_CHECKPOINT\n')
    side=dict(schema='rc3-trainer-checkpoint-integrity-v1',fixture_scope='UNIT_ONLY_METADATA',phase=phase,
        binding=dict(binding,optimizer_seed=seed,run=folder.relative_to(ROOT).as_posix(),phase=phase),cursors={phase:cursor},weight_version=cursor,sha256=sha(f),bytes=f.stat().st_size,model_tensor_sha256='UNIT_ONLY_NOT_A_MODEL')
    save(f.with_suffix('.pt.json'),side)
    save(f.with_suffix('.pt.commit.json'),dict(schema='rc3-checkpoint-boundary-commit-v1',phase=phase,checkpoint_sha256=sha(f),sidecar_sha256=sha(f.with_suffix('.pt.json')),cursors={phase:cursor}))
    return f


def checkpoint_case():
    folder=OUT/'checkpoint';binding=fixture_binding('UNIT_CHECKPOINT');seed=auth.SEEDS[0]
    f=make_checkpoint(folder,seed,'S1',256,binding);check(len(adapter.checkpoint_artifacts(folder,seed,'S1',256,binding))==3)
    original=auth.read(f.with_suffix('.pt.json'));commit=auth.read(f.with_suffix('.pt.commit.json'))
    for field in ['protocol_sha256','registry_sha256','H_sha256','source_sha256','optimizer_seed','run']:
        row=copy.deepcopy(original);row['binding'][field]='WRONG';save(f.with_suffix('.pt.json'),row)
        c=dict(commit,sidecar_sha256=sha(f.with_suffix('.pt.json')));save(f.with_suffix('.pt.commit.json'),c)
        rejects(lambda:adapter.checkpoint_artifacts(folder,seed,'S1',256,binding),'wrong checkpoint '+field)
    save(f.with_suffix('.pt.json'),original);save(f.with_suffix('.pt.commit.json'),commit)
    wrong=copy.deepcopy(original);wrong['weight_version']=1;save(f.with_suffix('.pt.json'),wrong)
    rejects(lambda:adapter.checkpoint_artifacts(folder,seed,'S1',256,binding),'selected earlier weight')
    save(f.with_suffix('.pt.json'),original);f.write_bytes(b'UNIT_ONLY_CORRUPT')
    rejects(lambda:adapter.checkpoint_artifacts(folder,seed,'S1',256,binding),'bad checkpoint bytes')
    return dict(protocol_registry_H_source_seed_run_weight_bytes_rejected=True,metadata_only_not_restore=True,production_engineering_checkpoint_use='Excluded by exact run/binding and frozen tensor loader; byte fixture never passes tensor load')


def dirty_case():
    # The old accepted checkpoint is read only as a NEGATIVE fixture. No model
    # is constructed, optimizer loaded or tensor used as a formal source.
    origin=ROOT/'rc3_global_local_trainer/verification_runs/acceptance_01/orchestration/S1/update_0001.pt'
    folder=OUT/'dirty_boundary';folder.mkdir(parents=True)
    f=folder/origin.name
    for suffix in ['','.json','.commit.json']:shutil.copyfile(Path(str(origin)+suffix),Path(str(f)+suffix))
    (folder/'updates.jsonl').write_text('{"UNIT_ONLY_dirty_after_checkpoint":true}\n',encoding='utf8')
    run=object.__new__(TrainingRun);trainer=SimpleNamespace(phase='S1')
    rejects(lambda:run._resume(f,trainer,None,None,folder),'dirty after closed boundary',(RuntimeError,))
    return dict(frozen_production_dirty_log_precheck=True,model_instantiations=0,torch_dictionary_deserializations=1,prior_engineering_checkpoint_not_formal_source=True)


def adapter_record_case():
    # Unit-only stub invokes the real execute/record path without a production
    # permit. Constructor/canonical-path admission is tested separately above.
    folder=OUT/'adapter_construct_failure';binding=fixture_binding('UNIT_FACTORY')
    run=object.__new__(adapter.StageRun);run.folder=folder;run.chain=auth.StageChain(folder,binding)
    run.permit=SimpleNamespace(stage='S1-A',seed=None,execution=SimpleNamespace(),authorization_sha256='UNIT_ONLY_AUTH',predecessor_hashes={},authorization={'status':'UNIT_ONLY'},binding=binding)
    run._validate_upstream=lambda:None
    def failure(*_):raise RuntimeError('UNIT_ONLY expected constructor failure')
    run.run_factory=failure
    rejects(lambda:run.execute(),'factory construction failure',(RuntimeError,))
    failed=auth.read(run.chain.path('S1-A','FAILED.json'));check(failed['status']=='FAILED_INCOMPLETE')
    check(failed['cumulative_stage_resources']['science_calls']==0)
    rejects(lambda:run.execute(),'failed stage cannot start another factory')
    # Hash-before-resource-callback order and cap enforcement.
    chain=auth.StageChain(OUT/'hash_clock',fixture_binding('UNIT_HASH_CLOCK'))
    ad=chain.admit('S1-A',None,'UNIT_A',{});f=artifact(chain.folder)
    observed=[];real_digest=auth.digest
    def spy(path):
        if Path(path)==f:observed.append('hash')
        return real_digest(path)
    def resources():
        observed.append('resources');check(observed[0]=='hash');return {'UNIT_ONLY_elapsed_s':4.}
    with patch.object(auth,'digest',spy):chain.close(ad,[f],resources,complete=True)
    check(observed==['hash','resources'])
    cap_chain=auth.StageChain(OUT/'cap_clock',fixture_binding('UNIT_CAP_CLOCK'));ad=cap_chain.admit('S1-A',None,'UNIT_A',{})
    clock=StageClock(1);clock.before_s=2
    rejects(lambda:cap_chain.close(ad,[artifact(cap_chain.folder)],lambda:clock.check(),complete=True),'hash-end cap check',(TimeoutError,))
    check(not cap_chain.path('S1-A','COMPLETE.json').exists())
    return dict(factory_failure_recorded=True,failed_relaunch_rejected=True,artifact_hash_before_resource_snapshot=True,exhausted_clock_cannot_complete=True)


CASES=dict(protocol_metadata=protocol_case,production_gates=gates_case,separate_stage_chain=chain_case,
    same_stage_resume=resume_case,teacher_pool_metadata=pool_case,checkpoint_identity=checkpoint_case,
    dirty_boundary=dirty_case,adapter_records=adapter_record_case,parent_completion_mutation=parent_completion_mutation_case)


@contextlib.contextmanager
def forbid_science():
    import torch
    import rc3_global_local.policy as policy
    import rc3_global_local.terminal as terminal
    import rc3_global_local_trainer.runner as oldrunner
    import rc3_fast.scenes as scenes
    import rc3_constructive.problem as problem
    import rc3_baseline.baseline as baseline
    def banned(name):
        def reject(*_,**__):
            FORBIDDEN[name]=FORBIDDEN.get(name,0)+1
            event('forbidden_operation_attempt',operation=name)
            raise AssertionError('Registered zero-operation cap reached: '+name)
        return reject
    with contextlib.ExitStack() as stack:
        for obj,name,counter in [(policy.GlobalLocalPolicy,'__init__','model_instantiations'),
            (policy.GlobalLocalPolicy,'encode','model_forward'),(policy.GlobalLocalPolicy,'score','model_forward'),
            (torch.optim.Adam,'step','optimizer_updates'),(torch.autograd,'backward','backward'),
            (oldrunner,'run_endpoint','real_teacher_queries'),(oldrunner,'finish','real_full_H'),
            (terminal,'finish','real_full_H'),(scenes,'make_complex_scene','scene_task_generation'),
            (problem,'resample_tasks','scene_task_generation'),(baseline,'navigation','navigation_generation')]:
            stack.enter_context(patch.object(obj,name,banned(counter)))
        yield


def run(binding_path):
    global OUT,JOURNAL
    binding_path=Path(binding_path).resolve();binding=auth.read(binding_path)
    if not binding_path.is_relative_to(HERE):raise ValueError('Binding must be in new study')
    for name,value in binding['source_sha256'].items():check(sha(ROOT/name)==value,'Source changed: '+name)
    if 'protocol_sha256' in binding:check(binding['protocol_sha256']==sha(auth.PROTOCOL))
    if 'registry_sha256' in binding:check(binding['registry_sha256']==sha(auth.REGISTRY))
    check(binding['engineering_protocol_sha256']==sha(HERE/'ENGINEERING_REGRESSION_PROTOCOL.json'),'Engineering registration changed')
    protocol=auth.read(HERE/'ENGINEERING_REGRESSION_PROTOCOL.json')
    JOURNAL=HERE/'ENGINEERING_REGRESSION_JOURNAL.jsonl'
    history=[json.loads(x) for x in JOURNAL.read_text(encoding='utf8').splitlines()] if JOURNAL.exists() else []
    starts=[e for e in history if e['event']=='run_start'];ends=[e for e in history if e['event']=='run_end']
    if len(starts)!=len(ends):raise RuntimeError('Prior incomplete attempt accounting requires audit')
    check(len(starts)<protocol['limits']['regression_process_attempts'])
    used=sum(e['wall_s'] for e in ends);check(used<protocol['limits']['wall_s_all_attempts'])
    run_id=binding.get('run_id','acceptance_'+str(len(starts)+1).zfill(2))
    OUT=HERE/'engineering_runs'/run_id;OUT.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();event('run_start',run_id=run_id,binding_sha256=sha(binding_path))
    result=dict(schema='rc3-zero-science-stage-regression-v1',status='RUNNING',run_id=run_id,
        source_sha256=binding['source_sha256'],binding_sha256=sha(binding_path),engineering_fixture_only=True,cases={},error=None)
    try:
        with forbid_science():
            for name,fn in CASES.items():
                if used+time.perf_counter()-started>=protocol['limits']['wall_s_all_attempts']:raise TimeoutError('Cumulative acceptance wall reached')
                event('case_start',run_id=run_id,case=name);tick=time.perf_counter()
                result['cases'][name]=dict(status='PASS',evidence=fn(),wall_s=time.perf_counter()-tick)
                event('case_complete',run_id=run_id,case=name)
        check(not FORBIDDEN);result['status']='PASS'
    except BaseException as exc:
        result['status']='FAIL';result['error']=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    finally:
        result.update(assertions=ASSERTIONS,forbidden_operation_attempts=FORBIDDEN,
            scientific_call_counters={k:0 for k,v in protocol['limits'].items() if v==0},
            wall_s=time.perf_counter()-started,
            scope='UNIT_ONLY structure/byte/ledger fixtures, one negative trusted torch dictionary read; no production permission, real teachers/models, optimizer or task generation.')
        save(OUT/'RESULT.json',result)
        event('run_end',run_id=run_id,status=result['status'],wall_s=time.perf_counter()-started,scientific_call_counters=result['scientific_call_counters'])
    print(json.dumps(dict(status=result['status'],run_id=run_id,cases=len(result['cases']),assertions=ASSERTIONS,wall_s=result['wall_s'],error=result['error']),ensure_ascii=True))
    if result['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--binding',required=True)
    run(parser.parse_args().binding)
