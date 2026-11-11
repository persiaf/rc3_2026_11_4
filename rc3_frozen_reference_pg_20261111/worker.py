import time
START_M,START_W=time.monotonic(),time.time()
import argparse
import copy
import io
import json
import platform
import random
import subprocess
import traceback
from .common import HOME,ROOT,ROLES,METHODS,TABLE,read,put,sha,Ledger

def bootstrap(stage):
    ledger=Ledger(stage,START_M,START_W)
    globals()['CURRENT_LEDGER']=ledger
    from rc3_r04_m1_real_20261111.provenance_use import ScientificTracedProvenance,verify_real_environment
    binding=read(HOME/'SOURCE_BINDING.json')
    trace=ScientificTracedProvenance(ROOT,ledger.path,binding,ledger.check,'frozen-reference-PG-'+stage+'-01');trace.install_import_guard()
    assert read(HOME/(stage.upper()+'_AUTHORIZATION.json'))['source_binding_sha256']==sha(HOME/'SOURCE_BINDING.json')
    assert read(HOME/'IMPLEMENTATION_REVIEW.json')['status']=='GO'
    assert sha(HOME/'PROTOCOL.json')==binding['protocol_sha256'] and sha(HOME/'SCHEDULE.json')==binding['schedule_sha256']
    with trace.stage('real_environment_and_readonly_input_interfaces'):
        import numpy as np
        import torch
        from rc3_conservative_refinement import saved_inputs as si,runtime as old
        from . import actor,reference,trainer,runtime,analysis,audit
        env=verify_real_environment(trace,torch,dict(actor=actor,trainer=trainer,runtime=runtime))
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        random.seed(11622026);np.random.seed(11622026);torch.manual_seed(11622026)
        actual=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device='cpu',threads=1,deterministic_algorithms=True)
        assert actual==binding['environment']
        si.digest=trace.hash_file;si.read=trace.read_json;old.file_sha256=trace.hash_file
        from rc3_global_local_trainer import navigation_compat,approval
        from rc3_global_local_impl_review import runtime_guards
        navigation_compat.read=approval.read=trace.read_json
        navigation_compat.digest=approval.digest=runtime_guards.file_hash=trace.hash_file
        raw_np_load=np.load
        def checked_np_load(p,*a,**kw):
            if isinstance(p,(str,__import__('os').PathLike)):p=io.BytesIO(trace.read_bytes(p))
            return raw_np_load(p,*a,**kw)
        np.load=checked_np_load
        raw_torch_load=torch.load
        def checked_torch_load(p,*a,**kw):
            if isinstance(p,(str,__import__('os').PathLike)) and not __import__('pathlib').Path(p).resolve().is_relative_to(HOME):p=io.BytesIO(trace.read_bytes(p))
            return raw_torch_load(p,*a,**kw)
        torch.load=checked_torch_load
        from rc3_conservative_refinement import model as historical_model
        def no_critic(*a,**kw):raise RuntimeError('Neural Critic construction forbidden')
        historical_model.RemainingCostCritic.__init__=no_critic
        inputs=si.SavedInputs();schedule=read(HOME/'SCHEDULE.json')
        assert inputs.evaluation_cases==schedule['evaluation']
        for r in schedule['training']:
            assert r['specs']==inputs.training(11622026,r['opportunity']) and r['teacher_indices']==inputs.teacher_indices(11622026,r['opportunity'])
        ref=reference.FrozenReference(read(HOME/'PROTOCOL.json')['frozen_reference']['sha256'])
        runtime.install_H_accounting(ledger);runtime.install_feature_timing()
        identity=dict(source_binding_sha256=sha(HOME/'SOURCE_BINDING.json'),protocol_sha256=sha(HOME/'PROTOCOL.json'),schedule_sha256=sha(HOME/'SCHEDULE.json'),table_sha256=ref.sha256)
        trace.check_loaded_modules();put(ledger.path/'ACTUAL_ENVIRONMENT.json',dict(actual=actual,source_classification=env,commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    return ledger,trace,inputs,ref,identity

def engineering(ledger,trace,inputs,ref,identity):
    import torch
    from .actor import load_actor,export,load_export
    from .trainer import Trainer,checkpoint_equal,rng
    from . import runtime,audit
    schedule=read(HOME/'SCHEDULE.json');specs=schedule['training'][0]['specs'];spec=specs[0]
    tests=[]
    def passed(name):tests.append(dict(check=name,status='PASS'));ledger.event('engineering_check_pass',check=name)
    x=torch.zeros(5,268);x[:,256]=torch.tensor([1.,.75,.5,.25,0.]);v=ref.values(x,'10x10')
    expected=-ref.table['global'][0]-.75*ref.table['global'][1]+ref.table['global'][2]
    assert torch.allclose(v['global_value'],torch.full((5,),expected),atol=1e-6,rtol=1e-6)
    assert not v['global_value'].requires_grad and not v['conditional_value'].requires_grad
    u=ref.values(x,'unknown');assert u['fallback']==[True]*5 and torch.equal(u['global_value'],u['conditional_value'])
    passed('M2_units_no_second_n_division_frozen_bins_fallback_detach')
    trainers={r:Trainer(load_actor(r),ref,ledger,identity) for r in ROLES}
    from rc3_conservative_refinement.runtime import tensor_digest
    assert len({tensor_digest(t.actor.correction.state_dict()) for t in trainers.values()})==1
    passed('three_exact_roles_same_initial_A1_tensors_and_frozen_SL256')
    p=ledger.path/'models';p.mkdir()
    t=trainers['PG-global'];zero=runtime.request(ledger,inputs,spec,'eng:zero','PG-global',actor=t.actor,greedy=True,equivalence=True)
    sl=runtime.request(ledger,inputs,spec,'eng:FrozenSL','FrozenSL',greedy=True)
    assert zero['record']['raw_route']==sl['raw_route'] and zero['record']['route']==sl['route'] and zero['record']['metrics']==sl['metrics']
    entry=export(p/'zero.actor.pt',t.actor,usage='engineering',opportunity=0)
    loaded=runtime.request(ledger,inputs,spec,'eng:zero_export','PG-global',entry=entry,greedy=True,equivalence=True)
    assert loaded['record']['route']==zero['record']['route']
    try:load_export(entry,role='PG-cond',usage='engineering')
    except ValueError:pass
    else:raise AssertionError('Role alias accepted')
    try:load_export(entry,role='PG-global',usage='science')
    except ValueError:pass
    else:raise AssertionError('Engineering model leaked into science')
    passed('zero_logits_routes_actor_only_export_and_role_stage_rejection')
    for r in ('PG-global','PG-cond'):
        episodes=[runtime.request(ledger,inputs,s,f'eng:{r}:{j}',r,actor=trainers[r].actor,seed=1162202600+j) for j,s in enumerate(specs)]
        report=trainers[r].update(episodes);assert report['accepted'] and report['actor_before']!=report['actor_after']
        del episodes
    control=trainers['continued_SL_A1']
    teachers=[runtime.teacher_episode(ledger,inputs,control.actor,i,f'eng:teacher:{j}') for j,i in enumerate(schedule['training'][0]['teacher_indices'])]
    report=control.update(teachers);assert report['accepted'];del teachers
    passed('real_online_updates_both_references_and_actual_matched_SL_no_Critic')
    e=runtime.request(ledger,inputs,specs[3],'eng:rejection','PG-global',actor=t.actor,seed=1162202677)
    before=copy.deepcopy(t.actor.correction.state_dict());opt=copy.deepcopy(t.optimizer.state_dict())
    result=t.update([e],force_reject=True)
    assert not result['accepted'] and result['rollback_exact'] and checkpoint_equal(before,t.actor.correction.state_dict()) and checkpoint_equal(opt,t.optimizer.state_dict())
    passed('actual_Adam_candidate_forced_rejection_exact_head_and_Adam_rollback')
    saved=t.save(p/'accepted.training.pt');snapshot=t.restore(saved)
    assert checkpoint_equal(t.actor.correction.state_dict(),snapshot['actor']) and checkpoint_equal(rng(),snapshot['rng'])
    restored=Trainer(load_actor('PG-global'),ref,ledger,identity);restored.restore(saved)
    e=runtime.request(ledger,inputs,specs[3],'eng:resume_next','PG-global',actor=t.actor,seed=1162202688)
    t.update([e]);restored.update([e])
    assert checkpoint_equal(t.actor.correction.state_dict(),restored.actor.correction.state_dict()) and checkpoint_equal(t.optimizer.state_dict(),restored.optimizer.state_dict())
    passed('checkpoint_actual_next_update_exact_Model_Adam_RNG_cursor_query_ledger')
    entry=export(p/'accepted.actor.pt',t.actor,usage='engineering',opportunity=t.counts['opportunities'])
    a=runtime.request(ledger,inputs,specs[3],'eng:updated_live','PG-global',actor=t.actor,greedy=True)
    b=runtime.request(ledger,inputs,specs[3],'eng:updated_export','PG-global',entry=entry,greedy=True)
    assert a['record']['raw_route']==b['record']['raw_route'] and a['record']['route']==b['record']['route'] and a['record']['metrics']==b['record']['metrics']
    passed('updated_production_Actor_only_complete_raw_H_route_equivalence')
    assert ledger.counts['requests']==19 and ledger.counts['actor_optimizer_attempts']==6
    put(ledger.path/'INDEPENDENT_NEW_OUTPUT_AUDIT.json',audit.run(ledger.path,ledger))
    put(ledger.path/'CHECKS.json',dict(status='PASS',tests=tests,actual_requests=19,actual_Actor_attempts=6,neural_Critic_created=0,
        engineering_models_must_not_initialize_science=True,forced_rejection_is_test_injection_not_scientific_rejection=True))

def science(ledger,trace,inputs,ref,identity):
    from .actor import load_actor,export
    from .trainer import Trainer
    from . import runtime,analysis,audit
    from rc3_conservative_refinement.runtime import tensor_digest
    schedule=read(HOME/'SCHEDULE.json');trainers={r:Trainer(load_actor(r),ref,ledger,identity) for r in ROLES}
    assert len({tensor_digest(t.actor.correction.state_dict()) for t in trainers.values()})==1
    models=ledger.path/'models';models.mkdir()
    for r,t in trainers.items():t.save(models/f'{r}_0000.training.pt')
    initial={r:tensor_digest(t.actor.correction.state_dict()) for r,t in trainers.items()}
    put(ledger.path/'INITIALIZATION.json',dict(fresh_same_tensors=initial,engineering_weights_loaded=False,neural_Critic_created=0))
    decisions=[]
    for row in schedule['training']:
        op=row['opportunity'];ledger.check()
        # Same task list and common random numbers. Each arm's CURRENT policy
        # samples its own trajectories; no updated or off-policy data are shared.
        for r in ('PG-global','PG-cond'):
            episodes=[]
            for slot,spec in enumerate(row['specs']):
                seed=int(__import__('hashlib').sha256(f'11622026|frozen-reference-PG|{op}|{slot}'.encode()).hexdigest()[:15],16)
                episodes.append(runtime.request(ledger,inputs,spec,f'train:{r}:{op}:{slot}',r,op=op,actor=trainers[r].actor,seed=seed))
            trainers[r].update(episodes);del episodes
        control=trainers['continued_SL_A1']
        teachers=[runtime.teacher_episode(ledger,inputs,control.actor,index,f'teacher:{op}:{slot}') for slot,index in enumerate(row['teacher_indices'])]
        control.update(teachers);del teachers
        if op%8==0:print(json.dumps(dict(opportunity=op,counts=ledger.counts,trainers={r:t.counts for r,t in trainers.items()},wall_s=ledger.elapsed())),flush=True)
        if any(t.counts['consecutive_rejected']>=16 for t in trainers.values()):raise RuntimeError('16 consecutive rejections: paired stop; no refill')
        if op in (64,128):
            with trace.stage('checkpoint_and_complete_evaluation_'+str(op)):
                entries={}
                for r,t in trainers.items():
                    t.save(models/f'{r}_{op:04d}.training.pt')
                    entries[r]=export(models/f'{r}_{op:04d}.actor.pt',t.actor,usage='science',opportunity=op)
                for ci,case in enumerate(schedule['evaluation']):
                    offset=(ci+op)%6;order=list(METHODS[offset:]+METHODS[:offset])
                    for r in order:
                        runtime.request(ledger,inputs,case['spec'],f'eval:{op}:{case["id"]}:{r}',r,op=op,kind='evaluation',
                            entry=entries.get(r),greedy=True,case=case)
                    if (ci+1)%11==0:print(f'Endpoint{op}: {ci+1}/66 cases x6 methods evaluated',flush=True)
                decisions.append(analysis.evaluate(ledger.path,op))
    with trace.stage('independent_new_request_and_online_diagnostic_audit'):
        put(ledger.path/'INDEPENDENT_NEW_OUTPUT_AUDIT.json',audit.run(ledger.path,ledger))
    put(ledger.path/'FINAL_DECISION.json',dict(status='QUALITY_SCREEN_PASS' if decisions[-1]['status']=='QUALITY_SCREEN_PASS' else 'STOP_CURRENT_CONFIGURATION',
        endpoints=decisions,automatic_extension=False,neural_Critic_updates=0,
        scope='single-seed exposed development only; not independent generalization or robot IK certification',
        trainer_counts={r:t.counts for r,t in trainers.items()}))

def main(stage):
    ledger=None;trace=None
    try:
        ledger,trace,inputs,ref,identity=bootstrap(stage)
        with trace.stage(stage+'_execution'):
            globals()[stage](ledger,trace,inputs,ref,identity)
        trace.check_loaded_modules();put(ledger.path/'ACTUAL_DEPENDENCY_VERIFICATION.json',trace.summary())
        assert sha(ROOT/TABLE)==ref.sha256
        ledger.status='COMPLETE';ledger.event('stage_complete',counts=ledger.counts,input_loader_counts=inputs.counts)
        put(ledger.path/'RESOURCE_ACCOUNTING.json',dict(ledger.snapshot(),input_loader_counts=inputs.counts))
        print(json.dumps(dict(stage=stage,status='COMPLETE',wall_s=ledger.elapsed(),counts=ledger.counts)),flush=True)
    except BaseException as e:
        # Bootstrap failures still retain an externally timed process receipt.
        if ledger is None:ledger=globals().get('CURRENT_LEDGER')
        target=ledger.path if ledger else HOME/stage
        target.mkdir(exist_ok=True)
        put(target/'FAILURE.json',dict(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),
            resource=ledger.snapshot() if ledger else None))
        if ledger:
            ledger.status='FAILED_PRESERVED';put(target/'RESOURCE_ACCOUNTING.json',ledger.snapshot())
        raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('engineering','science'));main(p.parse_args().stage)
