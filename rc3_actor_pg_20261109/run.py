"""Fixed output roots, parent-paid wall caps and one-shot scientific phases."""
import argparse
import collections
import os
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
from .common import *


def saved_provider(cases):
    from rc3_s3_evaluator.inputs import SavedDevelopmentInputs
    return SavedDevelopmentInputs(SimpleNamespace(scope='engineering',registration=dict(cases=cases)),HOME/'runs')


def deployment(mode,deadline,started):
    from unittest.mock import patch
    import torch
    from rc3_s3_evaluator.provenance import configure_runtime
    configure_runtime();admit();clock=Deadline(deadline);folder=HOME/'runs/DEPLOYMENT'/mode
    log=Log(folder)
    with patch.object(torch.optim.Optimizer,'__init__',side_effect=RuntimeError('Frozen deployment forbids optimizer')), \
         patch.object(torch.autograd,'backward',side_effect=RuntimeError('Frozen deployment forbids backward')):
        try:
            from .artifacts import export_all,Catalog
            if mode=='export':
                export_all();clock.check();write_new(folder/'COMPLETE.json',dict(status='COMPLETE',exported_models=12,wall_s=time.perf_counter()-started));return
            from rc3_s3_evaluator.inputs import NavigationPool
            from rc3_s3_evaluator.pipelines import evaluate
            registry=read(HOME/'INPUT_REGISTRY.json');cases=registry['development_cases'][:6]
            provider=saved_provider(cases);lookup={c['id']:c for c in cases}
            models=read(HOME/'actors/ACTOR_INDEX.json')['models'];catalog=Catalog(mode,models)
            pool=NavigationPool(identity([EXPERIMENT,mode]))
            write_new(folder/'STARTUP.json',dict(mode=mode,process_to_ready_s=time.perf_counter()-started,
                resident_models=len(catalog.resident),model_materializations=catalog.loads,
                process_start_scope='Parent timestamp immediately before Python child launch; includes imports/admission/provider/catalog setup',
                only_parameters_resident=True,dynamic_encodings_reused=False))
            for q in read(HOME/'DEPLOYMENT_GRID.json')['requests']:
                if q['mode']!=mode:continue
                clock.check();case=lookup[q['case_id']]
                if q['kind']=='hot':
                    prime=Meter(log,clock.check);pool.prime(provider.geometry(case,prime),prime)
                    log(dict(event='hot_prime',id=q['id'],counts=prime.counts,seconds=prime.seconds))
                log(dict(event='request_start',request=q));t=time.perf_counter();meter=Meter(log,clock.check)
                try:
                    row=evaluate(q,provider,case,pool,catalog,meter,clock.remaining)
                    row.update(experiment_id=EXPERIMENT,deployment_mode=mode,model_lifecycle=mode,navigation_cache=q['kind'])
                    with meter.span('artifact_write'):
                        path=folder/'results'/(q['id']+'.json');write_new(path,row);sha=digest(path)
                    elapsed=time.perf_counter()-t;clock.check()
                    log(dict(event='request_complete',id=q['id'],artifact_path=label(path),artifact_sha256=sha,
                        complete_time_s=elapsed,counts=meter.counts,seconds=meter.seconds))
                except BaseException as exc:
                    log(dict(event='request_error',id=q['id'],error=repr(exc),counts=meter.counts,seconds=meter.seconds));raise
                if sum(e['event']=='request_complete' for e in log.events)%21==0:
                    print(f'{mode} completed={sum(e["event"]=="request_complete" for e in log.events)}/252',flush=True)
            write_new(folder/'COMPLETE.json',dict(status='COMPLETE',requests=252,model_loads=catalog.loads,
                resident_hits=catalog.hits,event_tail=log.tail,model_tensor_equivalence='Checked at every load and every request'))
        finally:log.close()


def data_provider(clock,ledger,log):
    from rc3_global_local_trainer.approval import ExecutionPermit
    from rc3_global_local_trainer.data import TrainingRegistry,SceneProvider
    b=read(HOME/'READ_ONLY_DATA_SOURCE_BINDING.json')['binding'];reg=ROOT/'rc3_global_local_execution_v3/REGISTRY.json'
    # This object carries original DATA interpretation only, never old S2 authority.
    p=ExecutionPermit(ROOT/'rc3_global_local_execution_v3/V3_EXECUTION_CANDIDATE.json',reg,OLD_PROTOCOL,REGISTRY,b,
        ('read_only_source',),digest(HOME/'AUTHORIZATION.json'),digest(HOME/'AUTHORIZATION.json'))
    provider=SceneProvider(TrainingRegistry(reg),OLD_RUN/'data',ledger,p,record=log,read_only=True)
    provider.check=clock.check;return provider


def checkpoint(folder,trainer,provider,clock):
    from rc3_global_local_trainer.checkpoint import save_checkpoint
    cursor=trainer.cursors['RL'];p=folder/f'checkpoints/update_{cursor:04}.pt'
    receipt=save_checkpoint(p,trainer,dict(provider=provider.state_dict(),remaining_wall_s=clock.remaining(),
        protocol_sha256=digest(HOME/'PROTOCOL.json'),new_experiment=EXPERIMENT))
    write_new(p.with_suffix('.pt.commit.json'),dict(checkpoint_sha256=digest(p),sidecar_sha256=digest(p.with_suffix('.pt.json')),
        cursor=cursor,model_tensor_sha256=receipt['model_tensor_sha256'],closed_update_boundary=True))
    return receipt


def train(arm,deadline):
    import torch
    from rc3_s3_evaluator.provenance import configure_runtime
    configure_runtime();p=admit();clock=Deadline(deadline);folder=HOME/'runs'/arm;log=Log(folder);ledger={}
    from rc3_global_local.policy import GlobalLocalPolicy,tensor_hash
    from rc3_global_local_trainer.engine import make_optimizer
    from rc3_global_local_trainer.rng import NamedStreams
    from .mechanism import AuditTrainer,IIDTrainer,RecordingTerminal,save_gradient_and_weights
    try:
        e=read(ROOT/'rc3_s3_evaluator/S3_MODEL_INDEX.json')['models'][str(SEED)]['SL256']
        require(digest(ROOT/e['path'])==e['checkpoint_sha256'],'Shared SL256 source changed')
        state=torch.load(ROOT/e['path'],map_location='cpu',weights_only=False)
        with torch.device('meta'):model=GlobalLocalPolicy()
        model.load_state_dict(state['model'],strict=True,assign=True);del state
        model.train().requires_grad_(True);require(tensor_hash(model)==e['model_tensor_sha256'],'Wrong common warm start')
        optimizer=make_optimizer(model,p['mechanism']['optimizer']);require(len(optimizer.state)==0,'Inherited optimizer moments')
        provider=data_provider(clock,ledger,log)
        terminal=RecordingTerminal(ledger,folder,read(HOME/'READ_ONLY_DATA_SOURCE_BINDING.json')['binding']['H_sha256'],clock,log)
        streams=NamedStreams(OLD_PROTOCOL,SEED)
        config=dict(phase='RL',max_optimizer_updates=128,batch_instances=4,binding=dict(experiment_id=EXPERIMENT,arm=arm,
            protocol_sha256=digest(HOME/'PROTOCOL.json'),source_binding_sha256=digest(HOME/'SOURCE_BINDING.json'),
            registry_sha256=REGISTRY,H_identity=H_ID,data_rng_protocol=OLD_PROTOCOL,start=e))
        trainer=(AuditTrainer if arm=='A128' else IIDTrainer)(model,optimizer,streams,terminal,config,ledger)
        write_new(folder/'WEIGHT_BRANCH.json',dict(arm=arm,start=e,initial_tensor_sha256=tensor_hash(model),
            fresh_Adam=True,optimizer_state_entries=0,optimizer_parameters=optimizer.defaults,recipe=trainer.config))
        checkpoint(folder,trainer,provider,clock)
        for u,specs in enumerate(read(HOME/'INPUT_REGISTRY.json')['training_batches']):
            clock.check();log(dict(event='update_start',update=u+1,specs=specs,weight_tensor_before=tensor_hash(model)))
            views=[provider.get(s) for s in specs]
            diagnostics=trainer.s2_step(views,u)
            # Explicit identical per-batch input identities across both arms.
            diagnostics['specs']=specs;diagnostics['view_identities']=[v.identity for v in views]
            diagnostics['full_gradient_and_weight_artifact']=save_gradient_and_weights(folder/'gradients_and_weights',trainer,u+1)
            write_new(folder/'updates'/f'update_{u+1:04}.json',diagnostics)
            log(dict(event='update_complete',update=u+1,loss=diagnostics['loss'],gradient_norm=diagnostics['gradient_norm_before_clip'],
                clipped=diagnostics['gradient_norm_before_clip']>1.,zero_signal=diagnostics['zero_signal'],
                tensor_sha256=diagnostics['model_tensor_sha256'],terminal_requests=terminal.request_count,
                update_artifact_sha256=digest(folder/'updates'/f'update_{u+1:04}.json')))
            if (u+1)%32==0:checkpoint(folder,trainer,provider,clock)
            clock.check()
            print(f'{arm} update={u+1}/128 terminal={terminal.request_count}/1536 grad={diagnostics["gradient_norm_before_clip"]:.6f}',flush=True)
        require(terminal.request_count==1536 and trainer.cursors['RL']==128,'Wrong fixed endpoint/update count')
        write_new(folder/'COMPLETE.json',dict(status='COMPLETE_FIXED_128',updates=128,terminal_requests=terminal.request_count,
            tensor_sha256=tensor_hash(model),checkpoint=label(folder/'checkpoints/update_0128.pt'),ledger=ledger,
            event_tail=log.tail,start=e,recipe=trainer.config))
    finally:log.close()


def development(deadline):
    import torch
    from unittest.mock import patch
    from rc3_s3_evaluator.provenance import configure_runtime
    configure_runtime();admit();clock=Deadline(deadline);folder=HOME/'runs/DEV54';log=Log(folder)
    from .artifacts import export_one,Catalog
    from rc3_s3_evaluator.inputs import NavigationPool
    from rc3_s3_evaluator.pipelines import evaluate
    with patch.object(torch.optim.Optimizer,'__init__',side_effect=RuntimeError('Frozen DEV forbids optimizers')), \
         patch.object(torch.autograd,'backward',side_effect=RuntimeError('Frozen DEV forbids backward')):
        try:
            actors={'SL256':read(HOME/'actors/ACTOR_INDEX.json')['models'][str(SEED)]['SL256']}
            for arm in ('A128','B128'):
                complete=read(HOME/'runs'/arm/'COMPLETE.json')
                require(complete['updates']==128,'No fixed PG endpoint')
                actors[arm]=export_one(ROOT/complete['checkpoint'],HOME/'screen_actors',SEED,arm,complete['tensor_sha256'])
            write_new(folder/'ACTOR_INDEX.json',dict(models={str(SEED):actors}))
            cases=read(HOME/'INPUT_REGISTRY.json')['development_cases'];provider=saved_provider(cases)
            pool=NavigationPool(identity([EXPERIMENT,'DEV54']))
            done=0
            for case in cases:
                for role in ('SL256','A128','B128'):
                    clock.check();q=dict(seed=SEED,case_id=case['id'],method='frozen_supervised',study_method=role,kind='quality',repeat=0)
                    q['id']=identity(q);log(dict(event='request_start',request=q));t=time.perf_counter();meter=Meter(log,clock.check)
                    catalog=Catalog('D1',{str(SEED):actors},override_role=role)
                    try:
                        row=evaluate(q,provider,case,pool,catalog,meter,clock.remaining)
                        row.update(study_method=role,training_weight_role=role,new_experiment=EXPERIMENT)
                        row['plan']['method']=role
                        with meter.span('artifact_write'):
                            path=folder/'results'/(q['id']+'.json');write_new(path,row);sha=digest(path)
                        log(dict(event='request_complete',id=q['id'],artifact_path=label(path),artifact_sha256=sha,
                            complete_time_s=time.perf_counter()-t,counts=meter.counts,seconds=meter.seconds));done+=1
                    except BaseException as exc:
                        log(dict(event='request_error',id=q['id'],error=repr(exc),counts=meter.counts,seconds=meter.seconds));raise
                    print(f'DEV54 completed={done}/54 {case["id"]} {role}',flush=True)
            require(done==54,'Wrong DEV count');clock.check()
            write_new(folder/'COMPLETE.json',dict(status='COMPLETE',requests=done,event_tail=log.tail))
        finally:log.close()


def child(phase,mode,deadline,started,pid):
    from rc3_s3_recovery_20261108.storage import verify_supervisor_lease
    verify_supervisor_lease(HOME/'SUPERVISOR.lease',pid,EXPERIMENT)
    if phase=='DEPLOYMENT':deployment(mode,deadline,started)
    elif phase in ('A128','B128'):train(phase,deadline)
    else:require(phase=='DEV54','Bad phase');development(deadline)


def launch(phase,mode,end,folder):
    begin=time.perf_counter();tag=mode or phase;logfile=folder/f'process_{tag}.log'
    command=[sys.executable,'-B','-X','utf8','-m','rc3_actor_pg_20261109.run','--worker','--phase',phase,
        '--deadline',str(end),'--started',str(begin),'--supervisor-pid',str(os.getpid())]
    if mode:command+=['--mode',mode]
    with logfile.open('xb') as f:
        process=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        try:code=process.wait(timeout=max(.001,end-time.perf_counter()))
        except subprocess.TimeoutExpired:
            process.kill();process.wait();raise TimeoutError('Parent enforced phase hard wall cap')
    require(code==0,'Scientific worker stopped, preserve evidence: '+str(logfile))


def supervise():
    invocation=time.perf_counter();admit()
    with Lease(HOME/'SUPERVISOR.lease',EXPERIMENT):
        for phase,cap in (('DEPLOYMENT',1800),('A128',1800),('B128',1800),('DEV54',600)):
            start=invocation if phase=='DEPLOYMENT' else time.perf_counter();folder=HOME/'runs'/phase
            if (folder/'PHASE_FINAL.json').exists():
                final=read(folder/'PHASE_FINAL.json');require(final['status']=='COMPLETE','Failed phase cannot reset or silently restart');continue
            require(not folder.exists(),'Existing incomplete phase cannot reset budget')
            write_new(folder/'STARTED.json',dict(experiment_id=EXPERIMENT,phase=phase,wall_cap_s=cap,
                protocol_sha256=digest(HOME/'PROTOCOL.json'),source_binding_sha256=digest(HOME/'SOURCE_BINDING.json'),utc=now()))
            error=None;status='FAILED';end=start+cap
            try:
                if phase=='DEPLOYMENT':
                    for mode in ('export','D0','D1','D2'):launch(phase,mode,end,folder)
                    from .analysis import verify_deployment
                    verify_deployment()
                else:launch(phase,None,end,folder)
                require(time.perf_counter()<end,'Phase cap exceeded');status='COMPLETE'
            except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
            # Independent parent resource closure does not depend on failed worker log.
            write_new(folder/'PHASE_FINAL.json',dict(status=status,phase=phase,wall_s=time.perf_counter()-start,
                cap_s=cap,error=error,utc=now(),includes_startup_prepare_imports_H_validation_checkpoints_IO=True))
            require(error is None,'Phase stopped; '+str(error))
            print(f'{phase} COMPLETE',flush=True)
    print('All bounded phases finished; analyze saved data only.',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--worker',action='store_true')
    p.add_argument('--phase');p.add_argument('--mode');p.add_argument('--deadline',type=float);p.add_argument('--started',type=float);p.add_argument('--supervisor-pid',type=int)
    a=p.parse_args()
    if a.worker:child(a.phase,a.mode,a.deadline,a.started,a.supervisor_pid)
    elif a.execute:supervise()
    else:print('No scientific work without --execute')


if __name__=='__main__':main()
