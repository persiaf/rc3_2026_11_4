"""One-shot authorized G128 plus72x6 evaluation, shared7200-second hard cap."""
import argparse
import os
import subprocess
import sys
import time
import traceback
from .common import *


def checkpoint(folder,trainer,provider,clock):
    from rc3_global_local_trainer.checkpoint import save_checkpoint
    u=trainer.cursors['RL'];p=folder/f'checkpoints/update_{u:04}.pt'
    receipt=save_checkpoint(p,trainer,dict(provider=provider.state_dict(),remaining_wall_s=clock.remaining(),
        protocol_sha256=digest(HOME/'PROTOCOL.json'),experiment=EXPERIMENT))
    write_new(p.with_suffix('.pt.commit.json'),dict(checkpoint_sha256=digest(p),sidecar_sha256=digest(p.with_suffix('.pt.json')),
        cursor=u,model_tensor_sha256=receipt['model_tensor_sha256'],closed_update_boundary=True))


def train(clock):
    import torch
    from rc3_global_local.policy import GlobalLocalPolicy,tensor_hash
    from rc3_global_local_trainer.engine import make_optimizer
    from rc3_global_local_trainer.rng import NamedStreams
    from .mechanism import GreedyBaselineTrainer,RecordingTerminal,save_gradient_and_weights
    from .reference import FrozenReference
    from .data import training_provider
    folder=HOME/'runs/SCIENCE/G128';log=Log(folder);ledger={}
    try:
        clock.check();r=read(HOME/'INPUT_REGISTRY.json');p=read(HOME/'PROTOCOL.json')
        e=read(ROOT/'rc3_s3_evaluator/S3_MODEL_INDEX.json')['models'][str(SEED)]['SL256']
        require(digest(ROOT/e['path'])==e['checkpoint_sha256'],'Common SL256 changed')
        payload=torch.load(ROOT/e['path'],map_location='cpu',weights_only=False)
        with torch.device('meta'):model=GlobalLocalPolicy()
        model.load_state_dict(payload['model'],strict=True,assign=True);del payload
        model.train().requires_grad_(True);require(tensor_hash(model)==e['model_tensor_sha256'],'Wrong common SL256')
        optimizer=make_optimizer(model,p['training']['optimizer']);require(not optimizer.state,'Adam inherited momentum')
        provider=training_provider(clock,ledger,log)
        reference=FrozenReference(folder/'reference',clock,log,r['actors']['SL256'])
        terminal=RecordingTerminal(ledger,folder,H_ID,clock,log);terminal.reference=reference
        config=dict(phase='RL',max_optimizer_updates=128,batch_instances=4,binding=dict(experiment_id=EXPERIMENT,
            arm='G128',protocol_sha256=digest(HOME/'PROTOCOL.json'),source_binding_sha256=digest(HOME/'SOURCE_BINDING.json'),
            registry_sha256=REGISTRY,H_identity=H_ID,data_rng_protocol=OLD_PROTOCOL,start=e))
        trainer=GreedyBaselineTrainer(model,optimizer,NamedStreams(OLD_PROTOCOL,SEED),terminal,config,ledger)
        trainer.reference=reference
        write_new(folder/'WEIGHT_BRANCH.json',dict(study_method='G128',checkpoint_role='G128',start=e,
            initial_tensor_sha256=tensor_hash(model),fresh_Adam=True,optimizer_state_entries=0,recipe=trainer.config,
            reference_identity=reference.identity))
        checkpoint(folder,trainer,provider,clock)
        for u,specs in enumerate(r['training_batches']):
            clock.check();log(dict(event='update_start',update=u+1,study_method='G128',weight_tensor_before=tensor_hash(model)))
            views=[provider.get(s) for s in specs];trainer.instance_gradient_norms=[]
            d=trainer.s2_step(views,u);d.update(specs=specs,view_identities=[v.identity for v in views],study_method='G128',checkpoint_role='G128')
            d['full_gradient_and_weight_artifact']=save_gradient_and_weights(folder/'gradients_and_weights',trainer,u+1)
            write_new(folder/'updates'/f'update_{u+1:04}.json',d)
            log(dict(event='update_complete',update=u+1,loss=d['loss'],gradient_norm=d['gradient_norm_before_clip'],
                clipped=d['gradient_norm_before_clip']>1.,tensor_sha256=d['model_tensor_sha256'],
                sample_requests=terminal.request_count,reference_requests=reference.requests,
                update_artifact_sha256=digest(folder/'updates'/f'update_{u+1:04}.json')))
            if (u+1)%32==0:checkpoint(folder,trainer,provider,clock)
            clock.check();print(f'G128 update={u+1}/128 sampled={terminal.request_count}/1536 reference={reference.requests}/512 grad={d["gradient_norm_before_clip"]:.5f}',flush=True)
        require(trainer.cursors['RL']==128 and terminal.request_count==1536 and reference.requests==512,'Wrong fixed G endpoint')
        write_new(folder/'COMPLETE.json',dict(status='COMPLETE_FIXED_128',updates=128,sample_requests=terminal.request_count,
            reference_requests=reference.requests,tensor_sha256=tensor_hash(model),checkpoint=label(folder/'checkpoints/update_0128.pt'),
            ledger=ledger,reference_state=reference.state_dict(),event_tail=log.tail,start=e,recipe=trainer.config))
    finally:log.close()


def development(cases,clock):
    import torch
    from unittest.mock import patch
    from rc3_gl_v4_20261109.artifacts import export_one,Catalog
    from rc3_gl_v4_20261109.evaluation import evaluate
    from .data import NewDevelopmentInputs,SavedNavigation
    from .identities import label_result,actor_identity
    folder=HOME/'runs/SCIENCE/DEV432';log=Log(folder)
    with patch.object(torch.optim.Optimizer,'__init__',side_effect=RuntimeError('Evaluation optimizer forbidden')),\
         patch.object(torch.autograd,'backward',side_effect=RuntimeError('Evaluation backward forbidden')):
        try:
            r=read(HOME/'INPUT_REGISTRY.json');actors=r['actors'].copy();done=read(HOME/'runs/SCIENCE/G128/COMPLETE.json')
            require(done['updates']==128,'Intermediate endpoint refused')
            actors['G128']=export_one(ROOT/done['checkpoint'],HOME/'actors',SEED,'G128',done['tensor_sha256'])
            for role,e in actors.items():actor_identity(role,e)
            write_new(folder/'ACTOR_INDEX.json',dict(models={str(SEED):actors}))
            provider=NewDevelopmentInputs(cases);pool=SavedNavigation(r['new_development_parents'],log,clock);count=0
            for case in cases:
                for role in ('SL256','B128','C128','G128','Fixed3','Detour16'):
                    method={'Fixed3':'Fixed3_endpoint_adapter','Detour16':'Detour16_endpoint_adapter'}.get(role,'frozen_supervised')
                    q=dict(seed=SEED,case_id=case['id'],method=method,study_method=role,kind='quality',repeat=0);q['id']=identity(q)
                    clock.check();require(count<432,'Evaluation request cap');log(dict(event='request_start',request=q));tick=time.perf_counter();meter=Meter(log,clock.check)
                    try:
                        catalog=Catalog('D1',{str(SEED):actors},override_role=role) if role in actors else None
                        row=evaluate(q,provider,case,pool,catalog,meter,clock.remaining)
                        row=label_result(row,role);row.update(new_experiment=EXPERIMENT,model_lifecycle='D1' if role in actors else None)
                        with meter.span('artifact_write'):
                            path=folder/'results'/(q['id']+'.json');write_new(path,row);sha=digest(path)
                        clock.check();log(dict(event='request_complete',id=q['id'],artifact_path=label(path),artifact_sha256=sha,
                            complete_time_s=time.perf_counter()-tick,counts=meter.counts,seconds=meter.seconds));count+=1
                    except BaseException as exc:
                        log(dict(event='request_error',id=q['id'],error=repr(exc),counts=meter.counts,seconds=meter.seconds));raise
                    print(f'DEV432 completed={count}/432 {case["id"]} {role}',flush=True)
            require(count==432,'Incomplete fixed evaluation');write_new(folder/'COMPLETE.json',dict(status='COMPLETE',requests=count,event_tail=log.tail))
        finally:log.close()


def science(end):
    from rc3_s3_evaluator.provenance import configure_runtime
    configure_runtime();admit();clock=Deadline(end);log=Log(HOME/'runs/SCIENCE/preparation')
    try:
        from .data import prepare_bank
        cases=prepare_bank(clock,log)
    finally:log.close()
    train(clock);development(cases,clock)


def supervise():
    tick=time.perf_counter();carry=read(HOME/'REGISTRATION_ACCOUNTING.json')['wall_s'];end=tick+7200-carry
    folder=HOME/'runs/SCIENCE';require(not folder.exists(),'One-shot GL-v5 run exists; consumed budget cannot reset')
    with Lease(HOME/'SUPERVISOR.lease',EXPERIMENT):
        write_new(folder/'STARTED.json',dict(experiment=EXPERIMENT,cap_s=7200,registration_carry_s=carry,
            protocol_sha256=digest(HOME/'PROTOCOL.json'),source_binding_sha256=digest(HOME/'SOURCE_BINDING.json'),utc=now()))
        status='FAILED';error=None
        try:
            from rc3_s3_evaluator.provenance import configure_runtime
            configure_runtime()
            from .tests import verify
            verify();admit()
            cmd=[sys.executable,'-B','-X','utf8','-m','rc3_gl_v5_20261109.run','--worker','--deadline',str(end),'--supervisor-pid',str(os.getpid())]
            with (folder/'process.log').open('xb') as f:
                process=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
                try:code=process.wait(timeout=max(.001,end-time.perf_counter()))
                except subprocess.TimeoutExpired:
                    process.kill();process.wait();raise TimeoutError('Shared parent7200 cap')
            require(code==0,'Worker stopped; preserve debits and process.log');require(time.perf_counter()<end,'Shared budget exceeded');status='COMPLETE'
        except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        write_new(folder/'PHASE_FINAL.json',dict(status=status,wall_s=time.perf_counter()-tick+carry,cap_s=7200,
            registration_carry_s=carry,error=error,includes_all_registration_tests_prepare_reference_H_train_evaluate_validation_IO=True,utc=now()))
        require(error is None,'GL-v5 stopped; '+str(error))
    print('GL-v5 science COMPLETE; saved-data analysis only',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--worker',action='store_true')
    p.add_argument('--deadline',type=float);p.add_argument('--supervisor-pid',type=int);a=p.parse_args()
    if a.worker:
        from rc3_s3_recovery_20261108.storage import verify_supervisor_lease
        verify_supervisor_lease(HOME/'SUPERVISOR.lease',a.supervisor_pid,EXPERIMENT);science(a.deadline)
    elif a.execute:supervise()
    else:print('GL-v5 explicit authorized --execute required')


if __name__=='__main__':main()
