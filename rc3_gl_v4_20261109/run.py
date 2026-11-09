"""One-shot396 deployment requests and one shared7200-second science phase."""
import argparse
import os
import subprocess
import sys
import time
import traceback
from .common import *


def complete_request(q,provider,case,pool,catalog,clock,log,folder):
    from .evaluation import evaluate
    clock.check();log(dict(event='request_start',request=q));start=time.perf_counter();meter=Meter(log,clock.check)
    try:
        row=evaluate(q,provider,case,pool,catalog,meter,clock.remaining)
        row.update(study_method=q.get('study_method',q['method']),model_lifecycle=q.get('mode','D1'),new_experiment=EXPERIMENT)
        with meter.span('artifact_write'):
            path=folder/'results'/(q['id']+'.json');write_new(path,row);sha=digest(path)
        clock.check();log(dict(event='request_complete',id=q['id'],artifact_path=label(path),artifact_sha256=sha,
            complete_time_s=time.perf_counter()-start,counts=meter.counts,seconds=meter.seconds))
        return row
    except BaseException as exc:
        log(dict(event='request_error',id=q['id'],error=repr(exc),counts=meter.counts,seconds=meter.seconds));raise


def deployment(deadline,started):
    import torch
    from unittest.mock import patch
    from rc3_s3_evaluator.provenance import configure_runtime
    configure_runtime();admit();clock=Deadline(deadline);folder=HOME/'runs/DEPLOYMENT';log=Log(folder)
    with patch.object(torch.optim.Optimizer,'__init__',side_effect=RuntimeError('Deployment optimizer forbidden')),\
         patch.object(torch.autograd,'backward',side_effect=RuntimeError('Deployment backward forbidden')):
        try:
            from .artifacts import export_all,Catalog
            from .data import saved_provider
            from rc3_s3_evaluator.inputs import NavigationPool
            export_start=time.perf_counter();models=export_all();export_s=time.perf_counter()-export_start
            cases=read(HOME/'INPUT_REGISTRY.json')['deployment_cases'];provider=saved_provider(cases);bycase={c['id']:c for c in cases}
            catalogs={};startups={};pools={}
            for mode in ('D0','D1','D2'):
                t=time.perf_counter();catalogs[mode]=Catalog(mode,models)
                startups[mode]=dict(catalog_and_model_ready_s=time.perf_counter()-t,resident_model_count=len(catalogs[mode].resident),
                    model_loads=catalogs[mode].loads,model_parameters_only_resident=True)
                pools[mode]=NavigationPool(identity([EXPERIMENT,mode]))
            write_new(folder/'STARTUP.json',dict(process_to_all_catalogs_ready_s=time.perf_counter()-started,
                export_and_tensor_roundtrip_s=export_s,modes=startups,
                scope='Parent timestamp before Python child launch; imports,admission,exports and all three catalog setup included; catalog subclocks separate',
                model_state_not_navigation_cache_label=True))
            stable={};encodings=0;done=0
            for q in read(HOME/'DEPLOYMENT_GRID.json')['requests']:
                clock.check();case=bycase[q['case_id']];pool=pools[q['mode']]
                if q['kind']=='hot':
                    meter=Meter(log,clock.check);pool.prime(provider.geometry(case,meter),meter)
                    log(dict(event='hot_prime',id=q['id'],counts=meter.counts,seconds=meter.seconds))
                row=complete_request(q,provider,case,pool,catalogs[q['mode']],clock,log,folder)
                key=(q['case_id'],q['role']);v=(row['raw_route'],row['route'],row['metrics'],row['source']['view_identity'],
                    row['source']['D_sha256'],row['trace'],row['plan']['assignments'])
                require(key not in stable or stable[key]==v,'Lifecycle changed mask/logits/route/metrics/assignments')
                stable[key]=v;encodings+=row['counts'].get('fresh_encodings',0);done+=1
                if done%6==0:print(f'DEPLOYMENT completed={done}/396',flush=True)
            require(done==396 and encodings==396 and len(stable)==12,'Wrong deployment/equivalence grid')
            for role,m in read(ROOT/'rc3_s3_evaluator/S3_MODEL_INDEX.json')['models'][str(SEED)].items():
                require(digest(ROOT/m['path'])==m['checkpoint_sha256'],'Export/evaluation changed old checkpoint')
            write_new(folder/'EQUIVALENCE.json',dict(status='PASS',requests=done,case_weight_pairs=12,fresh_encodings=encodings,
                actual_masks_logits_raw_final_routes_assignments_JLK_exact=True,precision_changed=False,
                mode_model_loads={m:c.loads for m,c in catalogs.items()},resident_hits={m:c.hits for m,c in catalogs.items()},
                common_H_results_not_cached=True,old_model_bytes_unchanged=True))
            write_new(folder/'COMPLETE.json',dict(status='COMPLETE',requests=done,event_tail=log.tail))
        finally:log.close()


class NoQueryTerminal:
    def __call__(self,*args):raise RuntimeError('C supervised branch may not query a terminal')
    def state_dict(self):return dict(teacher_queries=0,native=0,H=0)


def checkpoint(folder,trainer,provider,clock):
    from rc3_global_local_trainer.checkpoint import save_checkpoint
    cursor=trainer.cursors[trainer.phase];p=folder/f'checkpoints/update_{cursor:04}.pt'
    receipt=save_checkpoint(p,trainer,dict(provider=provider.state_dict(),remaining_wall_s=clock.remaining(),
        protocol_sha256=digest(HOME/'PROTOCOL.json'),new_experiment=EXPERIMENT))
    write_new(p.with_suffix('.pt.commit.json'),dict(checkpoint_sha256=digest(p),sidecar_sha256=digest(p.with_suffix('.pt.json')),
        cursor=cursor,model_tensor_sha256=receipt['model_tensor_sha256'],closed_update_boundary=True))
    return receipt


def train(arm,clock):
    import torch
    from rc3_global_local.policy import GlobalLocalPolicy,tensor_hash
    from rc3_global_local_trainer.engine import make_optimizer
    from rc3_global_local_trainer.rng import NamedStreams
    from .mechanism import AuditTrainer,NaturalTrainer,RecordingTerminal,save_gradient_and_weights
    from .data import training_provider
    p=read(HOME/'PROTOCOL.json');folder=HOME/'runs/SCIENCE'/arm;log=Log(folder);ledger={}
    try:
        clock.check();e=read(ROOT/'rc3_s3_evaluator/S3_MODEL_INDEX.json')['models'][str(SEED)]['SL256']
        require(digest(ROOT/e['path'])==e['checkpoint_sha256'],'Shared SL256 source changed')
        state=torch.load(ROOT/e['path'],map_location='cpu',weights_only=False)
        with torch.device('meta'):model=GlobalLocalPolicy()
        model.load_state_dict(state['model'],strict=True,assign=True);del state
        model.train().requires_grad_(True);require(tensor_hash(model)==e['model_tensor_sha256'],'Wrong shared exact weight')
        optimizer=make_optimizer(model,p['mechanism']['optimizer']);require(not optimizer.state,'Adam momentum inherited')
        provider=training_provider(clock,ledger,log)
        terminal=NoQueryTerminal() if arm=='C128' else RecordingTerminal(ledger,folder,
            read(HOME/'READ_ONLY_DATA_SOURCE_BINDING.json')['binding']['H_sha256'],clock,log)
        phase='continued_SL' if arm=='C128' else 'RL';streams=NamedStreams(OLD_PROTOCOL,SEED)
        config=dict(phase=phase,max_optimizer_updates=128,batch_instances=4,binding=dict(experiment_id=EXPERIMENT,arm=arm,
            protocol_sha256=digest(HOME/'PROTOCOL.json'),source_binding_sha256=digest(HOME/'SOURCE_BINDING.json'),
            registry_sha256=REGISTRY,H_identity=H_ID,data_rng_protocol=OLD_PROTOCOL,start=e))
        trainer=(NaturalTrainer if arm=='B128' else AuditTrainer)(model,optimizer,streams,terminal,config,ledger)
        write_new(folder/'WEIGHT_BRANCH.json',dict(arm=arm,start=e,initial_tensor_sha256=tensor_hash(model),fresh_Adam=True,
            optimizer_state_entries=0,recipe=trainer.config,initial_parameter_storage={n:p.data_ptr() for n,p in model.named_parameters()}))
        checkpoint(folder,trainer,provider,clock)
        registry=read(HOME/'INPUT_REGISTRY.json');pool=None
        if arm=='C128':
            from rc3_global_local_trainer.runner import TeacherPool
            pool=TeacherPool(OLD_RUN/'teachers256',provider.registry,provider,ledger,clock,
                read(HOME/'READ_ONLY_DATA_SOURCE_BINDING.json')['binding']['H_sha256']);pool.validate()
        for u in range(128):
            clock.check();log(dict(event='update_start',update=u+1,arm=arm,weight_tensor_before=tensor_hash(model)))
            if arm=='C128':
                indices=registry['C_teacher_batches'][u];examples=[pool.example(i) for i in indices]
                d=trainer.s1_step(examples,u);d['teacher_indices']=indices
            else:
                specs=registry['training_batches'][u];views=[provider.get(s) for s in specs]
                d=trainer.s2_step(views,u);d['specs']=specs;d['view_identities']=[v.identity for v in views]
            d['full_gradient_and_weight_artifact']=save_gradient_and_weights(folder/'gradients_and_weights',trainer,u+1)
            write_new(folder/'updates'/f'update_{u+1:04}.json',d)
            log(dict(event='update_complete',update=u+1,arm=arm,loss=d['loss'],gradient_norm=d['gradient_norm_before_clip'],
                gradient_norm_after_clip=d['gradient_norm_after_clip'],clipped=d['gradient_norm_before_clip']>1.,
                tensor_sha256=d['model_tensor_sha256'],terminal_requests=0 if arm=='C128' else terminal.request_count,
                update_artifact_sha256=digest(folder/'updates'/f'update_{u+1:04}.json')))
            if (u+1)%32==0:checkpoint(folder,trainer,provider,clock)
            clock.check();print(f'{arm} update={u+1}/128 terminal={0 if arm=="C128" else terminal.request_count} grad={d["gradient_norm_before_clip"]:.6f}',flush=True)
        require(trainer.cursors[phase]==128 and (arm=='C128' or terminal.request_count==1536),'Wrong fixed endpoint')
        write_new(folder/'COMPLETE.json',dict(status='COMPLETE_FIXED_128',updates=128,terminal_requests=0 if arm=='C128' else terminal.request_count,
            tensor_sha256=tensor_hash(model),checkpoint=label(folder/'checkpoints/update_0128.pt'),ledger=ledger,
            event_tail=log.tail,start=e,recipe=trainer.config))
    finally:log.close()


def development(cases,clock):
    import torch
    from unittest.mock import patch
    from .artifacts import export_one,Catalog
    from .data import NewDevelopmentInputs,SavedNavigation
    folder=HOME/'runs/SCIENCE/DEV216';log=Log(folder)
    with patch.object(torch.optim.Optimizer,'__init__',side_effect=RuntimeError('Frozen DEV optimizer forbidden')),\
         patch.object(torch.autograd,'backward',side_effect=RuntimeError('Frozen DEV backward forbidden')):
        try:
            actors={'SL256':read(HOME/'actors/ACTOR_INDEX.json')['models'][str(SEED)]['SL256']}
            for arm in ('A128','B128','C128'):
                c=read(HOME/'runs/SCIENCE'/arm/'COMPLETE.json');require(c['updates']==128,'Intermediate checkpoint refused')
                actors[arm]=export_one(ROOT/c['checkpoint'],HOME/'endpoint_actors',SEED,arm,c['tensor_sha256'])
            write_new(folder/'ACTOR_INDEX.json',dict(models={str(SEED):actors}))
            provider=NewDevelopmentInputs(cases);pool=SavedNavigation(read(HOME/'INPUT_REGISTRY.json')['new_development_parents'],log,clock)
            count=0
            for case in cases:
                for role in ('SL256','A128','B128','C128','Fixed3','Detour16'):
                    method={'Fixed3':'Fixed3_endpoint_adapter','Detour16':'Detour16_endpoint_adapter'}.get(role,'frozen_supervised')
                    q=dict(seed=SEED,case_id=case['id'],method=method,study_method=role,kind='quality',repeat=0);q['id']=identity(q)
                    catalog=Catalog('D1',{str(SEED):actors},override_role=role) if role not in ('Fixed3','Detour16') else None
                    complete_request(q,provider,case,pool,catalog,clock,log,folder);count+=1
                    print(f'DEV216 completed={count}/216 {case["id"]} {role}',flush=True)
            require(count==216,'Wrong fixed evaluation grid');clock.check()
            write_new(folder/'COMPLETE.json',dict(status='COMPLETE',requests=count,event_tail=log.tail))
        finally:log.close()


def science(deadline):
    from rc3_s3_evaluator.provenance import configure_runtime
    configure_runtime();admit();clock=Deadline(deadline);folder=HOME/'runs/SCIENCE';log=Log(folder/'preparation')
    try:
        from .data import prepare_bank
        cases=prepare_bank(clock,log)
    finally:log.close()
    for arm in ('A128','B128','C128'):train(arm,clock)
    development(cases,clock)


def launch(phase,end,folder):
    begin=time.perf_counter();command=[sys.executable,'-B','-X','utf8','-m','rc3_gl_v4_20261109.run','--worker',
        '--phase',phase,'--deadline',str(end),'--started',str(begin),'--supervisor-pid',str(os.getpid())]
    with (folder/'process.log').open('xb') as f:
        process=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        try:code=process.wait(timeout=max(.001,end-time.perf_counter()))
        except subprocess.TimeoutExpired:
            process.kill();process.wait();raise TimeoutError('Hard cumulative parent wall cap')
    require(code==0,'Worker stopped; preserve process.log and all consumed resources')


def supervise():
    invocation=time.perf_counter()
    with Lease(HOME/'SUPERVISOR.lease',EXPERIMENT):
        for phase,cap in (('DEPLOYMENT',1800),('SCIENCE',7200)):
            start=invocation if phase=='DEPLOYMENT' else time.perf_counter();folder=HOME/'runs'/phase
            require(not folder.exists(),'One-shot phase exists; directory/budget cannot be reset')
            write_new(folder/'STARTED.json',dict(experiment=EXPERIMENT,phase=phase,wall_cap_s=cap,
                protocol_sha256=digest(HOME/'PROTOCOL.json'),source_binding_sha256=digest(HOME/'SOURCE_BINDING.json'),utc=now()))
            error=None;status='FAILED';end=start+cap
            try:
                if phase=='DEPLOYMENT':
                    from rc3_s3_evaluator.provenance import configure_runtime
                    configure_runtime()
                    from .tests import verify
                    verify()
                admit();launch(phase,end,folder);require(time.perf_counter()<end,'Cumulative cap exceeded');status='COMPLETE'
            except BaseException as exc:error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
            write_new(folder/'PHASE_FINAL.json',dict(status=status,phase=phase,wall_s=time.perf_counter()-start,cap_s=cap,
                error=error,utc=now(),includes_startup_source_prepare_navigation_H_validation_checkpoints_IO=True))
            require(error is None,'Stopped phase; '+str(error));print(phase+' COMPLETE',flush=True)
    print('All GL-v4 bounded phases finished; saved-data analysis only.',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--worker',action='store_true')
    p.add_argument('--phase');p.add_argument('--deadline',type=float);p.add_argument('--started',type=float);p.add_argument('--supervisor-pid',type=int);a=p.parse_args()
    if a.worker:
        from rc3_s3_recovery_20261108.storage import verify_supervisor_lease
        verify_supervisor_lease(HOME/'SUPERVISOR.lease',a.supervisor_pid,EXPERIMENT)
        if a.phase=='DEPLOYMENT':deployment(a.deadline,a.started)
        else:require(a.phase=='SCIENCE','Unknown phase');science(a.deadline)
    elif a.execute:supervise()
    else:print('GL-v4 requires explicitly authorized --execute')


if __name__=='__main__':main()
