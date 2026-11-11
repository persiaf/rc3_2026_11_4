"""One supervised recovery; frozen M1 scientific functions reused unchanged."""
import time
ENTRY=time.perf_counter()
import csv
import gzip
import io
import json
import os
import platform
import random
import subprocess
import traceback
import unittest
from .ledger import Budget,HOME,ROOT,digest,write_new

def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))

def main():
    budget=Budget('engineering_and_M1',ENTRY)
    logs=[]
    try:
        from .provenance import TracedProvenance
        binding=read(HOME/'SOURCE_BINDING.json')
        provenance=TracedProvenance(ROOT,HOME,binding,budget.check,budget.run_id)
        budget.tracer=provenance
        provenance.install_import_guard()
        provenance.event('stage_start',name='preflight_and_imports')
        for name, expected in binding['startup_files_sha256'].items():
            if digest(HOME/name)!=expected: raise ValueError('Startup control identity differs: '+name)
        authorization=read(HOME/'STAGE_AUTHORIZATION.json')
        review=read(HOME/'IMPLEMENTATION_REVIEW.json')
        if (authorization['authorized'] is not True or
            authorization['source_binding_sha256']!=digest(HOME/'SOURCE_BINDING.json') or
            review['status']!='GO' or review['source_binding_sha256']!=digest(HOME/'SOURCE_BINDING.json')):
            raise ValueError('Exact supervised M1 authorization/review missing')
        engineering=read(HOME/'engineering'/'ENGINEERING_TEST_RESULTS.json')
        if engineering['status']!='GO':raise ValueError('Synthetic timeout engineering did not pass')
        with provenance.stage('heavy_imports'):
            import numpy as np
            import torch
            from rc3_r04_m1_20261111 import critic,data,metrics,test_critic,restricted_inputs
            from . import review_adapter as independent_review
            from rc3_critic_diagnostics_20261111 import critic as old_critic
            from rc3_conservative_refinement import saved_inputs as si,runtime
        provenance.check_loaded_modules()
        def checked_hash(path):
            budget.charge('source_reads');budget.charge('source_hashes')
            return provenance.hash_file(path)
        registration=provenance.read_json(HOME/'M1_DATA_AND_SOURCE_IDENTITY.json')
        schedule=provenance.read_json(HOME/'BATCH_SCHEDULE.json')['batches']
        for name,key in (('M1_EXECUTION_PROTOCOL.json','protocol_sha256'),('M1_DATA_AND_SOURCE_IDENTITY.json','data_sha256'),('BATCH_SCHEDULE.json','schedule_sha256')):
            if checked_hash(HOME/name)!=binding[key]:raise ValueError('Frozen M1 science bytes changed')
        if registration['status']!='GO_METADATA' or len(registration['trajectories'])!=116:
            raise ValueError('Locked M1 selection differs')
        provenance.event('stage_end',name='preflight_and_imports',status='GO')
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        torch.manual_seed(11622026);np.random.seed(11622026);random.seed(11622026)
        environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
            device='cpu',threads=torch.get_num_threads(),deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
        if environment!=binding['environment']: raise ValueError('Exact registered environment differs')
        write_new(HOME/'ACTUAL_ENVIRONMENT.json',dict(environment,platform=platform.platform(),
            code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
        original_call=torch.nn.Module._call_impl
        def counted_call(self,*args,**kwargs):
            budget.charge('module_forward_calls');return original_call(self,*args,**kwargs)
        torch.nn.Module._call_impl=counted_call
        original_backward=torch.autograd.backward
        def counted_backward(*args,**kwargs):
            budget.charge('backward_calls');return original_backward(*args,**kwargs)
        torch.autograd.backward=counted_backward
        original_step=torch.optim.Adam.step
        admitted_step={'optimizer':None,'configuration':None}
        def protected_step(self,*args,**kwargs):
            if self is not admitted_step['optimizer']: raise RuntimeError('Unregistered/Actor Adam update forbidden')
            result=original_step(self,*args,**kwargs)
            budget.charge('critic_optimizer_completed');admitted_step['optimizer']=None
            return result
        torch.optim.Adam.step=protected_step
        provenance.event('stage_start',name='zero_update_critic_interface_tests')
        suite=unittest.defaultTestLoader.loadTestsFromModule(test_critic)
        stream=io.StringIO();t=time.perf_counter()
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        write_new(HOME/'UNIT_TEST_RESULTS.json',dict(status='GO' if result.wasSuccessful() else 'FAIL',
            tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),wall_s=time.perf_counter()-t,
            optimizer_attempts=0,output=stream.getvalue()))
        if not result.wasSuccessful(): raise RuntimeError('Registered engineering unit test failed')
        provenance.event('stage_end',name='zero_update_critic_interface_tests',status='GO')
        torch.manual_seed(11622026);np.random.seed(11622026);random.seed(11622026)
        old_read,old_numpy_load=si.read,np.load
        def audited_read(path):
            budget.charge('source_reads');return provenance.read_json(path)
        def audited_load(path,*args,**kwargs):
            if isinstance(path,(str,os.PathLike)):
                budget.charge('source_reads');return old_numpy_load(io.BytesIO(provenance.read_bytes(path)),*args,**kwargs)
            return old_numpy_load(path,*args,**kwargs)
        si.read=audited_read;si.digest=checked_hash;np.load=audited_load;runtime.file_sha256=checked_hash
        from rc3_global_local_trainer import navigation_compat,approval
        from rc3_global_local_impl_review import runtime_guards
        navigation_compat.read=approval.read=audited_read
        navigation_compat.digest=approval.digest=runtime_guards.file_hash=checked_hash
        from rc3_critic_diagnostics_20261111 import data as legacy_data
        def exact_saved_bytes(path,charge):
            p=__import__('pathlib').Path(path)
            p=(p if p.is_absolute() else ROOT/p).resolve()
            charge('source_reads');charge('source_hashes')
            payload=provenance.read_bytes(p)
            return payload,dict(path=p.relative_to(ROOT).as_posix(),sha256=provenance.hash_file(p),bytes=len(payload))
        legacy_data._bytes=exact_saved_bytes
        original_torch_load=torch.load
        def verified_torch_load(path,*args,**kwargs):
            if isinstance(path,(str,os.PathLike)) and not __import__('pathlib').Path(path).resolve().is_relative_to(HOME):
                return original_torch_load(io.BytesIO(provenance.read_bytes(path)),*args,**kwargs)
            return original_torch_load(path,*args,**kwargs)
        torch.load=verified_torch_load
        provenance.event('stage_start',name='saved_input_loader')
        inputs=restricted_inputs.make_inputs(registration)
        def forbidden(*args,**kwargs):
            budget.emit('forbidden_path_attempt');raise RuntimeError('Planning/action/env.step/teacher/navigation/H forbidden')
        inputs.teacher=forbidden
        runtime.rollout=runtime.plan_request=runtime.sl_route=forbidden
        from rc3_global_local.environment import Environment
        from rc3_global_local import terminal
        Environment.step=forbidden;terminal.finish=terminal.export=forbidden
        provenance.event('stage_end',name='saved_input_loader',status='GO')
        provenance.event('stage_start',name='frozen_SL_loading')
        budget.charge('model_loads',role='frozen_exact_SL256')
        encoder=runtime.load_actor(True,11622026);encoder.eval().requires_grad_(False)
        base_before=critic.tensor_digest(encoder.encoder.state_dict())
        provenance.event('stage_end',name='frozen_SL_loading',status='GO')
        provenance.event('stage_start',name='restore_116_saved_trajectories')
        episodes=[];restored=[];t=time.perf_counter()
        for i,row in enumerate(registration['trajectories']):
            budget.admit('restore_saved_trajectory')
            provenance.event('trajectory_restore_start',trajectory_id=row['trajectory_id'])
            budget.charge('saved_trajectories',trajectory_id=row['trajectory_id'])
            episode=data.load_episode(row,encoder,inputs,budget.charge)
            critic.episode_labels(episode)
            episodes.append(episode)
            provenance.event('trajectory_restore_end',trajectory_id=row['trajectory_id'],status='GO',states=len(episode['v']))
            restored.append(dict(trajectory_id=episode['trajectory_id'],parent_id=episode['parent_id'],split=episode['split'],
                window=episode['window_key'],source_stage=episode['stage'],task_count=episode['task_count'],states=len(episode['v']),
                feature_tensor_sha256=critic.tensor_digest({'v':episode['v']}),request_source=episode['source']))
            if (i+1)%16==0: print(f'Restored {i+1}/{len(registration["trajectories"])} exact saved trajectories',flush=True)
        if (critic.tensor_digest(encoder.encoder.state_dict())!=base_before or
            any(p.requires_grad or p.grad is not None for p in encoder.encoder.parameters())):
            raise ValueError('Frozen SL encoder changed')
        write_new(HOME/'RESTORED_INPUT_INDEX.json',dict(trajectories=restored,base_tensor_sha256=base_before,
            source='saved prefixes only; 268 reconstruction, no new labels/decisions',wall_s=time.perf_counter()-t,loader_counts=inputs.counts))
        provenance.event('stage_end',name='restore_116_saved_trajectories',status='GO')
        provenance.event('stage_start',name='dataset_save_and_fit_references')
        budget.check();torch.save(dict(schema='rc3-r04-m1-restored268-v1',
            data_sha256=binding['data_sha256'],restored_input_sha256=digest(HOME/'RESTORED_INPUT_INDEX.json'),episodes=episodes),HOME/'M1_RESTORED_DATASET.pt')
        fit=[e for e in episodes if e['split']=='fit']
        references=old_critic.fit_references(fit)
        if set(references['parent_ids'])!=set(registration['fit_physical_parents']): raise ValueError('Fit reference parent identity differs')
        write_new(HOME/'FIT_ONLY_REFERENCES.json',references)
        provenance.event('stage_end',name='dataset_save_and_fit_references',status='GO')
        identity=dict(source_sha256=digest(HOME/'SOURCE_BINDING.json'),config_sha256=binding['protocol_sha256'],
            data_sha256=binding['data_sha256'],schedule_sha256=binding['schedule_sha256'],protocol_sha256=binding['protocol_sha256'],
            authorization_sha256=digest(HOME/'STAGE_AUTHORIZATION.json'),restored_input_sha256=digest(HOME/'RESTORED_INPUT_INDEX.json'))
        models=critic.make_matched_models();optimizers={name:torch.optim.Adam(model.parameters(),**critic.ADAM) for name,model in models.items()}
        initial={name:critic.tensor_digest(model.state_dict()) for name,model in models.items()}
        if initial['C0']!=initial['C1']: raise ValueError('Unmatched initial tensors')
        write_new(HOME/'INITIALIZATION.json',dict(seed=11622026,tensor_sha256=initial,
            physical_outputs_differ=True,target_informed_initialization=False,
            description='Same fresh raw net tensors; C1 physical components are n times C0 for same state atstep0; not pure gradient-scale causal identification'))
        (HOME/'models').mkdir(exist_ok=False)
        all_metrics=[];interval_records=[];endpoint_records=[];roundtrips=[]
        prediction_file=gzip.open(HOME/'M1_STATE_PREDICTIONS.csv.gz','wt',encoding='utf-8',newline='',compresslevel=1)
        logs.append(prediction_file);prediction_writer=None
        training_file=(HOME/'TRAINING_LOG.csv').open('x',encoding='utf-8',newline='');logs.append(training_file)
        training_writer=csv.DictWriter(training_file,fieldnames=['configuration','step','batch_trajectory_ids','loss','grad_norm',
            'parameter_tensor_sha256','output_semantics','reduction','cumulative_optimizer_attempts','cumulative_wall_s'])
        training_writer.writeheader()
        def endpoint(step):
            nonlocal prediction_writer
            budget.admit('checkpoint_evaluation')
            provenance.event('stage_start',name='endpoint_'+str(step))
            rows=[]
            for name in ('C0','C1'):
                budget.check();model=models[name];optimizer=optimizers[name]
                snapshot=dict(budget.data['counts'])
                package=critic.make_checkpoint(model,optimizer,identity=identity,cursor=step,resources=snapshot)
                path=HOME/'models'/f'{name}_step{step:04d}.checkpoint.pt'
                torch.save(package,path);budget.charge('checkpoint_saves')
                loaded=torch.load(path,map_location='cpu',weights_only=False);budget.charge('checkpoint_loads')
                if snapshot!=budget.data['counts']:
                    # File I/O counters advance, but the exact scientific resource snapshot stays bound.
                    if snapshot['critic_optimizer_attempts']!=budget.data['counts']['critic_optimizer_attempts']:
                        raise ValueError('Consumed optimizer budget changed while checkpoint was saved')
                before=critic.predict_episode(model,fit[0])
                restored_model=critic.MatchedCritic(name);restored_optimizer=torch.optim.Adam(restored_model.parameters(),**critic.ADAM)
                answer=critic.restore_checkpoint(loaded,restored_model,restored_optimizer,expected_identity=identity,
                    expected_cursor=step,expected_resources=snapshot)
                if critic.tree_digest(critic.capture_rng())!=critic.tree_digest(package['rng']): raise ValueError('RNG restore mismatch')
                after=critic.predict_episode(restored_model,fit[0])
                if any(not torch.equal(before[k],after[k]) for k in before): raise ValueError('Saved Critic predictions changed')
                export_path=HOME/'models'/f'{name}_step{step:04d}.export.pt'
                torch.save(critic.make_export(restored_model,identity=identity,cursor=step),export_path)
                export=critic.load_export(torch.load(export_path,map_location='cpu',weights_only=False),expected_variant=name,
                    expected_identity=identity,expected_cursor=step)
                exported=critic.predict_episode(export,fit[0])
                if any(not torch.equal(after[k],exported[k]) for k in after): raise ValueError('Critic export predictions changed')
                roundtrips.append(dict(configuration=name,endpoint=step,checkpoint_sha256=digest(path),
                    export_sha256=digest(export_path),model_adam_rng_cursor_resource_exact=True,predictions_bitwise_equal=True,
                    loaded_as_next_step_state=step<256,scope='Critic-only; not Actor candidate rollback/G2'))
                models[name]=restored_model;optimizers[name]=restored_optimizer
                rows.extend(metrics.evaluate(restored_model,episodes,step,references))
            if prediction_writer is None:
                prediction_writer=csv.DictWriter(prediction_file,fieldnames=list(rows[0]));prediction_writer.writeheader()
            prediction_writer.writerows(rows);prediction_file.flush()
            summary=metrics.summarize(rows);all_metrics.extend(summary)
            intervals=metrics.paired_intervals(rows,step);interval_records.append(intervals)
            decision=metrics.gate(summary,intervals,step,registration);endpoint_records.append(decision)
            write_new(HOME/f'GATE_STEP_{step:04d}.json',decision)
            budget.emit('endpoint',step=step,status=decision['status'],ratios=decision['current_ratios'],midpoint_stop=decision['midpoint_stop_triggered'])
            print(json.dumps(dict(endpoint=step,status=decision['status'],C1_over_conditional=decision['current_ratios'],midpoint_stop=decision['midpoint_stop_triggered'])),flush=True)
            provenance.event('stage_end',name='endpoint_'+str(step),status='GO')
            return decision
        decision=endpoint(0);by_id={e['trajectory_id']:e for e in episodes}
        provenance.event('stage_start',name='matched_fit_1_128')
        for step in range(1,257):
            budget.admit('paired_critic_batch')
            batch_ids=schedule[step-1];batch=[by_id[r] for r in batch_ids]
            if len({e['parent_id'] for e in batch})!=8 or any(e['split']!='fit' for e in batch): raise ValueError('Batch leaked or lost parent balance')
            for name in ('C0','C1'):
                budget.charge('critic_optimizer_attempts',configuration=name,step=step)
                model,optimizer=models[name],optimizers[name];optimizer.zero_grad(set_to_none=True)
                loss=critic.matched_loss(model,batch)
                if not bool(torch.isfinite(loss)): raise ValueError('Nonfinite Critic loss')
                loss.backward()
                if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()): raise ValueError('Nonfinite/missing Critic gradient')
                grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                admitted_step.update(optimizer=optimizer,configuration=name);optimizer.step()
                if any(not bool(torch.isfinite(p).all()) for p in model.parameters()): raise ValueError('Nonfinite updated Critic')
                training_writer.writerow(dict(configuration=name,step=step,batch_trajectory_ids=json.dumps(batch_ids),loss=float(loss.detach()),
                    grad_norm=float(grad_norm),parameter_tensor_sha256=critic.tensor_digest(model.state_dict()),
                    output_semantics=critic.SEMANTICS[name],reduction=critic.REDUCTION,
                    cumulative_optimizer_attempts=budget.data['counts']['critic_optimizer_attempts'],cumulative_wall_s=budget.elapsed()))
                training_file.flush();os.fsync(training_file.fileno())
            if step%32==0: print(f'Completed matched C0/C1 step {step}; cumulative attempts {budget.data["counts"]["critic_optimizer_attempts"]}',flush=True)
            if step in (128,256):
                provenance.event('stage_end',name='matched_fit_1_128' if step==128 else 'matched_fit_129_256',status='GO')
                decision=endpoint(step)
                if decision['midpoint_stop_triggered']: break
                if step==128:provenance.event('stage_start',name='matched_fit_129_256')
        for f in logs:f.close()
        logs=[]
        provenance.event('stage_start',name='save_summaries')
        metrics.write_csv(HOME/'M1_METRICS.csv',all_metrics)
        write_new(HOME/'PAIRED_PARENT_INTERVALS.json',dict(endpoints=interval_records))
        write_new(HOME/'CHECKPOINT_INTERFACE_RESULTS.json',dict(status='GO',roundtrips=roundtrips,
            resume_scope='Each actual endpoint loaded into fresh same-role Critic+Adam and continued once; no duplicated update or independent full continuous-run equivalence claim; Actor/G2 not tested'))
        write_new(HOME/'M1_GATE_DECISION.json',decision)
        budget.persist()
        provenance.event('stage_end',name='save_summaries',status='GO')
        budget.admit('independent_numerical_review')
        provenance.event('stage_start',name='independent_numerical_review')
        t=time.perf_counter();review=independent_review.recheck(HOME,heartbeat=budget.check,trace=provenance.event)
        review['wall_s']=time.perf_counter()-t
        write_new(HOME/'INDEPENDENT_NUMERICAL_RECHECK.json',review)
        if review['status']!='PASS': raise RuntimeError('Independent new M1 numerical review failed')
        provenance.event('stage_end',name='independent_numerical_review',status='GO')
        provenance.check_loaded_modules()
        write_new(HOME/'ACTUAL_DEPENDENCY_VERIFICATION.json',provenance.summary())
        budget.check();budget.close('COMPLETE_'+decision['status'])
        print(json.dumps(dict(status=budget.data['status'],endpoint=decision['endpoint'],counts=budget.data['counts'],wall_s=budget.data['wall_s'])),flush=True)
    except BaseException as exc:
        for f in logs:
            try:f.close()
            except Exception:pass
        if budget.tracer is not None:budget.tracer.event('worker_failure',error=str(exc))
        budget.close('FAILED_PRESERVED',exc)
        write_new(HOME/'FAILURE.json',dict(error=str(exc),traceback=traceback.format_exc(),attempts=budget.data['counts'],
            next_action='No automatic extra attempts/configuration/data change; consumed budget and original evidence retained'))
        raise

if __name__=='__main__':main()
