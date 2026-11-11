"""Real-dependency preflight and continuous M1; frozen science reused unchanged."""
import time
ENTRY=time.perf_counter()
import csv
import gzip
import io
import json
import math
from xml.sax.saxutils import escape
import os
import platform
import random
import subprocess
import traceback
import unittest
from .ledger import Budget,HOME,ROOT,digest,write_new

def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))


class EntryClassificationError(RuntimeError):
    """Known source-classification interface error, never an identity mismatch."""


def restart_eligibility(error,phase,resource):
    counts=resource['counts']
    before_science=(counts.get('critic_optimizer_attempts',0)==0 and
                    counts.get('actor_optimizer_attempts',0)==0 and
                    not resource.get('step0_started',False))
    message=str(error).lower()
    prohibited=any(token in message for token in ('sha mismatch','hash mismatch','identity differs',
        'bytes changed','sha256 mismatch','frozen source','sealed','original60','olds3','forbidden',
        'corrupt','nonfinite','label','parent','trajectory'))
    category=None
    if isinstance(error,(ImportError,ModuleNotFoundError)):
        category='import_interface'
    elif phase=='source_classification' and (isinstance(error,(EntryClassificationError,AttributeError,TypeError)) or
            isinstance(error,ValueError) and any(token in message for token in
            ('virtual namespace falsely classified','missing valid source spec/loader','no modulespec',
             'unsupported source loader','sourcefileloader path disagree','source specification disagree','no file location'))):
        category='source_classification_interface'
    elif phase in ('environment_record','real_preflight_record') and isinstance(error,OSError):
        category='recording_interface'
    return dict(eligible=bool(before_science and category is not None and not prohibited),
        before_any_optimizer_and_step0=before_science,category=category,
        prohibited_identity_or_science_failure=prohibited,
        owner_review_required=True,maximum_task_worker_launches=3,
        consumed_time_and_reads_must_carry_forward=True)


def draw_saved_curves(home,metrics,budget,provenance):
    """Plot already computed log values only, inside the execution clock."""
    start=time.perf_counter();budget.check()
    provenance.event('stage_start',name='saved_result_curves')
    with (home/'TRAINING_LOG.csv').open(encoding='utf-8',newline='') as stream:
        rows=list(csv.DictReader(stream))
    panels=[('Matched training component loss',
        {role:[(int(r['step']),float(r['loss'])) for r in rows if r['configuration']==role]
         for role in ('C0','C1')})]
    for aggregation in ('state_pooled','parent_equal'):
        panels.append(('Held-out value MSE: '+aggregation,
            {role:[(int(r['endpoint']),float(r['mse'])) for r in metrics
                if r['configuration']==role and r['split']=='validation' and r['component']=='value'
                and r['window']=='ALL' and r['source_stage']=='ALL' and r['aggregation']==aggregation]
             for role in ('C0','C1')}))
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1050" height="375" viewBox="0 0 1050 375">',
         '<rect width="1050" height="375" fill="white"/>',
         '<g font-family="sans-serif" font-size="12" fill="#202020">',
         '<text x="20" y="23">R04-M1 saved results; calibration error is not route benefit</text>']
    for index,(title,series) in enumerate(panels):
        budget.check();left=35+index*350;top=64;width=290;height=245
        points=[point for values in series.values() for point in values]
        if not points or any(not math.isfinite(y) for _,y in points):
            raise ValueError('Saved curve requires finite existing values')
        xmax=max(1,max(x for x,_ in points));ymax=max(y for _,y in points)
        ymax=max(ymax,1e-12)
        out.extend([f'<text x="{left}" y="{top-13}">{escape(title)}</text>',
            f'<path d="M {left} {top} V {top+height} H {left+width}" fill="none" stroke="#555"/>',
            f'<text x="{left}" y="{top+height+20}">0</text>',
            f'<text x="{left+width-35}" y="{top+height+20}">{xmax}</text>',
            f'<text x="{left}" y="{top+14}">max {ymax:.5g}</text>'])
        for role,color in (('C0','#b34d17'),('C1','#1760ab')):
            coords=[(left+x/xmax*width,top+height-y/ymax*height) for x,y in series[role]]
            encoded=' '.join(f'{x:.3f},{y:.3f}' for x,y in coords)
            out.append(f'<polyline points="{encoded}" fill="none" stroke="{color}" stroke-width="1.7"/>')
            if len(coords)<5:
                for x,y in coords:out.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="3" fill="{color}"/>')
            out.append(f'<text x="{left+(20 if role=="C0" else 85)}" y="350" fill="{color}">{role}</text>')
    out.extend(['</g>','</svg>'])
    with (home/'M1_CURVES.svg').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write('\n'.join(out)+'\n');stream.flush();os.fsync(stream.fileno())
    budget.check()
    provenance.event('stage_end',name='saved_result_curves',status='GO',seconds=time.perf_counter()-start,
        source='Existing TRAINING_LOG.csv and in-memory frozen-metrics summaries only')

def main():
    budget=Budget('real_environment_and_M1',ENTRY)
    logs=[]
    phase='startup_controls'
    preflight={}
    try:
        from .provenance_use import ScientificTracedProvenance, verify_real_environment
        binding=read(HOME/'SOURCE_BINDING.json')
        provenance=ScientificTracedProvenance(ROOT,HOME,binding,budget.check,budget.run_id)
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
        engineering_identity=binding['accepted_engineering_receipt']
        engineering_path=ROOT/engineering_identity['path']
        if digest(engineering_path)!=engineering_identity['sha256']:
            raise ValueError('Accepted engineering receipt SHA mismatch')
        engineering=read(engineering_path)
        if engineering['status']!='GO':raise ValueError('Previously accepted timeout engineering did not pass')
        preflight['accepted_engineering_receipt']=engineering_identity
        preflight['timeout_tests_rerun']=False
        phase='heavy_imports'
        with provenance.stage('heavy_imports'):
            import numpy as np
            import torch
            from rc3_r04_m1_20261111 import critic,data,metrics,test_critic,restricted_inputs
            from . import review_adapter as independent_review
            from rc3_critic_diagnostics_20261111 import critic as old_critic
            from rc3_conservative_refinement import saved_inputs as si,runtime
        phase='source_classification'
        runtime_classification=verify_real_environment(provenance,torch,
            {'critic':critic,'data':data,'runtime':runtime})
        checked_modules=provenance.check_loaded_modules()
        preflight.update(runtime_classification=runtime_classification,
            checked_project_module_count=len(checked_modules),used_for_scientific_attempt=True)
        phase='locked_source_validation'
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
        phase='environment_record'
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
        phase='zero_update_critic_interface_tests'
        provenance.event('stage_start',name=phase)
        suite=unittest.defaultTestLoader.loadTestsFromModule(test_critic)
        stream=io.StringIO();t=time.perf_counter()
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        write_new(HOME/'UNIT_TEST_RESULTS.json',dict(status='GO' if result.wasSuccessful() else 'FAIL',
            tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),wall_s=time.perf_counter()-t,
            optimizer_attempts=0,output=stream.getvalue()))
        if not result.wasSuccessful() or result.testsRun!=14:
            raise RuntimeError('All 14 registered engineering unit tests must pass')
        preflight['critic_interface_tests']=dict(status='GO',tests_run=result.testsRun,optimizer_attempts=0)
        phase='verified_input_adapters'
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
        phase='saved_input_loader'
        provenance.event('stage_start',name=phase)
        inputs=restricted_inputs.make_inputs(registration)
        def forbidden(*args,**kwargs):
            budget.emit('forbidden_path_attempt');raise RuntimeError('Planning/action/env.step/teacher/navigation/H forbidden')
        inputs.teacher=forbidden
        runtime.rollout=runtime.plan_request=runtime.sl_route=forbidden
        from rc3_global_local.environment import Environment
        from rc3_global_local import terminal
        Environment.step=forbidden;terminal.finish=terminal.export=forbidden
        provenance.event('stage_end',name='saved_input_loader',status='GO')
        phase='frozen_SL_loading'
        provenance.event('stage_start',name=phase)
        budget.charge('model_loads',role='frozen_exact_SL256')
        encoder=runtime.load_actor(True,11622026);encoder.eval().requires_grad_(False)
        base_before=critic.tensor_digest(encoder.encoder.state_dict())
        provenance.event('stage_end',name='frozen_SL_loading',status='GO')
        phase='restore_116_saved_trajectories'
        provenance.event('stage_start',name=phase)
        episodes=[];restored=[];t=time.perf_counter()
        for i,row in enumerate(registration['trajectories']):
            budget.admit('restore_saved_trajectory')
            provenance.event('trajectory_restore_start',trajectory_id=row['trajectory_id'])
            budget.charge('saved_trajectories',trajectory_id=row['trajectory_id'])
            episode=data.load_episode(row,encoder,inputs,budget.charge)
            labels=critic.episode_labels(episode)
            if episode['v'].ndim!=2 or episode['v'].shape[1]!=268 or not bool(torch.isfinite(episode['v']).all()):
                raise ValueError('Actual fixed-trajectory features are not finite T x 268')
            if len(episode['v'])!=len(labels['value']):
                raise ValueError('Actual saved feature and complete label state counts differ')
            if episode['trajectory_id']!=row['trajectory_id'] or episode['parent_id']!=row['parent_id']:
                raise ValueError('Locked trajectory or physical parent identity differs')
            episodes.append(episode)
            provenance.event('trajectory_restore_end',trajectory_id=row['trajectory_id'],status='GO',states=len(episode['v']))
            restored.append(dict(trajectory_id=episode['trajectory_id'],parent_id=episode['parent_id'],split=episode['split'],
                window=episode['window_key'],source_stage=episode['stage'],task_count=episode['task_count'],states=len(episode['v']),
                feature_tensor_sha256=critic.tensor_digest({'v':episode['v']}),request_source=episode['source']))
            if i==0:
                preflight.update(status='GO',first_fixed_trajectory=restored[0],
                    feature_shape=list(episode['v'].shape),
                    label_shapes={k:list(v.shape) for k,v in labels.items()},
                    label_tensor_sha256=critic.tensor_digest(labels),
                    units=dict(raw_L='distance',raw_K='stops',raw_H='distance-equivalent H credit',
                        normalized_L='distance/initial_task',normalized_K='stops/initial_task',
                        normalized_H='distance-equivalent H credit/initial_task',
                        value='negative distance-equivalent/initial_task'),
                    RTG_identity=f'-(L/n + {critic.PENALTY}*K/n - H/n); original label validator tolerance unchanged',
                    task_count=episode['task_count'],base_tensor_sha256=base_before,
                    real_torch_version=torch.__version__,real_numpy_version=np.__version__,
                    same_supervised_process_pid=os.getpid(),new_labels_or_actions=0,
                    continued_restore_expected_count=115,source_binding_sha256=digest(HOME/'SOURCE_BINDING.json'))
                phase='real_preflight_record'
                write_new(HOME/'REAL_PREFLIGHT.json',preflight)
                phase='restore_116_saved_trajectories'
                print('Real Torch/classification/14 interfaces/exact SL256/first fixed trajectory preflight GO; continuing locked remaining 115',flush=True)
            if (i+1)%16==0: print(f'Restored {i+1}/{len(registration["trajectories"])} exact saved trajectories',flush=True)
        if (critic.tensor_digest(encoder.encoder.state_dict())!=base_before or
            any(p.requires_grad or p.grad is not None for p in encoder.encoder.parameters())):
            raise ValueError('Frozen SL encoder changed')
        write_new(HOME/'RESTORED_INPUT_INDEX.json',dict(trajectories=restored,base_tensor_sha256=base_before,
            source='saved prefixes only; 268 reconstruction, no new labels/decisions',wall_s=time.perf_counter()-t,loader_counts=inputs.counts))
        provenance.event('stage_end',name='restore_116_saved_trajectories',status='GO')
        phase='dataset_save_and_fit_references'
        provenance.event('stage_start',name=phase)
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
        phase='formal_initialization'
        # Reset after all preflight/restoration; no diagnostic RNG may alter the formal start.
        torch.manual_seed(11622026);np.random.seed(11622026);random.seed(11622026)
        formal_rng_sha256=critic.tree_digest(critic.capture_rng())
        models=critic.make_matched_models();optimizers={name:torch.optim.Adam(model.parameters(),**critic.ADAM) for name,model in models.items()}
        initial={name:critic.tensor_digest(model.state_dict()) for name,model in models.items()}
        if initial['C0']!=initial['C1']: raise ValueError('Unmatched initial tensors')
        write_new(HOME/'INITIALIZATION.json',dict(seed=11622026,tensor_sha256=initial,
            explicit_rng_reset_after_restoration=True,pre_initialization_rng_sha256=formal_rng_sha256,
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
            nonlocal prediction_writer,phase
            phase='endpoint_'+str(step)
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
        # Durable boundary before any scientific endpoint prediction/checkpoint work.
        budget.data['step0_started']=True
        budget.persist()
        budget.emit('step0_started',same_process_pid=os.getpid())
        decision=endpoint(0);by_id={e['trajectory_id']:e for e in episodes}
        phase='matched_fit_1_128'
        provenance.event('stage_start',name=phase)
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
                if step==128:
                    phase='matched_fit_129_256'
                    provenance.event('stage_start',name=phase)
        for f in logs:f.close()
        logs=[]
        phase='save_summaries'
        provenance.event('stage_start',name=phase)
        metrics.write_csv(HOME/'M1_METRICS.csv',all_metrics)
        write_new(HOME/'PAIRED_PARENT_INTERVALS.json',dict(endpoints=interval_records))
        write_new(HOME/'CHECKPOINT_INTERFACE_RESULTS.json',dict(status='GO',roundtrips=roundtrips,
            resume_scope='Each actual endpoint loaded into fresh same-role Critic+Adam and continued once; no duplicated update or independent full continuous-run equivalence claim; Actor/G2 not tested'))
        write_new(HOME/'M1_GATE_DECISION.json',decision)
        draw_saved_curves(HOME,all_metrics,budget,provenance)
        budget.persist()
        provenance.event('stage_end',name='save_summaries',status='GO')
        budget.admit('independent_numerical_review')
        phase='independent_numerical_review'
        provenance.event('stage_start',name=phase)
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
        cleanup_errors=[]
        for f in logs:
            try:f.close()
            except Exception as close_error:cleanup_errors.append(repr(close_error))
        eligibility=restart_eligibility(exc,phase,budget.data)
        if budget.tracer is not None:budget.tracer.event('worker_failure',error=str(exc),failure_phase=phase,restart_eligibility=eligibility)
        budget.close('FAILED_PRESERVED',exc)
        write_new(HOME/'FAILURE.json',dict(error=str(exc),exception_type=type(exc).__name__,
            traceback=traceback.format_exc(),attempts=budget.data['counts'],failure_phase=phase,
            step0_started=bool(budget.data.get('step0_started')),restart_eligibility=eligibility,
            cleanup_errors=cleanup_errors,
            next_action='Owner may repair an eligible entry only under the same cumulative allowance; no automatic retry by this worker'))
        raise

if __name__=='__main__':main()
