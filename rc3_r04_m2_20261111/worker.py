"""Single bounded real-Torch worker. Exactly CG128 + CB128; no planners."""
import time
START_M, START_W = time.monotonic(), time.time()
import copy
import csv
import io
import json
import os
import platform
import random
import subprocess
import traceback
from .common import HOME, ROOT, OLD, RUN_ID, Ledger, put, read, sha

def main():
    ledger = Ledger(START_M, START_W)
    trace = None
    try:
        from rc3_r04_m1_real_20261111.provenance_use import ScientificTracedProvenance, verify_real_environment
        binding = read(HOME/'SOURCE_BINDING.json')
        trace = ScientificTracedProvenance(ROOT, HOME, binding, ledger.check, RUN_ID)
        trace.install_import_guard()
        for name in ('SOURCE_BINDING.json', 'M2_PROTOCOL.json'):
            if name == 'M2_PROTOCOL.json': assert sha(HOME/name) == binding['protocol_sha256']
        auth, review = read(HOME/'STAGE_AUTHORIZATION.json'), read(HOME/'IMPLEMENTATION_REVIEW.json')
        assert auth['authorized'] and review['status'] == 'GO'
        assert auth['source_binding_sha256'] == review['source_binding_sha256'] == sha(HOME/'SOURCE_BINDING.json')
        ledger.event('preflight_start')
        with trace.stage('real_imports_and_source_verification'):
            import numpy as np
            import torch
            from rc3_r04_m1_20261111 import critic as units
            from . import model, analysis, audit
            actual = verify_real_environment(trace, torch, dict(model=model, units=units, analysis=analysis, audit=audit))
            trace.check_loaded_modules()
            torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
            environment = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                device='cpu', threads=torch.get_num_threads(), deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
            assert environment == binding['environment']
            receipt_path = ROOT/binding['accepted_engineering_receipt']['path']
            assert trace.hash_file(receipt_path) == binding['accepted_engineering_receipt']['sha256']
            assert read(receipt_path)['status'] == 'GO'
            put(HOME/'REAL_PREFLIGHT.json', dict(status='GO', actual_environment=environment, real_Torch=actual,
                accepted_supervisor_receipt=binding['accepted_engineering_receipt'], historical_supervisor_tests_rerun=False,
                code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
        # No optimizer may step unless it is one of the registered objects and pre-debited.
        adam_step = torch.optim.Adam.step
        admission = {'optimizer': None}
        def protected_step(self, *args, **kwargs):
            if self is not admission['optimizer']: raise RuntimeError('Unregistered/Actor optimizer forbidden')
            result = adam_step(self, *args, **kwargs)
            ledger.counts['critic_optimizer_completed'] += 1; admission['optimizer'] = None
            return result
        torch.optim.Adam.step = protected_step
        with trace.stage('zero_update_link_and_runtime_tests'):
            put(HOME/'ENGINEERING_TEST_RESULTS.json', model.engineering_tests())
        with trace.stage('exact_restored268_load_no_navigation_or_label_generation'):
            identities = binding['inputs']
            registration = json.loads(trace.read_bytes(ROOT/identities['M1_DATA_AND_SOURCE_IDENTITY.json']['path']))
            restored_index = json.loads(trace.read_bytes(ROOT/identities['RESTORED_INPUT_INDEX.json']['path']))
            old_preflight = json.loads(trace.read_bytes(ROOT/identities['REAL_PREFLIGHT.json']['path']))
            assert restored_index['base_tensor_sha256'] == old_preflight['base_tensor_sha256']
            assert old_preflight['source_binding_sha256'] == binding['prior_binding_sha256']
            for p, expected in binding['artifact_sha256'].items():
                if p.endswith('.pt') and p not in (identities['M1_RESTORED_DATASET.pt']['path'], identities['models/C1_step0128.export.pt']['path']):
                    assert trace.hash_file(ROOT/p) == expected
            data_bytes = trace.read_bytes(ROOT/identities['M1_RESTORED_DATASET.pt']['path'])
            dataset = torch.load(io.BytesIO(data_bytes), map_location='cpu', weights_only=False)
            ledger.counts['dataset_loads'] += 1
            assert dataset['schema'] == 'rc3-r04-m1-restored268-v1'
            assert dataset['data_sha256'] == identities['M1_DATA_AND_SOURCE_IDENTITY.json']['sha256']
            assert dataset['restored_input_sha256'] == identities['RESTORED_INPUT_INDEX.json']['sha256']
            episodes = dataset['episodes']; assert len(episodes) == 116
            registered = {e['trajectory_id']: e for e in registration['trajectories']}
            indexed = {e['trajectory_id']: e for e in restored_index['trajectories']}
            assert len(registered) == len(indexed) == len(episodes)
            fit_parents, held_parents = set(registration['fit_physical_parents']), set(registration['holdout_physical_parents'])
            assert len(fit_parents) == 33 and len(held_parents) == 16 and not fit_parents & held_parents
            checked = []
            for e in episodes:
                r, i = registered[e['trajectory_id']], indexed[e['trajectory_id']]
                assert e['parent_id'] == r['parent_id'] == i['parent_id'] and e['split'] == r['split'] == i['split']
                assert e['window_key'] == i['window'] == f"{r['width']}x{r['height']}"
                assert e['stage'] == i['source_stage'] == r['phase']
                assert (e['parent_id'] in fit_parents) == (e['split'] == 'fit')
                assert e['parent_id'] in fit_parents | held_parents
                assert e['v'].ndim == 2 and e['v'].shape[1] == 268 and len(e['v']) == i['states']
                assert not e['v'].requires_grad and e['v'].dtype == torch.float32
                assert units.tensor_digest({'v': e['v']}) == i['feature_tensor_sha256']
                assert e['task_count'] == i['task_count']
                assert e['source'] == i['request_source']
                assert e['source']['path'] == r['request_path'] and e['source']['sha256'] == r['request_sha256']
                units.episode_labels(e)  # Validate saved accounting only; never generate labels.
                checked.append(dict(trajectory_id=e['trajectory_id'], parent_id=e['parent_id'], split=e['split'],
                    window=e['window_key'], states=len(e['v']), feature_tensor_sha256=i['feature_tensor_sha256']))
            assert {e['parent_id'] for e in episodes if e['split']=='fit'} == fit_parents
            assert {e['parent_id'] for e in episodes if e['split']=='validation'} == held_parents
            put(HOME/'DATA_REUSE_VERIFICATION.json', dict(status='PASS', dataset_identity=identities['M1_RESTORED_DATASET.pt'],
                data_sha256=dataset['data_sha256'], restored_index_sha256=dataset['restored_input_sha256'],
                original_restoration_source_binding_sha256=binding['prior_binding_sha256'],
                frozen_base_tensor_sha256=restored_index['base_tensor_sha256'], original_base_artifact_hashes_verified=True,
                fit_parents=33, exposed_holdout_parents=16, episodes=checked, state_count=sum(len(e['v']) for e in episodes),
                trajectories_replayed=0, navigation_requests=0, generated_labels=0, base_model_loads=0))
        with trace.stage('fit_only_anchor_table_and_frozen_C1_reference'):
            fit = [e for e in episodes if e['split']=='fit']
            table = model.fit_table(fit)
            assert set(table['parent_ids']) == fit_parents
            old_refs = json.loads(trace.read_bytes(ROOT/identities['FIT_ONLY_REFERENCES.json']['path']))
            for key, reference in [('global', old_refs['global_constant'])]+list(old_refs['conditional_constants'].items()):
                actual_components = table['global'] if key == 'global' else table['conditional'][key]
                expected = [max(model.FLOOR, reference['values'][n]) if j<2 else reference['values'][n] for j,n in enumerate(model.NAMES)]
                assert actual_components == expected
            put(HOME/'FIT_ONLY_COMPONENT_ANCHORS.json', table)
            schedule = json.loads(trace.read_bytes(ROOT/identities['BATCH_SCHEDULE.json']['path']))['batches'][:128]
            old_export = torch.load(io.BytesIO(trace.read_bytes(ROOT/identities['models/C1_step0128.export.pt']['path'])), map_location='cpu', weights_only=False)
            expected_identity = dict(source_sha256=binding['prior_binding_sha256'],
                config_sha256=identities['M1_EXECUTION_PROTOCOL.json']['sha256'], data_sha256=identities['M1_DATA_AND_SOURCE_IDENTITY.json']['sha256'],
                schedule_sha256=identities['BATCH_SCHEDULE.json']['sha256'], protocol_sha256=identities['M1_EXECUTION_PROTOCOL.json']['sha256'],
                authorization_sha256=identities['STAGE_AUTHORIZATION.json']['sha256'],
                restored_input_sha256=identities['RESTORED_INPUT_INDEX.json']['sha256'])
            c1 = units.load_export(old_export, expected_variant='C1', expected_identity=expected_identity, expected_cursor=128)
            c1.eval().requires_grad_(False)
            put(HOME/'C1_REFERENCE_IDENTITY.json', dict(status='PASS', endpoint=128, source=identities['models/C1_step0128.export.pt'],
                original_identity=expected_identity, net_tensor_sha256=units.tensor_digest(c1.state_dict()), new_C1_optimizer_attempts=0))
        torch.manual_seed(11622026); np.random.seed(11622026); random.seed(11622026)
        models = model.matched_models(table)
        optimizers = {a: torch.optim.Adam(m.parameters(), **units.ADAM) for a,m in models.items()}
        initial = {a: units.tensor_digest(m.state_dict()) for a,m in models.items()}
        assert initial['CG'] == initial['CB']
        put(HOME/'INITIALIZATION.json', dict(same_net_tensor_sha256=initial, zero_output_layer=True,
            target_informed_net_bias=False, fit_only_external_anchor=True, seed=11622026,
            trainable_parameters={a: sum(p.numel() for p in m.parameters()) for a,m in models.items()}))
        identity = dict(source_binding_sha256=sha(HOME/'SOURCE_BINDING.json'), protocol_sha256=sha(HOME/'M2_PROTOCOL.json'),
            original_dataset_sha256=identities['M1_RESTORED_DATASET.pt']['sha256'],
            data_sha256=identities['M1_DATA_AND_SOURCE_IDENTITY.json']['sha256'], schedule_sha256=identities['BATCH_SCHEDULE.json']['sha256'],
            anchor_table_sha256=sha(HOME/'FIT_ONLY_COMPONENT_ANCHORS.json'))
        (HOME/'models').mkdir()
        roundtrips = []
        def checkpoint(arm, step):
            m, o = models[arm], optimizers[arm]
            package = dict(schema='rc3-r04-m2-training-checkpoint-v1', arm=arm, cursor=step, identity=identity,
                model=copy.deepcopy(m.state_dict()), adam=copy.deepcopy(o.state_dict()), rng=units.capture_rng(),
                table=table, floor=model.FLOOR, resources=ledger.snapshot(), stage_authorization_sha256=sha(HOME/'STAGE_AUTHORIZATION.json'),
                usage='training-only Critic; not Actor/deployment artifact')
            path = HOME/'models'/f'{arm}_step{step:04d}.checkpoint.pt'
            torch.save(package, path)
            loaded = torch.load(path, map_location='cpu', weights_only=False)
            assert units.tree_digest(loaded) == units.tree_digest(package)
            with torch.random.fork_rng(devices=[]): restored = model.AnchorCritic(arm, loaded['table'])
            restored.load_state_dict(loaded['model'])
            restored_opt = torch.optim.Adam(restored.parameters(), **units.ADAM); restored_opt.load_state_dict(loaded['adam'])
            assert units.tree_digest(restored_opt.state_dict()) == units.tree_digest(o.state_dict())
            assert len(restored_opt.state) == (0 if step == 0 else 6)
            if step:
                assert all(int(v['step']) == step for v in restored_opt.state.values())
            units.restore_rng(loaded['rng'])
            assert units.tree_digest(units.capture_rng()) == units.tree_digest(loaded['rng'])
            with torch.no_grad(): assert torch.equal(m(fit[0]['v'], fit[0]['window_key']), restored(fit[0]['v'], fit[0]['window_key']))
            models[arm], optimizers[arm] = restored, restored_opt
            roundtrips.append(dict(arm=arm, endpoint=step, sha256=sha(path), exact_model_adam_rng_cursor_table_resources=True))
        state_index = []
        targets = []
        for e in episodes:
            y = units.episode_labels(e, dtype=torch.float64)
            targets.append(np.column_stack([y[c].numpy() for c in analysis.COMPONENTS]))
            for t in range(len(e['v'])):
                state_index.append(dict(parent_id=e['parent_id'], trajectory_id=e['trajectory_id'], split=e['split'],
                    window=e['window_key'], stage=e['stage'], state_index=t, task_count=e['task_count'], remaining_fraction=float(e['v'][t,256])))
        targets = np.concatenate(targets)
        predictions = {}
        def array(components, n):
            a = components.detach().double().numpy()
            return np.column_stack((a*n, a, -(a[:,0]+.75*a[:,1]-a[:,2])))
        @torch.no_grad()
        def endpoint(step):
            with trace.stage('endpoint_'+str(step)):
                for arm in ('CG','CB'):
                    checkpoint(arm, step)
                    predictions[arm+str(step)] = np.concatenate([array(models[arm](e['v'], e['window_key']),e['task_count']) for e in episodes])
        with trace.stage('zero_update_reference_predictions'):
            for name, conditional in (('B-global0',False),('B-cond0',True)):
                predictions[name] = np.concatenate([array(model.lookup(table,e['window_key'],e['v'],conditional)[0],e['task_count']) for e in episodes])
            predictions['C1_fixed128'] = np.concatenate([np.column_stack([v.numpy() for v in units.predict_episode(c1,e).values()]) for e in episodes])
        endpoint(0)
        by_id = {e['trajectory_id']: e for e in episodes}
        with trace.stage('matched_CG_CB_128_steps'):
            with (HOME/'TRAINING_LOG.csv').open('x', encoding='utf-8', newline='') as f:
                w = csv.DictWriter(f, fieldnames=['arm','step','batch_ids','loss','grad_norm','net_tensor_sha256','optimizer_attempts','wall_s']); w.writeheader()
                for step, batch_ids in enumerate(schedule,1):
                    batch = [by_id[i] for i in batch_ids]
                    assert len(batch) == len({e['parent_id'] for e in batch}) == 8
                    for arm in ('CG','CB'):
                        ledger.admit_optimizer(arm,step)
                        m,o = models[arm],optimizers[arm]; o.zero_grad(set_to_none=True)
                        loss = model.loss(m,batch); assert torch.isfinite(loss)
                        ledger.counts['backward_calls'] += 1; loss.backward()
                        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
                        grad = torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
                        admission['optimizer'] = o; o.step()
                        assert all(torch.isfinite(p).all() for p in m.parameters())
                        w.writerow(dict(arm=arm,step=step,batch_ids=json.dumps(batch_ids),loss=float(loss.detach()),grad_norm=float(grad),
                            net_tensor_sha256=units.tensor_digest(m.state_dict()),optimizer_attempts=ledger.counts['critic_optimizer_attempts'],wall_s=ledger.elapsed()))
                        f.flush(); os.fsync(f.fileno())
                    if step%32 == 0: print(f'Matched CG/CB {step}/128; attempts={ledger.counts["critic_optimizer_attempts"]}',flush=True)
        endpoint(128)
        put(HOME/'CHECKPOINT_INTERFACE_RESULTS.json',dict(status='PASS',roundtrips=roundtrips,
            exact_step0_restored_state_used_for_training=True, no_resume_or_extra_steps=True))
        with trace.stage('new_endpoint_metrics_and_parent_intervals'):
            analysis.save_predictions(state_index,targets,predictions)
            rows, intervals = analysis.metrics(state_index,targets,predictions)
            analysis.write_csv(HOME/'M2_METRICS.csv',rows); analysis.write_csv(HOME/'PAIRED_PARENT_INTERVALS.csv',intervals)
            decision = analysis.decision(rows,intervals); put(HOME/'M2_DECISION.json',decision)
        with trace.stage('independent_new_output_numerical_audit'):
            report = audit.run(ledger.check); put(HOME/'INDEPENDENT_NUMERICAL_RECHECK.json',report)
        trace.check_loaded_modules()
        put(HOME/'ACTUAL_DEPENDENCY_VERIFICATION.json',trace.summary())
        assert ledger.counts['critic_optimizer_attempts'] == ledger.counts['critic_optimizer_completed'] == 256
        assert ledger.arms == dict(CG=128,CB=128)
        ledger.status = 'COMPLETE_'+decision['status']; ledger.event('completed',decision=decision['status'])
        put(HOME/'WORKER_RESOURCE_ACCOUNTING.json',ledger.snapshot())
        print(json.dumps(dict(status=ledger.status,wall_s=ledger.elapsed(),counts=ledger.counts)),flush=True)
    except BaseException as exc:
        ledger.status = 'FAILED_PRESERVED'
        put(HOME/'WORKER_FAILURE.json',dict(error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),resources=ledger.snapshot()))
        put(HOME/'WORKER_RESOURCE_ACCOUNTING.json',ledger.snapshot())
        raise

if __name__ == '__main__': main()
