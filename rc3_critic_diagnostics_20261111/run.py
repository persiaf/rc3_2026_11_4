"""One registered offline run. No Actor optimizer or terminal replay exists here."""
from __future__ import annotations

import time
ENTRY = time.perf_counter()
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import traceback

from .ledger import Budget, digest, write_new

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(__file__).resolve().parent


class Table:
    def __init__(self, path):
        self.path = Path(path)
        if self.path.exists():
            raise RuntimeError('Immutable table already exists: ' + str(path))
        self.stream = (gzip.open(self.path, 'wt', encoding='utf-8', newline='', compresslevel=1)
                       if str(path).endswith('.gz') else self.path.open('x', encoding='utf-8', newline=''))
        self.writer = None
        self.rows = 0

    def add(self, rows):
        for row in rows:
            if self.writer is None:
                self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
                self.writer.writeheader()
            self.writer.writerow(row)
            self.rows += 1

    def close(self):
        self.stream.close()


def main():
    registration = json.loads((HOME/'PREREGISTRATION.json').read_text(encoding='utf-8'))
    budget = Budget(HOME, registration, started=ENTRY)
    tables = []
    status, error = 'FAILED', None
    try:
        import numpy as np
        import torch
        from rc3_conservative_refinement import saved_inputs as si, runtime
        from rc3_conservative_refinement.model import SmallCorrection
        from rc3_conservative_refinement.trainer import tensor_digest
        from . import actor as rank, critic, data

        def checked_hash(path):
            path = Path(path)
            budget.charge('source_reads',path=str(path.relative_to(ROOT)),role='hash_file_read')
            budget.charge('source_hashes', path=str(path.relative_to(ROOT)))
            with budget.timed('source_hash'):
                return digest(path)

        for name, expected in registration['source_sha256'].items():
            if checked_hash(ROOT/name) != expected:
                raise ValueError('Registered source bytes changed: '+name)
        for name, expected in registration['artifact_sha256'].items():
            if checked_hash(ROOT/name) != expected:
                raise ValueError('Registered evidence bytes changed: '+name)
        for name, expected in registration['registration_files_sha256'].items():
            if checked_hash(HOME/name) != expected:
                raise ValueError('Registered configuration changed: '+name)
        budget.emit(event='source_verified', code_commit=subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(registration['seed'])
        np.random.seed(registration['seed'])
        import platform
        actual=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                    platform=platform.platform(),device='cpu',threads=torch.get_num_threads(),
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
        for key in ('python','torch','numpy','device','threads','deterministic_algorithms'):
            if actual[key]!=registration['environment'][key]:
                raise ValueError('Registered environment differs: '+key)
        write_new(HOME/'ACTUAL_ENVIRONMENT.json',actual)

        old_read, old_digest, old_np_load = si.read, si.digest, np.load
        def audited_read(path):
            budget.charge('source_reads', path=str(Path(path).relative_to(ROOT)))
            with budget.timed('source_read'):
                return old_read(path)
        def audited_np_load(path, *args, **kwargs):
            budget.charge('source_reads', path=str(Path(path).relative_to(ROOT)))
            with budget.timed('source_read'):
                return old_np_load(path, *args, **kwargs)
        si.read, si.digest, np.load = audited_read, checked_hash, audited_np_load
        runtime.file_sha256 = checked_hash
        with budget.timed('saved_input_loader_initialization'):
            inputs = si.SavedInputs()
        # These hard tripwires are additional to the restricted saved-input loader.
        def forbidden(*args, **kwargs):
            budget.emit(event='forbidden_planning_or_H_path_attempt')
            raise RuntimeError('No new task/navigation/teacher/native/H/rollout permitted')
        inputs.teacher = forbidden
        runtime.rollout = runtime.plan_request = runtime.sl_route = forbidden
        from rc3_global_local.environment import Environment
        from rc3_global_local import terminal
        Environment.step = forbidden
        terminal.finish = terminal.export = forbidden

        budget.charge('model_loads', role='frozen_SL_feature_encoder')
        budget.charge('source_reads',path=runtime.SL_PATH)
        with budget.timed('model_loading'):
            encoder = runtime.load_actor(True, registration['seed'])
            encoder.eval().requires_grad_(False)
        base_before = tensor_digest(encoder.encoder.state_dict())
        selection = json.loads((HOME/'TRAJECTORY_SELECTION.json').read_text(encoding='utf-8'))
        batches = json.loads((HOME/'BATCH_SCHEDULE.json').read_text(encoding='utf-8'))['batches']
        checkpoint_cache = {}
        def heads_for(checkpoint):
            name = checkpoint['path']
            if name not in checkpoint_cache:
                if checked_hash(ROOT/name) != checkpoint['sha256']:
                    raise ValueError('Historical checkpoint hash differs')
                budget.charge('model_loads', role='historical_checkpoint',path=name)
                budget.charge('source_reads', path=name)
                with budget.timed('model_loading'):
                    package = torch.load(ROOT/name,map_location='cpu',weights_only=False)
                    heads = {}
                    for role in ('ON','OFF'):
                        state = package['trainers'][role]['correction']
                        module = SmallCorrection(input_dim=206,hidden=64,max_adjustment=.25)
                        module.load_state_dict(state,strict=True)
                        module.eval().requires_grad_(False)
                        heads[role] = module
                    checkpoint_cache[name] = heads
                    del package
            return checkpoint_cache[name]

        actor_table=Table(HOME/'ACTOR_RANK_RESPONSE.csv');tables.append(actor_table)
        candidate_table=Table(HOME/'ACTOR_CANDIDATES.csv.gz');tables.append(candidate_table)
        episodes=[]
        labels=('ON_before','ON_after','OFF_before','OFF_after','ON_after_OFF','OFF_after_ON')
        for i,row in enumerate(selection['trajectories']):
            budget.check()
            ep=data.reconstruct_episode(row,encoder,inputs,budget.charge)
            ep['original_split']=ep['split']
            ep['split']='fit' if ep['split']=='fit' else 'validation'
            ep['window_key']=f"{row['width']}x{row['height']}"
            ep['stage']=row['phase']
            ep['cohort']='saved_online_R03_training_parent'
            ep['candidate_ids']=ep['record']['problem']['candidate_ids_global']
            before,after=heads_for(row['before_checkpoint']),heads_for(row['after_checkpoint'])
            own=row['method'];features=ep['batch']['xON'] if own=='ON' else ep['batch']['xOFF']
            if tensor_digest(before[own].state_dict()) != ep['record']['correction_tensor_sha256']:
                raise ValueError('Collection Actor is not the exact pre-update snapshot')
            if tensor_digest(after[own].state_dict()) != row['after_actor_tensor_sha256']:
                raise ValueError('Post-update Actor differs from the saved guard receipt')
            budget.charge('actor_head_forwards', role='collection_logits_recovery',trajectory_id=ep['trajectory_id'])
            with budget.timed('actor_head_forward'),torch.no_grad():
                ep['old_logits']=before[own](ep['batch']['z0'],features,ep['batch']['mask']).detach()
            ep['ratio_source']='reconstructed_exact_before_checkpoint_checked_selected_saved_probability'
            variants={
                'ON_before':dict(correction=before['ON'],features=ep['batch']['xON']),
                'ON_after':dict(correction=after['ON'],features=ep['batch']['xON']),
                'OFF_before':dict(correction=before['OFF'],features=ep['batch']['xOFF']),
                'OFF_after':dict(correction=after['OFF'],features=ep['batch']['xOFF']),
                'ON_after_OFF':dict(correction=after['ON'],features=ep['batch']['xOFF']),
                'OFF_after_ON':dict(correction=after['OFF'],features=ep['batch']['xON'])}
            identity={k:row[k] for k in ('trajectory_id','pair_id','parent_id','phase','width','height','opportunity','slot','condition')}
            identity.update(split=ep['split'],source_before=own+'_before',source_after=own+'_after')
            state_rows,candidates=rank.diagnose_trajectory(ep,identity=identity,variants=variants,
                source_before=own+'_before',source_after=own+'_after',charge=budget.charge,
                contrasts=[('ON_feature_switch','ON_after','ON_after_OFF'),
                           ('OFF_feature_switch','OFF_after_ON','OFF_after'),
                           ('ON_OFF_weights_fixed_ON','ON_after','OFF_after_ON'),
                           ('ON_OFF_weights_fixed_OFF','ON_after_OFF','OFF_after')])
            for sr in state_rows:
                if sr['saved_collection_logit_abs_difference'] is None or sr['saved_collection_logit_abs_difference']>1e-6:
                    raise ValueError('Recovered full logits fail selected saved logit check')
                if sr['saved_collection_probability_abs_difference'] is None or sr['saved_collection_probability_abs_difference']>2e-6:
                    raise ValueError('Recovered probabilities differ from saved collection')
                if abs(sr['pre_update_ratio']-1)>1e-12:
                    raise ValueError('Pre-update probability ratio differs from one')
            with budget.timed('actor_evidence_write'):
                actor_table.add(state_rows);candidate_table.add(candidates)
            # Calibration needs only the fixed 268-vector and the complete old labels.
            ep['v']=ep['batch']['v'];ep.pop('old_logits');ep.pop('batch')
            episodes.append(ep)
            if (i+1)%16==0: print(f'Reconstructed and diagnosed {i+1}/128 saved trajectories',flush=True)
        actor_table.close();candidate_table.close()
        if tensor_digest(encoder.encoder.state_dict()) != base_before:
            raise RuntimeError('Frozen base changed during diagnosis')
        if inputs.counts['teacher_loads'] or inputs.counts['navigation_solves'] or inputs.counts['task_generations']:
            raise RuntimeError('Saved-input zero-new-request boundary was crossed')
        write_new(HOME/'RECONSTRUCTION_RECEIPT.json',dict(trajectories=len(episodes),
            states=sum(len(e['v']) for e in episodes),actor_state_rows=actor_table.rows,
            legal_candidate_rows=candidate_table.rows,loader_counts=inputs.counts,
            base_sha_before=base_before,base_sha_after=tensor_digest(encoder.encoder.state_dict()),
            checkpoint_files_loaded=len(checkpoint_cache),
            full_old_logits_persisted=False,full_old_logits_recovered=True,
            collection_check_tolerances=dict(selected_logit_atol=1e-6,selected_probability_atol=2e-6),
            new_environment_steps=0,new_H_labels=0))
        del encoder,checkpoint_cache

        fit=[e for e in episodes if e['split']=='fit'];validation=[e for e in episodes if e['split']=='validation']
        if {e['parent_id'] for e in fit}&{e['parent_id'] for e in validation}:
            raise ValueError('Whole physical parent leaked across folds')
        with budget.timed('fit_only_constant_estimation'):
            refs=critic.fit_references(fit)
        write_new(HOME/'FIT_ONLY_REFERENCES.json',refs)
        models=critic.make_matched_models(registration['seed'])
        initial={name:tensor_digest(m.state_dict()) for name,m in models.items()}
        if len(set(initial.values()))!=1: raise ValueError('Initial net tensors differ')
        for name,model in models.items():
            model.register_forward_pre_hook(lambda module,args,label=name: budget.charge('critic_forwards',configuration=label,states=len(args[0])))
        model_dir=HOME/'models';model_dir.mkdir(exist_ok=False)
        torch.save(dict(schema='offline-critic-initial-v1',seed=registration['seed'],initial_tensor_sha=initial,
                        states={name:m.state_dict() for name,m in models.items()}),model_dir/'initial.pt')
        prediction_table=Table(HOME/'CRITIC_STATE_PREDICTIONS.csv.gz');tables.append(prediction_table)
        summary_table=Table(HOME/'CRITIC_MATCHED_CALIBRATION.csv');tables.append(summary_table)
        strata_table=Table(HOME/'CRITIC_STRATIFIED_CALIBRATION.csv');tables.append(strata_table)
        def save_eval(name,model,endpoint):
            for fold,subset in (('fit',fit),('validation',validation)):
                with budget.timed('critic_evaluation'):
                    rows=critic.evaluate_rows(model,subset,name,endpoint,fold,refs)
                    prediction_table.add(rows)
                    summary_table.add(critic.summarize_rows(rows))
                    strata_table.add(critic.summarize_rows(rows,group_by=('configuration','endpoint','split','component','window','source_stage')))
        oldrows,missing=critic.saved_online_prediction_rows(episodes,refs)
        prediction_table.add(oldrows);summary_table.add(critic.summarize_rows(oldrows))
        strata_table.add(critic.summarize_rows(oldrows,group_by=('configuration','endpoint','split','component','window','source_stage')))
        write_new(HOME/'MISSING_SAVED_FIELDS.json',dict(original_full_candidate_logits='not persisted; recovered from exact source hash checked before checkpoint',
            original_206_268_feature_tensors='not persisted; deterministic saved-prefix reconstruction',
            original_critic_predictions_missing=missing))
        del oldrows
        by_id={e['trajectory_id']:e for e in episodes}
        train_table=Table(HOME/'TRAINING_LOG.csv');tables.append(train_table)
        for name,model in models.items():
            save_eval(name,model,0)
        optimizers={name:torch.optim.Adam(model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0.) for name,model in models.items()}
        for step,ids in enumerate(batches,1):
            batch=[by_id[t] for t in ids]
            if any(e['split']!='fit' for e in batch): raise ValueError('Validation state entered an optimizer batch')
            for name,model in models.items():
                budget.check();optimizer=optimizers[name];optimizer.zero_grad(set_to_none=True)
                with budget.timed('critic_fitting_forward_backward'):
                    loss=critic.matched_loss(model,batch)
                    if not torch.isfinite(loss): raise RuntimeError('Nonfinite Critic fitting loss')
                    budget.charge('critic_backwards',configuration=name,step=step)
                    loss.backward()
                    grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
                budget.charge('critic_optimizer_attempts',configuration=name,step=step)
                with budget.timed('critic_optimizer'):
                    optimizer.step()
                if not all(bool(torch.isfinite(p).all()) for p in model.parameters()): raise RuntimeError('Nonfinite fitted Critic')
                train_table.add([dict(configuration=name,step=step,batch_trajectory_ids=json.dumps(ids,separators=(',',':')),
                    loss=float(loss.detach()),grad_norm=grad,gradient_finite=True,optimizer_attempt=True,optimizer_completed=True)])
            if step%64==0: print(f'All C0/C1/C2 completed {step}/256 fixed matched steps',flush=True)
        for name,model in models.items():
            save_eval(name,model,256)
            torch.save(dict(schema='offline-critic-fit-v1',configuration=name,steps=256,
                registration_sha256=budget.registration_sha256,initial_tensor_sha=initial[name],
                final_tensor_sha=tensor_digest(model.state_dict()),model=model.state_dict(),
                optimizer=optimizers[name].state_dict(),torch_rng=torch.get_rng_state()),model_dir/f'{name}_0256.pt')
        for table in tables: table.close()
        from .summaries import summarize as saved_summary
        budget.charge('source_reads',2,role='saved_summary_new_tables')
        with budget.timed('saved_summary_arithmetic'):
            saved_summary(HOME)
        from .independent_recheck import recheck
        budget.charge('numerical_rechecks')
        budget.charge('source_reads',12,role='independent_saved_table_hash_and_parse_reads')
        budget.charge('source_hashes',6,role='independent_saved_table_hashes')
        with budget.timed('independent_numerical_recheck'):
            review=recheck(HOME)
        if not (HOME/'NUMERICAL_RECHECK.json').is_file():
            raise RuntimeError('Independent checker did not preserve its receipt')
        if review['status']!='PASS': raise RuntimeError('Independent saved-only numerical recheck failed')
        for name,expected in registration['source_sha256'].items():
            if checked_hash(ROOT/name)!=expected: raise ValueError('Source changed mid-study: '+name)
        for name,expected in registration['artifact_sha256'].items():
            if checked_hash(ROOT/name)!=expected: raise ValueError('Original evidence changed mid-study: '+name)
        write_new(HOME/'FROZEN_EVIDENCE_CLOSEOUT.json',dict(status='PASS',
            source_files=len(registration['source_sha256']),artifacts=len(registration['artifact_sha256']),
            original_R03_bytes_unchanged=True,original60_and_S3_semantic_reads=0))
        status='COMPLETE_OFFLINE_DIAGNOSTIC_NO_PPO'
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        print(error['traceback'],flush=True)
    finally:
        for table in tables:
            try: table.close()
            except Exception: pass
        budget.close(status,error)
    if error: raise SystemExit(1)


if __name__=='__main__': main()
