"""Paired inference on saved training-teacher prefixes; no alternative rollout.

The six endpoint models are diagnostic instances, not new scientific starts.
Every Environment.step uses the original teacher action. No H is evaluated.
"""
from pathlib import Path
import csv
import json
import sys
import time
from contextlib import ExitStack
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from rc3_global_local.environment import Environment
from rc3_global_local.policy import GlobalLocalPolicy, tensor_hash
from rc3_global_local_trainer.data import TrainingRegistry, SceneProvider
from rc3_global_local_trainer.runner import TeacherPool, StageClock, BudgetedTerminal
from rc3_global_local_execution_v3.authority import require_stage_authority, PROTOCOL, REGISTRY
from rc3_s1b_results_20261106.audit_s1b import RUN, HERE, SEEDS, read, digest, write_new

def main():
    start = time.perf_counter()
    if (HERE/'DIAGNOSTIC_STARTED.json').exists():
        raise RuntimeError('Diagnostic attempt already exists; no implicit retry')
    plan = read(HERE/'DIAGNOSTIC_PLAN.json')
    complete = [read(RUN/f'stage_admission/S1-B/{seed}/COMPLETE.json') for seed in SEEDS]
    training_wall = [next(iter(c['cumulative_stage_resources'].values()))['elapsed_wall_s'] for c in complete]
    limit = min(plan['diagnostic_wall_cap_s'], 1800-max(training_wall))
    write_new(HERE/'DIAGNOSTIC_STARTED.json', dict(plan_sha256=digest(HERE/'DIAGNOSTIC_PLAN.json'),
        training_completion_sha256=[digest(RUN/f'stage_admission/S1-B/{s}/COMPLETE.json') for s in SEEDS],
        effective_wall_cap_s=limit, implicit_retry_allowed=False))
    ledger = dict(diagnostic_policy_instances=0, diagnostic_optimizer_updates=0,
        new_scenes=0, navigation_solves=0, native_calls=0, H_calls=0, teacher_queries=0,
        model_encodes=0, model_scores=0, saved_teacher_terminal_metric_replays=0)
    def check_time():
        if time.perf_counter()-start >= limit:
            raise TimeoutError('Registered diagnostic time cap exhausted')
    def forbid(*a, **k):
        raise AssertionError('A forbidden generation/optimizer/teacher operation was called during diagnostics')
    auth = ROOT/'rc3_global_local_execution_v3/authorizations/20261106_S1_B'
    provider_ledger = {}; model_rows = []; route_rows = []
    try:
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        permit = require_stage_authority(PROTOCOL, REGISTRY, auth/'SOURCE_REVISION_APPROVAL.json',
            auth/'S1_B_AUTHORIZATION.json', 'S1-B', SEEDS[0], 'rc3_v3_global_local_20261106_01',
            ROOT/'rc3_navigation_compat_20261106/SOURCE_BINDING_UPDATED.json')
        registry = TrainingRegistry(REGISTRY)
        provider = SceneProvider(registry, RUN/'data', provider_ledger, permit.execution,
            deadline=start+limit, read_only=True)
        clock = StageClock(limit)
        pool = TeacherPool(RUN/'teachers256', registry, provider, provider_ledger, clock, permit.binding['H_sha256'])
        pool.validate()
        with ExitStack() as stack:
            stack.enter_context(threadpool_limits(limits=1))
            for target in ['rc3_baseline.baseline.navigation', 'rc3_fast.scenes.make_complex_scene',
                'rc3_constructive.problem.resample_tasks', 'rc3_fast.solver.vectorized_feasibility',
                'rc3_global_local.terminal.finish', 'rc3_global_local_trainer.runner.run_endpoint',
                'torch.optim.Adam.step', 'torch.Tensor.backward']:
                stack.enter_context(patch(target, side_effect=forbid))
            stack.enter_context(patch.object(BudgetedTerminal, '__call__', side_effect=forbid))
            models = []
            for seed in SEEDS:
                for endpoint, name in [('initial','untrained.pt'), ('SL256','update_0256.pt')]:
                    p = RUN/f'seed_{seed}/S1'/name; side = read(p.with_suffix('.pt.json'))
                    if digest(p) != side['sha256']:
                        raise ValueError('Endpoint checkpoint checksum mismatch')
                    state = torch.load(p, map_location='cpu', weights_only=False)
                    torch.manual_seed(0)  # diagnostic construction only; overwritten by saved tensors
                    model = GlobalLocalPolicy(); ledger['diagnostic_policy_instances'] += 1
                    model.load_state_dict(state['model'], strict=True); model.eval()
                    if tensor_hash(model) != side['model_tensor_sha256']:
                        raise ValueError('Loaded endpoint tensor identity mismatch')
                    models.append((seed,endpoint,model))
                    model_rows.append(dict(seed=seed,endpoint=endpoint,checkpoint=p.relative_to(ROOT).as_posix(),
                        checkpoint_sha256=side['sha256'],tensor_sha256=side['model_tensor_sha256']))
            fields = ['seed','endpoint','teacher_index','view_identity','prefix_length','teacher_node','predicted_node',
                'legal_count','teacher_rank','CE','teacher_probability','top1','top3','top5','uniform_CE','uniform_top1']
            with (HERE/'teacher_state_metrics.csv').open('x',encoding='utf8',newline='') as f:
                writer = csv.DictWriter(f,fieldnames=fields);writer.writeheader()
                with torch.inference_mode():
                    for index in range(256):
                        check_time(); example = pool.example(index)
                        view, actions = example['view'], example['actions']
                        for seed,endpoint,model in models:
                            def terminal(p,raw):
                                ledger['saved_teacher_terminal_metric_replays'] += 1
                                return p.metrics(raw)
                            env = Environment(view, terminal)
                            encoded = model.encode(view); ledger['model_encodes'] += 1
                            measures = []
                            for prefix,node in enumerate(actions):
                                check_time(); mask = env.mask(); logits = model.score(env, encoded)
                                ledger['model_scores'] += 1
                                legal = torch.as_tensor(mask, dtype=torch.bool)
                                if not bool(legal[node-1]) or not bool(torch.isfinite(logits[legal]).all()) or not bool(torch.isneginf(logits[~legal]).all()):
                                    raise ValueError('Illegal teacher action, nonfinite score or mask leak')
                                prediction = int(torch.argmax(logits))+1
                                if not mask[prediction-1]:
                                    raise ValueError('Masked argmax returned illegal action')
                                logp = torch.log_softmax(logits,0)[node-1]
                                indices = torch.arange(len(logits))
                                rank = int((logits > logits[node-1]).sum()) + int(((logits == logits[node-1]) & (indices < node-1)).sum()) + 1
                                n = int(mask.sum())
                                row = dict(seed=seed,endpoint=endpoint,teacher_index=index,view_identity=view.identity,
                                    prefix_length=prefix,teacher_node=node,predicted_node=prediction,legal_count=n,
                                    teacher_rank=rank,CE=-float(logp),teacher_probability=float(logp.exp()),
                                    top1=int(prediction==node),top3=int(rank<=3),top5=int(rank<=5),
                                    uniform_CE=float(np.log(n)),uniform_top1=1./n)
                                writer.writerow(row); measures.append(row)
                                env.step(node)  # NEVER execute the predicted alternative
                            if not env.done or abs(env.result['J']-example['teacher_J']) > 1e-7:
                                raise ValueError('Saved teacher prefix did not reproduce its terminal metric')
                            route_rows.append(dict(seed=seed,endpoint=endpoint,teacher_index=index,view_identity=view.identity,
                                decisions=len(actions),route_macro_CE=float(np.mean([r['CE'] for r in measures])) if measures else 0.,
                                correct=sum(r['top1'] for r in measures),top3=sum(r['top3'] for r in measures),top5=sum(r['top5'] for r in measures),
                                CE_sum=sum(r['CE'] for r in measures),probability_sum=sum(r['teacher_probability'] for r in measures),
                                rank_sum=sum(r['teacher_rank'] for r in measures),uniform_CE_sum=sum(r['uniform_CE'] for r in measures),
                                uniform_correct_sum=sum(r['uniform_top1'] for r in measures),
                                nontrivial_decisions=sum(r['legal_count']>1 for r in measures),
                                nontrivial_correct=sum(r['top1'] for r in measures if r['legal_count']>1),
                                predicted_illegal_actions=0))
                        if (index+1)%32==0:
                            print(json.dumps(dict(teachers_diagnosed=index+1,wall_s=time.perf_counter()-start)),flush=True)
            groups = []
            for m in model_rows:
                routes = [r for r in route_rows if r['seed']==m['seed'] and r['endpoint']==m['endpoint']]
                n = sum(r['decisions'] for r in routes); nt = sum(r['nontrivial_decisions'] for r in routes)
                groups.append(dict(m,teachers=len(routes),decisions=n,route_macro_CE=float(np.mean([r['route_macro_CE'] for r in routes])),
                    state_micro_CE=sum(r['CE_sum'] for r in routes)/n,top1=sum(r['correct'] for r in routes)/n,
                    top3=sum(r['top3'] for r in routes)/n,top5=sum(r['top5'] for r in routes)/n,
                    teacher_probability_mean=sum(r['probability_sum'] for r in routes)/n,
                    mean_teacher_rank=sum(r['rank_sum'] for r in routes)/n,
                    uniform_state_micro_CE=sum(r['uniform_CE_sum'] for r in routes)/n,
                    uniform_top1=sum(r['uniform_correct_sum'] for r in routes)/n,
                    nontrivial_decisions=nt,nontrivial_top1=sum(r['nontrivial_correct'] for r in routes)/nt if nt else None,
                    predicted_illegal_actions=0))
            write_new(HERE/'teacher_route_metrics.json',route_rows)
            result = dict(schema='rc3-s1b-paired-teacher-diagnostics-v1',status='PASS',plan_sha256=digest(HERE/'DIAGNOSTIC_PLAN.json'),
                groups=groups,ledger=ledger,provider_ledger=provider_ledger,wall_s=time.perf_counter()-start,
                wall_cap_s=limit,conservative_training_plus_full_diagnostic_wall_s={str(seed):t+time.perf_counter()-start for seed,t in zip(SEEDS,training_wall)},
                assessment='Training teacher prefixes only; no new routes, generalization or DRL gain evaluated',
                diagnostic_script_sha256=digest(__file__),per_state_csv_sha256=digest(HERE/'teacher_state_metrics.csv'))
            if result['wall_s'] >= limit:
                raise TimeoutError('Diagnostic cap exceeded during output IO')
            write_new(HERE/'TEACHER_STATE_DIAGNOSTICS.json',result)
            print(json.dumps(dict(status=result['status'],groups=groups,wall_s=result['wall_s'])),flush=True)
    except Exception as exc:
        write_new(HERE/'DIAGNOSTIC_FAILED.json',dict(error=type(exc).__name__,message=str(exc),ledger=ledger,
            provider_ledger=provider_ledger,wall_s=time.perf_counter()-start,no_automatic_retry=True))
        raise

if __name__ == '__main__':
    main()
