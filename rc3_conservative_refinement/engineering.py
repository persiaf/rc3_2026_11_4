"""Registered real-input acceptance for the conservative206 PPO integration.

This is engineering with a small number of actual Adam calls, never a planning
quality experiment.  Source and resource admission precede every request/update.
No scene generation or old optimizer entry point is reachable from this runner.
"""
from __future__ import annotations

import time
ENTRY_START = time.perf_counter()

import argparse
import copy
import io
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .guard import GuardLimits
from .model import RemainingCostCritic
from .runtime import (audit_real_feature_state, export_actor, load_actor,
                      load_deployed_actor, load_frozen_sl, plan_request, sl_route,
                      tensor_digest)
from .saved_inputs import SavedInputs
from .storage import ROOT, Ledger, digest, read, write_new
from .trainer import ConservativePPO, PPOConfig, checkpoint_equal


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_plan(path):
    plan = read(path)
    binding_path = ROOT / plan['source_binding_path']
    require(digest(binding_path) == plan['source_binding_sha256'], 'Engineering source binding changed')
    binding = read(binding_path)
    sources = binding.get('source_sha256', binding.get('source_files', {}))
    require(bool(sources), 'An exact nonempty engineering source binding is required')
    for name, entry in sources.items():
        sha = entry['sha256'] if isinstance(entry, dict) else entry
        require(digest(ROOT / name) == sha, 'Engineering bound source changed: ' + name)
    for name, entry in binding.get('artifacts', {}).items():
        sha = entry['sha256'] if isinstance(entry, dict) else entry
        require(digest(ROOT / name) == sha, 'Engineering bound artifact changed: ' + name)
    caps = plan['caps']
    require(set(caps) >= {'requests', 'actor_optimizer_attempts', 'critic_optimizer_attempts', 'wall_s'},
            'Engineering caps missing')
    require(caps['requests'] <= 32 and caps['actor_optimizer_attempts'] <= 16
            and caps['critic_optimizer_attempts'] <= 16 and caps['wall_s'] <= 1800,
            'Engineering caps exceed the separately registered bounds')
    return plan, binding


def unit_suite(folder):
    """Arithmetic/mock tests only; globally prohibit optimizer execution."""
    stream = io.StringIO()
    start = time.perf_counter()
    counters = dict(optimizer_step_attempts=0)
    def forbidden(*args, **kwargs):
        counters['optimizer_step_attempts'] += 1
        raise AssertionError('Optimizer call forbidden inside the synthetic unit suite')
    with patch.object(torch.optim.Adam, 'step', forbidden), patch.object(torch.optim.SGD, 'step', forbidden):
        home = Path(__file__).resolve().parent
        suite = unittest.defaultTestLoader.discover(str(home), pattern='test_*.py', top_level_dir=str(ROOT))
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    with (folder / 'UNITTEST_LOG.txt').open('x', encoding='utf-8') as output:
        output.write(stream.getvalue())
    report = dict(status='PASS' if result.wasSuccessful() and not any(counters.values()) else 'FAIL',
                  tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  elapsed_s=time.perf_counter() - start, counters=counters,
                  real_planning_requests=0, actual_optimizer_calls=0)
    write_new(folder / 'UNITTEST_REPORT.json', report)
    require(report['status'] == 'PASS', 'Synthetic unit suite failed; real engineering not started')
    return report


def save_checkpoint(path, trainer, metadata):
    payload = trainer.state_dict(metadata)
    with Path(path).open('xb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return dict(path=Path(path).relative_to(ROOT).as_posix(), sha256=digest(path)), payload


def seeded_critic(seed):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return RemainingCostCritic()


def run(plan_path, output):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    require(plan_path.is_relative_to(ROOT) and output.is_relative_to(ROOT), 'Engineering paths must stay in project')
    plan, binding = verify_plan(plan_path)
    require(output == (ROOT / plan['output']).resolve(), 'Engineering output is fixed by its registration; moving it cannot reset caps')
    known = {p.relative_to(ROOT).as_posix() for p in output.parent.glob('engineering_*/RESOURCE_ACCOUNTING.json')}
    require(known == {r['path'] for r in plan.get('prior_attempts', [])},
            'Every prior attempt must remain in the cumulative engineering account')
    require(all((p / 'RESOURCE_ACCOUNTING.json').exists() for p in output.parent.glob('engineering_*') if p.is_dir()),
            'Unclosed prior engineering attempt requires explicit recovery, not another directory')
    prior_counts = {}
    prior_wall = 0.
    for receipt in plan.get('prior_attempts', []):
        require(digest(ROOT / receipt['path']) == receipt['sha256'], 'Prior engineering failure/resource receipt changed')
        resource = read(ROOT / receipt['path'])
        require(digest(ROOT / receipt['ledger_path']) == receipt['ledger_sha256'], 'Prior exact wall-clock ledger changed')
        events = [json.loads(line) for line in (ROOT / receipt['ledger_path']).read_bytes().splitlines()]
        prior_wall += sum(e['elapsed_s'] for e in events if e['event'] == 'session_close')
        for key, count in resource['counts'].items():
            prior_counts[key] = prior_counts.get(key, 0) + count
    absolute = dict(requests=32, actor_optimizer_attempts=16, critic_optimizer_attempts=16, teacher_episodes=2, wall_s=1800)
    require(all(prior_counts.get(key, 0) + plan['caps'][key] <= value for key, value in absolute.items() if key != 'wall_s')
            and prior_wall + plan['caps']['wall_s'] <= absolute['wall_s'],
            'Engineering cumulative attempt budget cannot be reset')
    require(not output.exists(), 'Engineering output exists; failure or budget cannot be reset')
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    random.seed(11622026)
    np.random.seed(11622026)
    torch.manual_seed(11622026)
    ledger = Ledger(output, dict(stage='engineering', plan_path=plan_path.relative_to(ROOT).as_posix(),
                                plan_sha256=digest(plan_path), source_binding_sha256=plan['source_binding_sha256']), plan['caps'], started_at=ENTRY_START)
    started = time.perf_counter()
    status = 'FAIL'
    inputs = None
    checks, reports = {}, {}
    try:
        units = unit_suite(output)
        ledger.check()
        inputs = SavedInputs()

        def logger(row):
            if row['event'] == 'optimizer_attempt':
                category = 'actor_optimizer_attempts' if row['role'] in ('actor', 'matched_SL') else 'critic_optimizer_attempts'
                ledger.debit({category: 1}, optimizer_role=row['role'], opportunity=row['opportunity'], engineering=True)
            ledger.emit(row)

        def trainer(actor=None, limits=None, branch='engineering'):
            actor = load_actor('ON', 11622026) if actor is None else actor
            return ConservativePPO(actor, seeded_critic(11622026), config=PPOConfig(),
                                   limits=limits or GuardLimits(), log=logger, context=dict(branch=branch))

        def request(case, actor, name, *, critic=None, equivalence=False, greedy=True,
                    sampling_seed=11622026, logical_episode_id=None, independent_checks=False):
            holder = {}
            audit_count = [0]
            def audit(env, features):
                if audit_count[0] >= 3:
                    return None
                audit_count[0] += 1
                return audit_real_feature_state(env, features)
            def prepare():
                problem, evidence = inputs.view(case['spec'])
                holder['evidence'] = evidence
                return problem
            def compute():
                value = plan_request(prepare, actor=actor, critic=critic, greedy=greedy,
                                     sampling_seed=sampling_seed, check=ledger.check, equivalence=equivalence,
                                     state_audit=audit if independent_checks else None)
                value['episode_id'] = logical_episode_id or name
                value['record'].update(request_id=name, input_evidence=holder['evidence'])
                holder['episode'] = value
                return value['record']
            record = ledger.request(name, compute, case_id=case['id'], kind='engineering')
            return holder['episode'], record

        # Twelve complete requests: independent plainSL plus each zero206 arm.
        zero = []
        initial_fingerprints = []
        for case in inputs.engineering_cases:
            records = {}
            for condition in ('ON', 'OFF'):
                actor = load_actor(condition, 11622026)
                initial_fingerprints.append(tensor_digest(actor.correction.state_dict()))
                for mode in (True, False, True):
                    actor.train(mode)
                    require(not actor.encoder.training and all(not p.requires_grad and p.grad is None
                            for p in actor.encoder.parameters()), 'Mode toggle thawed the frozen SL')
                _, records[condition] = request(case, actor, f"zero/{case['id']}/{condition}",
                                                equivalence=True, independent_checks=condition == 'ON')
            def plain_request():
                tick = time.perf_counter()
                problem, evidence = inputs.view(case['spec'])
                preparation = time.perf_counter() - tick
                tick = time.perf_counter()
                model = load_frozen_sl()
                load_s = time.perf_counter() - tick
                record = sl_route(problem, model, check=ledger.check)
                record.update(input_evidence=evidence, input_preparation_s=preparation, model_startup_s=load_s,
                              service_time_before_IO_s=preparation + load_s + record['prepared_problem_request_s'])
                return record
            records['FrozenSL'] = ledger.request(f"zero/{case['id']}/FrozenSL", plain_request,
                                                  case_id=case['id'], kind='engineering')
            for key in ('raw_route', 'route', 'metrics'):
                require(records['ON'][key] == records['OFF'][key] == records['FrozenSL'][key],
                        f'ZeroON/OFF/FrozenSL disagree: {case["id"]}/{key}')
            zero.append(dict(case_id=case['id'], input_identity=records['ON']['input_identity'],
                             status='PASS', metrics=records['ON']['metrics'],
                             raw_route=records['ON']['raw_route'], route=records['ON']['route'],
                             independent_state_audits=records['ON']['independent_state_audits']))
        require(len(set(initial_fingerprints)) == 1, 'ON/OFF residual initialization differs')
        checks['zero_exact'] = zero
        checks['initial_on_off_same_correction'] = True

        # A continuous versus B restore branch.  Actual duplicate engineering
        # Adam calls are separately charged; no scientific resources are reused.
        case = inputs.engineering_cases[1]
        a = trainer(branch='continuous_A')
        episode1, _ = request(case, a.actor, 'ppo/A/op1', critic=a.critic, greedy=False,
                              logical_episode_id='engineering-resume/op1')
        report1 = a.update([episode1])
        require(report1['accepted'], 'Registered ordinary candidate was not accepted; do not widen guard')
        metadata = dict(source_binding_sha256=plan['source_binding_sha256'], plan_sha256=digest(plan_path),
                        logical_cursor=1, logical_request_budget=1, seed=11622026,
                        H_identity=binding.get('H_identity', binding.get('H_sha256')),
                        request_ledger_closed_sequence=len(ledger.events) - 1,
                        request_ledger_closed_tail=ledger.tail)
        ref, saved = save_checkpoint(output / 'accepted_op1.pt', a, metadata)
        require(digest(ROOT / ref['path']) == ref['sha256'], 'Checkpoint bytes changed before reload')
        loaded = torch.load(ROOT / ref['path'], map_location='cpu', weights_only=False)
        require(checkpoint_equal(saved, loaded), 'Serialized checkpoint did not preserve tensors/Adam/RNG/cursor')
        episodeA2, recordA2 = request(case, a.actor, 'ppo/A/op2', critic=a.critic, greedy=False,
                                     sampling_seed=11622027, logical_episode_id='engineering-resume/op2')
        reportA2 = a.update([episodeA2])
        expected = a.state_dict(dict(metadata, logical_cursor=2, logical_request_budget=2))
        b = trainer(branch='resumed_B')
        b.load_state_dict(loaded, expected_metadata=metadata)
        require(checkpoint_equal(b.state_dict(metadata), saved), 'Accepted checkpoint restore not exact')
        episodeB2, recordB2 = request(case, b.actor, 'ppo/B/op2', critic=b.critic, greedy=False,
                                     sampling_seed=11622027, logical_episode_id='engineering-resume/op2')
        reportB2 = b.update([episodeB2])
        actual = b.state_dict(dict(metadata, logical_cursor=2, logical_request_budget=2))
        require(checkpoint_equal(expected, actual), 'Continuous and paused/resumed execution differs')
        require(recordA2['actions'] == recordB2['actions'] and recordA2['metrics'] == recordB2['metrics'],
                'Resumed real policy actions/complete route differs')
        reports.update(first_accepted=report1, continuous_op2=reportA2, resumed_op2=reportB2)
        checks['continuous_resume_exact'] = dict(status='PASS', checkpoint=ref,
                                                actual_calls_duplicate_branch_separately_debited=True,
                                                compared=['correction', 'critic', 'bothAdam', 'allRNG',
                                                          'logical_cursor', 'logical_request_budget', 'used_episodes',
                                                          'resource_counters', 'actions', 'complete_metrics'])

        # Tighter ZERO limits force a rejection in a separate test fixture;
        # formal guard thresholds are never relaxed or changed to obtain a pass.
        rejected = trainer(limits=GuardLimits(0., 0., 0.), branch='forced_rejection_fixture')
        rejected_episode, _ = request(case, rejected.actor, 'ppo/rejection/op1', critic=rejected.critic,
                                      greedy=False, logical_episode_id='engineering-rejection/op1')
        before_head = copy.deepcopy(rejected.actor.correction.state_dict())
        before_adam = copy.deepcopy(rejected.actor_opt.state_dict())
        rejection = rejected.update([rejected_episode])
        require(not rejection['accepted'] and rejection['rollback_exact'], 'Forced rejection did not reject and restore')
        require(checkpoint_equal(before_head, rejected.actor.correction.state_dict())
                and checkpoint_equal(before_adam, rejected.actor_opt.state_dict()),
                'Rejected real parameters or Adam were retained')
        require(rejected.counts['critic_optimizer_calls'] == 1, 'Training-only Critic was not independently fitted')
        reports['forced_rejection'] = rejection
        checks['forced_rejection_exact'] = True

        # Greedy export comparison is new evidence, not reused old request data.
        direct_episode, direct = request(case, a.actor, 'deployment/direct', greedy=True)
        export_ref = export_actor(output / 'actor_only.pt', a.actor,
                                   dict(engineering=True, accepted_opportunities=a.counts['actor_accepted'],
                                        source_binding_sha256=plan['source_binding_sha256']))
        cold_holder = {}
        def prepare_export():
            problem, evidence = inputs.view(case['spec'])
            cold_holder['evidence'] = evidence
            return problem
        def cold_compute():
            result = plan_request(prepare_export,
                actor_factory=lambda: load_deployed_actor(output / 'actor_only.pt', expected_sha256=export_ref['sha256']),
                greedy=True, check=ledger.check)
            result['record']['input_evidence'] = cold_holder['evidence']
            return result['record']
        deployed = ledger.request('deployment/actor_only_cold', cold_compute, kind='engineering', case_id=case['id'])
        for key in ('raw_route', 'route', 'metrics'):
            require(direct[key] == deployed[key], 'Actor-only export/reload changed complete route/metrics')
        package = torch.load(output / 'actor_only.pt', map_location='cpu', weights_only=True)
        require(package['critic_included'] is False and package['optimizer_included'] is False,
                'Training objects entered actor-only deployment')
        checks['actor_only_exact'] = dict(status='PASS', actor=export_ref, direct_metrics=direct['metrics'],
                                         cold_model_startup_s=deployed['model_startup_s'],
                                         complete_cold_request_s=deployed['service_time_before_IO_s'],
                                         critic_and_adam_absent=True)
        # Exercise the real matched-supervision optimizer on an already saved
        # label. No new teacher/search query and no formal weight is retained.
        from .runner import teacher_episode
        control = trainer(actor=load_actor('OFF', 11622026), branch='matched_supervision_fixture')
        label_index = inputs.teacher_indices(11622026, 1)[0]
        ledger.debit({'teacher_episodes': 1}, teacher_index=label_index, engineering=True)
        label = teacher_episode(control.actor, inputs, label_index, 'engineering/teacher/op1')
        supervised = control.supervised_update([label])
        require(supervised['accepted'] and control.counts['critic_optimizer_calls'] == 0,
                'Real matched-supervision control failed; no guard relaxation')
        reports['matched_supervision'] = supervised
        checks['matched_supervision_real'] = True
        from rc3_global_local_impl_review.endpoint_baselines import run_endpoint
        for method in ('fixed3', 'detour16'):
            def native_compute():
                tick = time.perf_counter()
                problem, evidence = inputs.view(case['spec'])
                stats = {}
                pi, info = run_endpoint(problem, method, time_limit_s=60., reject_fallback=True, counters=stats)
                route = info['local_route']
                return dict(method=method, input_identity=problem.identity, plan=pi,
                            raw_route=route, route=route, metrics=problem.metrics(route),
                            common_H_added=info['common_H_added'], source=evidence,
                            counts=stats, request_before_IO_s=time.perf_counter()-tick)
            native = ledger.request('native/'+method, native_compute, kind='engineering')
            require(native['plan']['validation']['valid_model_plan'] and not native['common_H_added'],
                    'Native baseline failed or received an extra H')
            checks[method+'_real'] = dict(status='PASS', metrics=native['metrics'])
        reports['unit_suite'] = units
        write_new(output / 'UPDATE_REPORTS.json', reports)
        write_new(output / 'CHECKS.json', checks)
        status = 'PASS'
        return dict(status=status, checks=checks, unit_suite=units)
    except BaseException as error:
        write_new(output / 'FAILURE.json', dict(error_type=type(error).__name__, message=str(error),
                                                counts=ledger.counts, partial_checks=checks,
                                                attempted_updates_and_failures_remain_charged=True))
        raise
    finally:
        elapsed = time.perf_counter() - started
        resource = dict(status=status, stage='REAL_INPUT_OPTIMIZER_ENGINEERING_ONLY',
                        elapsed_body_s=elapsed, counts=ledger.counts, caps=plan['caps'],
                        charged_session_wall_s=time.perf_counter()-ledger.started,
                        input_counts=None if inputs is None else inputs.counts,
                        python=sys.version, platform=platform.platform(), torch=torch.__version__,
                        numpy=np.__version__, device='cpu', formal_training_updates=0,
                        quality_performance_claim=False, source_binding_sha256=plan['source_binding_sha256'])
        write_new(output / 'RESOURCE_ACCOUNTING.json', resource)
        ledger.close(status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.plan, args.output)
    print('ENGINEERING_' + report['status'])
    print('UNIT_TESTS=' + str(report['unit_suite']['tests']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
