"""Fixed M2 development evaluation; no model or request runs on import.

LOCAL66 and FULL12 are always separate reporting populations.  A failed
request remains a failed row and terminates this evaluation transaction.
Only a complete endpoint is a resumable training boundary.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import time
import numpy as np

from .common import ROOT, put, sha
from .runtime import collect, load_export, instrument, request_record

METHODS = ('FrozenSL', 'RelationInit', 'RelationSL', 'RelationRL', 'Fixed3', 'Detour16')


def _native(pool, tid, ledger, rid, method):
    from rc3_global_local_impl_review.endpoint_baselines import run_endpoint
    tick = ledger.begin('request', rid, task=tid, purpose='native_' + method)
    try:
        started = time.perf_counter()
        p, source = pool.get(tid)
        input_s = time.perf_counter() - started
        counts = Counter()
        ledger.check()
        plan, info = run_endpoint(p, 'fixed3' if method == 'Fixed3' else 'detour16',
                                  time_limit_s=None, reject_fallback=True, counters=counts)
        route = info['local_route']
        metrics = p.metrics(route)
        if not plan['validation']['valid_model_plan']:
            raise ValueError('Native independent validation failed')
        ledger.counts['native_requests'] += 1
        internal = info['internal_search']
        for name, value in internal.get('counts', {}).items():
            ledger.counts['native_internal_' + name] += value
        record = dict(request_id=rid, task_id=tid, method=method, valid=True,
                      problem_identity=p.identity, parent_identity=p.parent.identity,
                      physical_key=source['physical_key'], entry=p.entry, exit=p.exit,
                      candidate_count=len(p.candidate_ids), task_count=len(p.task_ids),
                      metrics=dict(metrics, raw_J=metrics['J'], raw_L=metrics['L'], raw_K=metrics['K']),
                      raw_route=route, route=route, raw_plan=plan, plan=plan,
                      raw_semantics='Native final including original internal search; duplicated scalar columns are not a no-H policy result',
                      common_H_added=False, counts=counts, internal_search=internal,
                      input_sources=source['input_source_files'],
                      timing=dict(input_s=input_s, complete_request_s=time.perf_counter()-tick))
        return request_record(ledger, rid, record, tick)
    except BaseException as exc:
        ledger.fail('request', rid, exc)
        raise


def _subsets(rows, population):
    yield 'overall', rows
    if population == 'LOCAL66':
        for name, group in (('SEL', 'trained_parent_new_tasks_exposed_selection'),
                            ('DEV', 'independent_parent_exposed_development')):
            yield name, [r for r in rows if r['group'] == group]
        for w, h in ((4, 3), (6, 6), (8, 8)):
            yield f'{w}x{h}', [r for r in rows if (r['width'], r['height']) == (w, h)]


def summarize(rows):
    """Report completeness before means; no mean over a successful subset."""
    result = {}
    for population in ('LOCAL66', 'FULL12'):
        result[population] = {}
        for method in METHODS:
            base = [r for r in rows if r['population'] == population and r['method'] == method]
            result[population][method] = {}
            for group, subset in _subsets(base, population):
                expected = (12 if population == 'FULL12' else
                            {'overall': 66, 'SEL': 48, 'DEV': 18, '4x3': 22, '6x6': 22, '8x8': 22}[group])
                complete = len(subset) == expected and all(r['valid'] for r in subset)
                out = dict(planned=expected, processed=len(subset), valid=sum(r['valid'] for r in subset), complete=complete,
                           failure_ids=[r['case_id'] for r in subset if not r['valid']])
                if complete:
                    parents = defaultdict(list)
                    for r in subset:
                        parents[r['physical_key']].append(r)
                    out.update(physical_parents=len(parents),
                               request_mean={m: float(np.mean([r['metrics'][m] for r in subset]))
                                             for m in ('J', 'L', 'K', 'raw_J', 'raw_L', 'raw_K')},
                               parent_equal_mean={m: float(np.mean([np.mean([r['metrics'][m] for r in pr])
                                                                    for pr in parents.values()]))
                                                  for m in ('J', 'L', 'K', 'raw_J', 'raw_L', 'raw_K')},
                               time_p50_s=float(np.median([r['full_request_s'] for r in subset])),
                               time_p95_s=float(np.quantile([r['full_request_s'] for r in subset], .95)))
                result[population][method][group] = out
    return result


def paired_gain(rows, population, reference, group='overall', metric='J'):
    rr = [r for r in rows if r['population'] == population]
    if group != 'overall':
        subset_map = dict(_subsets(rr, population))
        rr = subset_map[group]
    candidate = {r['case_id']: r for r in rr if r['method'] == 'RelationRL'}
    control = {r['case_id']: r for r in rr if r['method'] == reference}
    if not candidate or set(candidate) != set(control) or any(not r['valid'] for r in [*candidate.values(), *control.values()]):
        return dict(complete=False, mean=None, parents={})
    grouped = defaultdict(list)
    for case_id, row in candidate.items():
        if row['physical_key'] != control[case_id]['physical_key']:
            raise ValueError('Paired physical-parent identity mismatch')
        grouped[row['physical_key']].append(case_id)
    gains = {}
    for parent, ids in grouped.items():
        baseline = float(np.mean([control[i]['metrics'][metric] for i in ids]))
        value = float(np.mean([candidate[i]['metrics'][metric] for i in ids]))
        if baseline <= 0 or not np.isfinite(baseline + value):
            return dict(complete=False, mean=None, parents={})
        gains[parent] = (baseline - value) / baseline
    return dict(complete=True, mean=float(np.mean(list(gains.values()))), parents=gains)


def gate(report):
    rows, endpoint = report['rows'], report['endpoint']
    comparisons = {}
    for pop in ('LOCAL66', 'FULL12'):
        comparisons[pop] = {}
        groups = ('overall', 'SEL', 'DEV', '4x3', '6x6', '8x8') if pop == 'LOCAL66' else ('overall',)
        for reference in ('FrozenSL', 'RelationInit', 'RelationSL'):
            comparisons[pop][reference] = {g: paired_gain(rows, pop, reference, g) for g in groups}
    complete = (report['complete'] and len(rows) == 468 and all(r['valid'] for r in rows)
                and all(v['complete'] for pop in comparisons.values() for ref in pop.values() for v in ref.values()))
    if not complete:
        return dict(status='INSUFFICIENT', passed=False, endpoint=endpoint, comparisons=comparisons)
    if endpoint == 0:
        return dict(status='BASELINE_RECORDED', passed=None, endpoint=0, comparisons=comparisons)
    if endpoint == 512:
        severe = any(comparisons[pop]['RelationInit']['overall']['mean'] < -.03 for pop in comparisons)
        return dict(status='STOP_SEVERE_DEGRADATION' if severe else 'CONTINUE_REGISTERED_2048',
                    passed=not severe, endpoint=512, comparisons=comparisons)
    if endpoint != 2048:
        raise ValueError('Unregistered evaluation endpoint')
    local, full = comparisons['LOCAL66'], comparisons['FULL12']
    passed = (local['FrozenSL']['overall']['mean'] >= .01
              and local['RelationInit']['overall']['mean'] >= .005
              and local['RelationSL']['overall']['mean'] >= .005
              and all(local[ref][g]['mean'] >= 0 for ref in ('RelationInit', 'RelationSL')
                      for g in ('SEL', 'DEV', '4x3', '6x6', '8x8'))
              and all(full[ref]['overall']['mean'] >= 0 for ref in full))
    return dict(status='SCIENTIFIC_PASS' if passed else 'NO_DEVELOPMENT_SIGNAL', passed=bool(passed),
                endpoint=2048, comparisons=comparisons,
                interpretation='Exposed development only; no new-parent generalization claim')


def evaluate_endpoint(ledger, pool, cases, endpoint, exports, device):
    """Execute the exact 6 x 78 grid once; partial grids cannot be restarted."""
    from rc3_conservative_refinement.runtime import load_frozen_sl
    if endpoint not in (0, 512, 2048) or len(cases['LOCAL66']) != 66 or len(cases['FULL12']) != 12:
        raise ValueError('Exact predeclared evaluation grid required')
    folder = ledger.folder / 'evaluation' / f'endpoint_{endpoint:04}'
    report_path = folder / 'RESULTS.json'
    if report_path.exists():
        raise RuntimeError('Endpoint already started; do not rerun a consumed grid')
    report = dict(endpoint=endpoint, complete=False, planned_requests=468, rows=[],
                  populations_separate=True, method_order=list(METHODS), model_setup={},
                  input_cache_policy='Clear immutable view cache before each request; no route/result cache',
                  timing='Full collect/native request including persistence; model load recorded separately',
                  device=str(device), native_device='cpu')
    put(report_path, report)
    actors = {}
    for method in METHODS[:4]:
        tick = time.perf_counter()
        actor = load_frozen_sl().to(device) if method == 'FrozenSL' else load_export(exports[method], device=device)
        actor.eval().requires_grad_(False)
        actors[method] = instrument(actor, ledger)
        report['model_setup'][method] = dict(load_s=time.perf_counter()-tick,
                                               source=exports.get(method, getattr(actor, 'base_identity', None)))
    put(report_path, report)
    try:
        for population in ('LOCAL66', 'FULL12'):
            for case in cases[population]:
                for method in METHODS:
                    rid = f'eval_{endpoint:04}_{population}_{case["id"]}_{method}'
                    row = dict(case_id=case['id'], population=population, method=method, endpoint=endpoint,
                               group=case.get('group', 'exposed_training_parent_full'),
                               physical_key=case['physical_key'], width=case['spec']['width'], height=case['spec']['height'],
                               mode=case['spec']['mode'], valid=False)
                    tick = time.perf_counter()
                    try:
                        pool.pool.cache.clear()
                        out = (_native(pool, case['id'], ledger, rid, method) if method in METHODS[4:] else
                               collect(actors[method], pool, case['id'], ledger, rid, greedy=True))
                        full_s = time.perf_counter()-tick
                        if out['physical_key'] != case['physical_key']:
                            raise ValueError('Evaluation parent changed')
                        row.update(valid=bool(out['valid']), metrics=out['metrics'], full_request_s=full_s,
                                   problem_identity=out['problem_identity'], entry=out['entry'], exit=out['exit'],
                                   candidate_count=out['candidate_count'], task_count=out['task_count'],
                                   record=dict(path=(ledger.folder/'requests'/(rid+'.json')).relative_to(ROOT).as_posix(),
                                               sha256=sha(ledger.folder/'requests'/(rid+'.json'))),
                                   raw_semantics=out.get('raw_semantics', 'Policy constructive raw and same fixed H'))
                    except BaseException as exc:
                        row.update(error=repr(exc), full_request_s=time.perf_counter()-tick)
                        report['rows'].append(row)
                        report['summary'] = summarize(report['rows'])
                        put(report_path, report)
                        raise
                    report['rows'].append(row)
                    put(report_path, report)
        report['complete'] = len(report['rows']) == 468
        report['summary'] = summarize(report['rows'])
        report['gate'] = gate(report)
        put(report_path, report)
        return dict(path=report_path.relative_to(ROOT).as_posix(), sha256=sha(report_path),
                    endpoint=endpoint, gate=report['gate'])
    finally:
        actors.clear()


__all__ = ['METHODS', 'evaluate_endpoint', 'summarize', 'paired_gain', 'gate']
