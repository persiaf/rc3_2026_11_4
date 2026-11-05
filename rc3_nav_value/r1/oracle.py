"""Budgeted, certified terminal costs for R1.

This module does not create scenes or policies. A terminal oracle invocation is
pre-debited before any objective/solver/export work; failures remain charged.
Only explicit caller-supplied cache keys can reuse a full-H result. The caller
must encode the full scene, ordered prefix/action, reference and H versions.

Instrumentation patches shared frozen modules, so calls are serialized by the
ledger's lock. Counts are function entries, not iterations or FLOPs. Inclusive
and exclusive internal times are kept separately to avoid summing nested work.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
from unittest.mock import patch

import numpy as np
from rc3_baseline.validator import validate
from rc3_constructive import baselines
from rc3_fast import solver
from rc3_v3 import environment as _old_env
from rc3_v4 import runtime as _runtime

ROOT = Path(__file__).resolve().parents[2]
LIMITS = dict(init=4096, round=8192, audit=1024, validation=3072)
TOTAL_LIMIT = 16384
INTERNAL = ('construct', 'two_opt', 'joint_search', 'recipe')
CERTIFICATION = ('export_attempts', 'export_completed', 'scalar_attempts', 'scalar_completed')
TOL = 1e-7


def _default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Not JSON serializable: {type(value).__name__}')


def _dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False, default=_default)


def _hash(value):
    return hashlib.sha256(_dumps(value).encode('utf-8')).hexdigest()


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _append(path, value):
    payload = _dumps(value) + '\n'
    with Path(path).open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _empty_internal():
    return dict(counts={k: 0 for k in INTERNAL}, inclusive_s={k: 0. for k in INTERNAL},
                exclusive_s={k: 0. for k in INTERNAL})


@contextmanager
def _instrument(result):
    stack = []

    def wrapped(name, fn):
        def call(*args, **kwargs):
            result['counts'][name] += 1
            frame = dict(start=time.perf_counter(), child_s=0.)
            stack.append(frame)
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - frame['start']
                stack.pop()
                result['inclusive_s'][name] += elapsed
                result['exclusive_s'][name] += max(0., elapsed-frame['child_s'])
                if stack:
                    stack[-1]['child_s'] += elapsed
        return call

    with patch.object(solver, 'construct', wrapped('construct', solver.construct)), \
         patch.object(solver, 'two_opt', wrapped('two_opt', solver.two_opt)), \
         patch.object(solver, 'joint_search', wrapped('joint_search', solver.joint_search)), \
         patch.object(_old_env, 'solve_route', wrapped('recipe', _old_env.solve_route)):
        yield


class Ledger:
    """Durable budget journal. ``deadline`` is an absolute perf_counter value.

    ``check_hook`` may be assigned a callable for a caller-specific wall check.
    ``check`` is also available to raw-label and model loops between calls.
    An incomplete previous call is never silently retried on construction.
    """
    def __init__(self, out):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.journal = self.out/'call_journal.jsonl'
        self.terminal = self.out/'terminal.jsonl'
        self.cache_journal = self.out/'cache_hits.jsonl'
        self.deadline = math.inf
        self.check_hook = None
        self.lock = threading.RLock()
        self.used = {k: 0 for k in LIMITS}
        self.started = self.used
        self.completed = {k: 0 for k in LIMITS}
        self.failed = {k: 0 for k in LIMITS}
        self.internal_counts = {k: 0 for k in INTERNAL}
        self.internal_inclusive_s = {k: 0. for k in INTERNAL}
        self.internal_exclusive_s = {k: 0. for k in INTERNAL}
        self.certification_counts = {k: 0 for k in CERTIFICATION}
        self.cache_hits = 0
        self.oracle_s = 0.
        pending = {}
        if self.journal.exists():
            for line in self.journal.read_text(encoding='utf-8').splitlines():
                record = json.loads(line)
                event = record['event']
                if event == 'start':
                    category = record['category']
                    if category not in LIMITS or record['call_id'] in pending or record['call_id'] != self.total+1:
                        raise ValueError('Malformed previous oracle journal')
                    self.used[category] += 1
                    pending[record['call_id']] = category
                elif event in ('complete', 'error'):
                    call_id = record['call_id']
                    if pending.pop(call_id, None) != record['category']:
                        raise ValueError('Oracle journal completion has no matching debit')
                    self._aggregate(record, completed=event == 'complete')
                elif event == 'cache_hit':
                    self.cache_hits += 1
                else:
                    raise ValueError('Unknown oracle journal event')
        if pending:
            raise RuntimeError(f'Prior terminal calls remain charged and incomplete; no implicit retry: {sorted(pending)}')
        if self.total > TOTAL_LIMIT or any(self.used[k] > LIMITS[k] for k in LIMITS):
            raise ValueError('Previous oracle journal exceeds frozen limits')

    @property
    def total(self):
        return sum(self.used.values())

    @property
    def counts(self):
        return self.used

    @property
    def internal(self):
        return self.internal_counts

    def check(self):
        if self.check_hook is not None:
            self.check_hook()
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise TimeoutError('R1 frozen wall limit reached before next oracle/model work')

    def _reserve(self, kind, category, metadata):
        self.check()
        if category not in LIMITS:
            raise ValueError(f'Unknown oracle budget category: {category}')
        if self.used[category] >= LIMITS[category] or self.total >= TOTAL_LIMIT:
            raise RuntimeError(f'R1 terminal budget exhausted: {category}; used={self.used}')
        call_id = self.total + 1
        _append(self.journal, dict(event='start', call_id=call_id, kind=kind, category=category,
                                  metadata=metadata, clock_s=time.perf_counter()))
        # Journal flush/fsync precedes all actual terminal work.
        self.used[category] += 1
        return call_id

    def _aggregate(self, record, completed):
        (self.completed if completed else self.failed)[record['category']] += 1
        for key in INTERNAL:
            self.internal_counts[key] += record['internal']['counts'][key]
            self.internal_inclusive_s[key] += record['internal']['inclusive_s'][key]
            self.internal_exclusive_s[key] += record['internal']['exclusive_s'][key]
        for key in CERTIFICATION:
            self.certification_counts[key] += record['certification_counts'][key]
        self.oracle_s += record['oracle_s']

    def _finish(self, record, completed):
        _append(self.terminal, record)
        journal_record = {k: record[k] for k in ('call_id', 'kind', 'category', 'internal',
                                                'certification_counts', 'oracle_s')}
        journal_record['event'] = 'complete' if completed else 'error'
        if not completed:
            journal_record['error'] = record['error']
        _append(self.journal, journal_record)
        self._aggregate(journal_record, completed)
        self.write_summary()

    def _cache_hit(self, category, metadata, key_hash, source_call_id):
        self.check()
        if category not in LIMITS:
            raise ValueError(f'Unknown oracle budget category: {category}')
        record = dict(event='cache_hit', category=category, metadata=metadata,
                      cache_key_sha256=key_hash, source_call_id=source_call_id, clock_s=time.perf_counter())
        _append(self.cache_journal, record)
        _append(self.journal, record)
        self.cache_hits += 1
        self.write_summary()

    def snapshot(self):
        return dict(schema='rc3-r1-oracle-ledger-v1', limits=dict(LIMITS), total_limit=TOTAL_LIMIT,
                    used=dict(self.used), started=dict(self.used), completed=dict(self.completed),
                    failed=dict(self.failed), total_started=self.total,
                    total_completed=sum(self.completed.values()), total_failed=sum(self.failed.values()),
                    cache_hits=self.cache_hits, internal_counts=dict(self.internal_counts),
                    internal_inclusive_s=dict(self.internal_inclusive_s),
                    internal_exclusive_s=dict(self.internal_exclusive_s),
                    certification_counts=dict(self.certification_counts), oracle_s=self.oracle_s,
                    timing_scope='Inclusive timers overlap nested recipe/search calls; exclusive timers do not. Oracle time excludes raw completion performed by caller and journal IO.',
                    accounting_scope='Every attempted native or full-H evaluation pre-debited once; failed calls remain charged. Explicit cache hits have no new solver/H or certification work.')

    def write_summary(self):
        value = self.snapshot()
        destination = self.out/'ledger_summary.json'
        temporary = destination.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                             encoding='utf-8', newline='\n')
        temporary.replace(destination)
        return value


def _route(route):
    values = list(route)
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in values):
        raise ValueError('Route nodes must be integer original navigation-node IDs')
    values = [int(x) for x in values]
    if len(values) < 2 or values[0] != 0 or values[-1] != 0:
        raise ValueError('Expected complete closed depot route')
    return values


def _metrics(item, route, positive_gains=False):
    route = _route(route)
    nodes = route[1:-1]
    if len(nodes) != len(set(nodes)) or any(n < 1 or n > item.C.shape[1] for n in nodes):
        raise ValueError('Repeated or out-of-range service node')
    remaining = np.ones(item.C.shape[0], dtype=bool)
    for a, b in zip(route, route[1:]):
        if not np.isfinite(item.D[a, b]):
            raise ValueError('Nonfinite navigation leg')
        if b:
            gain = remaining & item.C[:, b-1]
            if positive_gains and not gain.any():
                raise ValueError('Raw construction includes a zero-gain action')
            remaining &= ~item.C[:, b-1]
    if remaining.any():
        raise ValueError('Complete route leaves tasks uncovered')
    penalty = float(item.base.penalty)
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError('Invalid stop penalty')
    length = float(sum(item.D[a, b] for a, b in zip(route, route[1:])))
    return dict(J=length+penalty*len(nodes), L=length, K=len(nodes))


class Oracle:
    """Exact fixed-H and native strong-rule terminal evaluations.

    Returned ``plan`` is available for caller-selected final artifacts but is not
    duplicated in terminal.jsonl. ``plan_content_sha256`` hashes canonical JSON
    content, not bytes of an independently saved pretty-printed JSON file.
    """
    def __init__(self, ledger):
        self.ledger = ledger
        self.cache = {}
        h_sources = {name: _file_hash(ROOT/name) for name in
                     ('rc3_constructive/baselines.py', 'rc3_fast/solver.py')}
        self.H_sha256 = _hash(h_sources)
        self.source_versions = dict(H_sha256=self.H_sha256, H_source_sha256=h_sources,
            oracle_source_sha256=_file_hash(__file__),
            runtime_sha256=_file_hash(ROOT/'rc3_v4/runtime.py'),
            scalar_validator_sha256=_file_hash(ROOT/'rc3_baseline/validator.py'))

    def _export(self, item, route, method, certification):
        certification['export_attempts'] += 1
        plan = item.certify(route, method)
        certification['export_completed'] += 1
        return plan

    def _verify(self, item, plan, route, certification):
        exported = copy.deepcopy(plan.get('validation', {}))
        certification['scalar_attempts'] += 1
        independent = validate(item.base.scene, plan, item.base.sparse)
        certification['scalar_completed'] += 1
        if not exported.get('valid_model_plan') or not exported.get('all_tasks_served'):
            raise ValueError(f'Export validation failed: {exported}')
        if not independent.get('valid_model_plan') or not independent.get('all_tasks_served'):
            raise ValueError(f'Independent scalar validation failed: {independent}')
        actual = [0]+[int(x)+1 for x in plan['visit_order_candidate_ids']]+[0]
        if actual != route:
            raise ValueError('Plan export changed the evaluated route')
        metrics = _metrics(item, route)
        if abs(plan['objective_length_plus_stops_m']-metrics['J']) > TOL:
            raise ValueError('Plan objective differs from evaluated navigation route')
        if abs(plan['route_length_m']-metrics['L']) > TOL:
            raise ValueError('Plan length differs from evaluated navigation route')
        return exported, independent, metrics

    def _call(self, kind, category, metadata, compute):
        with self.ledger.lock:
            call_id = self.ledger._reserve(kind, category, metadata)
            internal = _empty_internal()
            certification = {k: 0 for k in CERTIFICATION}
            started = time.perf_counter()
            try:
                with _instrument(internal):
                    result = compute(certification)
            except Exception as error:
                # Reserve is never refunded even when a solver/export/check fails.
                record = dict(status='error', call_id=call_id, kind=kind, category=category,
                              metadata=copy.deepcopy(metadata), internal=internal,
                              certification_counts=certification, oracle_s=time.perf_counter()-started,
                              source_versions=copy.deepcopy(self.source_versions),
                              error=dict(type=type(error).__name__, message=str(error)))
                self.ledger._finish(record, completed=False)
                raise
            elapsed = time.perf_counter()-started
            result.update(call_id=call_id, kind=kind, category=category, metadata=copy.deepcopy(metadata),
                          cache_hit=False, charged=True, internal=internal,
                          certification_counts=certification, oracle_s=elapsed,
                          source_versions=copy.deepcopy(self.source_versions))
            record = {k: v for k, v in result.items() if k != 'plan'}
            record['status'] = 'complete'
            # A journal/IO failure leaves the debit pending; never append a
            # second terminal outcome for a partially written success record.
            self.ledger._finish(record, completed=True)
            return result

    def full(self, item, raw, category, metadata, cache_key=None):
        """Apply original H0, joint_search(passes=2), normalize and two checks.

        The caller performs raw policy completion before this function. Its
        measured time may be supplied as metadata['completion_s']; absent values
        remain None rather than inventing a zero-cost completion claim.
        """
        raw_values = list(raw)
        key_hash = _hash(cache_key) if cache_key is not None else None
        with self.ledger.lock:
            self.ledger.check()
            if key_hash is not None and key_hash in self.cache:
                cached = self.cache[key_hash]
                if cached['raw_route'] != raw_values:
                    raise ValueError('Explicit full-state cache key collided with another raw route')
                self.ledger._cache_hit(category, metadata, key_hash, cached['call_id'])
                result = copy.deepcopy(cached)
                result.update(cache_hit=True, charged=False, category=category,
                              metadata=copy.deepcopy(metadata), source_category=cached['category'],
                              source_call_id=cached['call_id'], internal=_empty_internal(),
                              certification_counts={k: 0 for k in CERTIFICATION}, oracle_s=0., H_s=0.,
                              export_validation_s=0., independent_validation_s=0.,
                              completion_s=metadata.get('completion_s'))
                return result

            def compute(certification):
                route = _route(raw_values)
                raw_metrics = _metrics(item, route, positive_gains=True)
                tick = time.perf_counter()
                h0 = baselines.postprocess(item, route)
                final = baselines.normalize(item, solver.joint_search(list(h0), item.C, item.D, passes=2))
                h_s = time.perf_counter()-tick
                final = _route(final)
                final_metrics = _metrics(item, final, positive_gains=True)
                if final_metrics['J'] > raw_metrics['J']+TOL:
                    raise ValueError('Complete H worsened raw objective beyond tolerance')
                tick = time.perf_counter()
                plan = self._export(item, final, 'r1_complete_H', certification)
                export_s = time.perf_counter()-tick
                tick = time.perf_counter()
                exported, independent, checked = self._verify(item, plan, final, certification)
                scalar_s = time.perf_counter()-tick
                return dict(**checked, cost=checked['J'], raw_J=raw_metrics['J'], raw_L=raw_metrics['L'],
                            raw_K=raw_metrics['K'], raw_route=route, route=final, final_route=final,
                            valid=True, plan=plan, export_validation=exported,
                            independent_validation=independent, H_s=h_s,
                            export_validation_s=export_s, independent_validation_s=scalar_s,
                            completion_s=metadata.get('completion_s'),
                            cache_key_sha256=key_hash,
                            plan_content_sha256=_hash(plan),
                            plan_hash_semantics='SHA256 of canonical UTF8 JSON content with sorted keys, compact separators; not output file bytes')

            result = self._call('full_H', category, metadata, compute)
            if key_hash is not None:
                # Label reuse needs costs, ordered routes and certificates, not
                # thousands of duplicated exported path/stop objects in RAM.
                self.cache[key_hash] = copy.deepcopy({k:v for k,v in result.items() if k != 'plan'})
            return result

    def native(self, item, method, category, metadata):
        """Keep original Fixed3/Detour16 pipeline, instrument and double-check."""
        def compute(certification):
            name = str(method).lower()
            if name not in ('fixed3', 'detour16'):
                raise ValueError(f'Unsupported native R1 reference: {method}')
            def export(problem, route, label):
                return self._export(item, route, label, certification)
            tick = time.perf_counter()
            with patch.object(_runtime, 'export_plan', export):
                plan, info = _runtime.run(item.base, 'fixed3' if name == 'fixed3' else 'detour',
                                          horizon=0 if name == 'fixed3' else 16)
            native_s = time.perf_counter()-tick
            route = [0]+[int(x)+1 for x in plan['visit_order_candidate_ids']]+[0]
            tick = time.perf_counter()
            exported, independent, checked = self._verify(item, plan, route, certification)
            scalar_s = time.perf_counter()-tick
            return dict(**checked, cost=checked['J'], raw_J=checked['J'], raw_L=checked['L'],
                        raw_K=checked['K'], raw_route=route, route=route, final_route=route,
                        raw_scope='native full-pipeline output, not an unobserved construction route',
                        valid=True, plan=plan, method=name, native_info=info,
                        fallback=bool(info.get('fallback_used', False)), native_s=native_s,
                        H_s=None, completion_s=None, export_validation=exported,
                        independent_validation=independent, independent_validation_s=scalar_s,
                        plan_content_sha256=_hash(plan),
                        plan_hash_semantics='SHA256 of canonical UTF8 JSON content with sorted keys, compact separators; not output file bytes')
        return self._call('native', category, metadata, compute)
