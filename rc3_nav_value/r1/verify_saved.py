"""Audit saved R1 evidence without generating or evaluating a route.

Only JSON/JSONL and saved checkpoint tensors are read. No project module is
imported; no scene construction, coverage/navigation preparation, policy
inference, solver, postprocessor or geometric validator is called. Run after
the registered experiment has stopped. The sole output is qa_saved.json.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SEED = 11412026
LIMITS = dict(init=4096, round=8192, audit=1024, validation=3072)
INTERNAL = ('construct', 'two_opt', 'joint_search', 'recipe')
CERTS = ('export_attempts', 'export_completed', 'scalar_attempts', 'scalar_completed')
GROUPS = dict(A='base', B='nav', C='base', D='nav')
INITIAL = ('warm_base', 'warm_nav', 'old128', 'prior', 'gBH', 'fixed3', 'detour16')
TIMING = ('A', 'B', 'C', 'D', 'warm_nav', 'old128', 'prior', 'fixed3', 'detour16')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def content_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def signature(value):
    """The frozen run.py root-key convention (default ensure_ascii=True)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def integer(x):
    return isinstance(x, int) and not isinstance(x, bool)


def close(a, b, abs_tol=1e-7, rel_tol=1e-8):
    return finite(a) and finite(b) and math.isclose(a, b, abs_tol=abs_tol, rel_tol=rel_tol)


def route(row, raw=False):
    names = ('raw_route', 'rawroute') if raw else ('final_route', 'finalroute', 'route')
    return next((row[n] for n in names if n in row), None)


def state_key(row):
    return (row['round'], row['group'], row['case'], tuple(row['prefix']))


class Audit:
    def __init__(self, out):
        self.out = Path(out).resolve()
        self.failures = []
        self.failure_count = 0
        self.check_count = 0
        self.sections = {}
        self.inputs = {}
        self.notes = {}
        self.p = {}
        self.scenes = {}
        self.scene_hashes = {}
        self.terminals = {}
        self.requests = []
        self.by_cache = {}
        self.checkpoints = {}
        self.quality = {}
        self.initial_labels = []
        self.round_labels = []
        self.promotion = {}
        self.resources = {}

    def check(self, condition, message):
        self.check_count += 1
        if not condition:
            self.failure_count += 1
            if len(self.failures) < 300:
                self.failures.append(message)
        return bool(condition)

    def file(self, path):
        path = Path(path)
        if not path.is_absolute():
            path = self.out/path
        if path.is_file():
            label = str(path.relative_to(self.out)) if path.is_relative_to(self.out) else str(path)
            self.inputs[label] = dict(sha256=digest(path), bytes=path.stat().st_size)
        return path

    def read(self, name, required=True):
        path = self.file(name)
        if not path.exists() and not required:
            return None
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(value, dict):
            raise ValueError(f'{name}: expected JSON object')
        return value

    def rows(self, name, optional=False):
        path = self.file(name)
        if not path.exists() and optional:
            return []
        result = []
        with path.open(encoding='utf-8-sig') as stream:
            for n, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError('expected object')
                    result.append(row)
                except (ValueError, TypeError) as exc:
                    self.check(False, f'{name}:{n}: {exc}')
        return result

    def section(self, name, fn):
        before = self.failure_count
        try:
            detail = fn()
        except Exception as exc:
            self.check(False, f'{name}: {type(exc).__name__}: {exc}')
            detail = {'aborted': True}
        self.sections[name] = dict(passed=self.failure_count == before,
                                   failures=self.failure_count-before, detail=detail)

    def registration(self):
        self.p = p = self.read('protocol.json')
        self.protocol_hash = digest(self.out/'protocol.json')
        receipt = self.read('registration_receipt.json')
        self.check(receipt['protocol_sha256'] == self.protocol_hash, 'registration protocol hash')
        self.check(receipt['base_commit'] == p['base_commit'], 'registration base commit')
        self.check(receipt['research_calls_before_registration'] == 0 and receipt['optimizer_updates'] == 0,
                   'registration must precede research calls and optimizer updates')
        self.check(p['training_seed'] == SEED, 'registered training seed')
        self.check(p['limits'] == {**LIMITS, 'total': 16384}, 'registered budget limits')
        for name, key in [('seeds.json', 'seeds_sha256'), ('seed_audit.json', 'seed_audit_sha256'),
                          ('seed_audit_inventory.jsonl', 'seed_inventory_sha256'),
                          ('seed_audit_boundary_sanity.json', 'seed_boundary_sha256')]:
            self.check(digest(self.file(name)) == p[key], f'{name}: registered raw file hash')
        for name, expected in p['source_sha256'].items():
            path = (ROOT/name).resolve()
            self.check(path.is_relative_to(ROOT), f'source pin outside workspace: {name}')
            if path.is_relative_to(ROOT):
                self.check(digest(self.file(path)) == expected, f'frozen source hash: {name}')
        self.check(signature(p['H_source_sha256']) == p['H_sha256'], 'H source manifest signature')
        for name, expected in p['H_source_sha256'].items():
            self.check(p['source_sha256'].get(name) == expected, f'H/source pin agreement: {name}')
        old = ROOT/'rc3_constructive/outputs/checkpoints/update0128.pt'
        self.check(digest(self.file(old)) == p['old128_sha256'], 'frozen old128 checkpoint bytes')
        started = self.read('execution_started.json')
        self.check(started['protocol_sha256'] == self.protocol_hash, 'execution protocol hash')
        audit = self.read('seed_audit.json')
        self.check(audit.get('verdict') == 'PASS_NO_RECORDED_PRIOR_USE', 'seed audit recorded-use verdict')
        inventory_hash = digest(self.out/'seed_audit_inventory.jsonl')
        # Raw inventory bytes and an optional canonical inventory digest have
        # deliberately different semantics; never equate the two digests.
        def inventory_pins(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == 'inventory_file_sha256':
                        yield v
                    else:
                        yield from inventory_pins(v)
        pins = list(inventory_pins(audit))
        self.check(bool(pins) and all(h == inventory_hash for h in pins), 'seed audit raw inventory hash')
        return dict(protocol_sha256=self.protocol_hash, frozen_sources=len(p['source_sha256']),
                    verifier_is_post_registration_tool='rc3_nav_value/r1/verify_saved.py' not in p['source_sha256'])

    def saved_scenes(self):
        seeds = self.read('seeds.json')
        expected = set()
        map_seeds, task_seeds = [], []
        for split, ms, ts in [('train', 8600000000, 8700000000), ('dev', 8800000000, 8900000000)]:
            self.check(len(seeds[split]) == 32, f'{split}: all32 seed entries')
            for case in range(32):
                name = f'scenes/{split}_{case:02}.json'
                expected.add(Path(name).name)
                spec = seeds[split][case]
                self.check(spec == dict(case=case, map_seed=ms+case, task_seed=ts+case,
                                        mode=('regular', 'mixed', 'blocked')[case % 3]), f'{split}/{case}: fixed seed pair')
                scene = self.read(name)
                self.scenes[split, case] = scene
                self.scene_hashes[split, case] = digest(self.out/name)
                self.check(scene['seed'] == ms+case and scene['task_seed'] == ts+case, f'{name}: saved seed identity')
                self.check(scene.get('complexity') == spec['mode'], f'{name}: scene mode')
                self.check([c['id'] for c in scene['candidates']] == list(range(len(scene['candidates']))),
                           f'{name}: candidate original-ID indexing')
                tids = [t['id'] for t in scene['tasks']]
                self.check(len(tids) == len(set(tids)), f'{name}: unique task IDs')
                map_seeds.append(scene['seed']); task_seeds.append(scene['task_seed'])
        files = set(p.name for p in (self.out/'scenes').glob('*.json'))
        self.check(files == expected, 'saved scenes: exactly train32/dev32, no extra scene files')
        self.check(len(set(map_seeds)) == 64 and len(set(task_seeds)) == 64, 'saved scene seed uniqueness')
        reserved = set(range(7700000000, 7700000060))
        self.check(not reserved.intersection(map_seeds+task_seeds), 'reserved60 seeds absent from saved scenes')
        named_reserved = []
        for path in self.out.rglob('*'):
            if path.is_file():
                values = [int(x) for x in re.findall(r'(?<![\d.])\d{10}(?![\d.])', path.name)]
                if reserved.intersection(values):
                    named_reserved.append(str(path.relative_to(self.out)))
        self.check(not named_reserved, 'reserved60 seeds absent from output filenames')
        return dict(saved_scenes=len(self.scenes), reserved_seed_filename_hits=named_reserved,
                    scope='Saved scene identity/filename audit; cannot prove absence of unsaved historical reads')

    def route_shape(self, value, scene, label):
        m = len(scene['candidates'])
        ok = (isinstance(value, list) and len(value) >= 2 and value[0] == value[-1] == 0
              and all(integer(x) for x in value)
              and all(integer(x) and 1 <= x <= m for x in value[1:-1])
              and len(set(value[1:-1])) == len(value)-2)
        return self.check(ok, f'{label}: closed unique candidate route')

    def certificates(self, row, label):
        self.check(row.get('valid') is True, f'{label}: valid flag')
        for name in ('export_validation', 'independent_validation'):
            report = row.get(name, {})
            self.check(report.get('valid_model_plan') is True and report.get('all_tasks_served') is True,
                       f'{label}: {name} flags')
            self.check(not report.get('errors') and not report.get('unserved_task_ids'), f'{label}: {name} errors/unserved')
        self.check(row.get('export_validation') == row.get('independent_validation'), f'{label}: two stored certificate reports agree')

    def versions(self, row, label):
        p = self.p
        expected = dict(H_sha256=p['H_sha256'], H_source_sha256=p['H_source_sha256'],
                        oracle_source_sha256=p['source_sha256']['rc3_nav_value/r1/oracle.py'],
                        runtime_sha256=p['source_sha256']['rc3_v4/runtime.py'],
                        scalar_validator_sha256=p['source_sha256']['rc3_baseline/validator.py'])
        self.check(row.get('source_versions') == expected, f'{label}: frozen H/runtime/validator versions')

    def terminal_metrics(self, row, label):
        split = 'dev' if row['category'] == 'validation' else 'train'
        case = row['metadata']['case']
        self.check(integer(case) and 0 <= case < 32, f'{label}: original case index')
        scene = self.scenes[split, case]
        self.route_shape(route(row), scene, label)
        self.route_shape(route(row, True), scene, label+' raw')
        self.check(row.get('route') == route(row), f'{label}: final route aliases')
        for prefix, rr in [('', route(row)), ('raw_', route(row, True))]:
            J, L, K = (row[prefix+n] for n in ('J', 'L', 'K'))
            self.check(finite(J) and J > 0 and finite(L) and L >= 0 and integer(K) and K >= 0,
                       f'{label}: {prefix}finite metrics')
            self.check(K == len(rr)-2 and close(J, L+.75*K), f'{label}: {prefix}J=L+.75K/stop count')
        self.check(close(row['cost'], row['J']), f'{label}: cost alias')
        if row['kind'] == 'full_H':
            self.check(row['J'] <= row['raw_J']+1e-7, f'{label}: complete H does not worsen raw objective')
        else:
            self.check(route(row, True) == route(row) and close(row['raw_J'], row['J']), f'{label}: native output raw scope')
        self.certificates(row, label)
        self.versions(row, label)
        self.check(isinstance(row.get('plan_content_sha256'), str) and len(row['plan_content_sha256']) == 64,
                   f'{label}: saved canonical plan content digest')

    def journal(self):
        records = self.rows('call_journal.jsonl')
        terminal = self.rows('terminal.jsonl')
        hits = self.rows('cache_hits.jsonl', optional=True)
        pending, starts, ends, terminal_by_id = {}, {}, {}, {}
        used = Counter({k: 0 for k in LIMITS})
        completed = Counter({k: 0 for k in LIMITS}); failed = completed.copy()
        icounts = Counter({k: 0 for k in INTERNAL})
        incl = {k: 0. for k in INTERNAL}; excl = dict(incl)
        cert = Counter({k: 0 for k in CERTS}); seconds = 0.
        journal_hits = []
        completed_seen = set()
        for n, row in enumerate(terminal):
            cid = row['call_id']
            self.check(cid not in terminal_by_id, f'terminal duplicate call_id {cid}')
            terminal_by_id[cid] = row
        for n, row in enumerate(records):
            event = row['event']; label = f'journal row{n+1}'
            if event == 'start':
                cid, cat = row['call_id'], row['category']
                self.check(cat in LIMITS and row['kind'] in ('full_H', 'native'), f'{label}: debit kind/category')
                self.check(integer(cid) and cid == len(starts)+1 and cid not in starts, f'{label}: contiguous unique predebit')
                self.check(not pending, f'{label}: serialized predebit ordering')
                starts[cid] = row; pending[cid] = row; used[cat] += 1
                self.check(used[cat] <= LIMITS.get(cat, 0) and sum(used.values()) <= 16384, f'{label}: budget at predebit')
            elif event in ('complete', 'error'):
                cid, cat = row['call_id'], row['category']
                start = pending.pop(cid, None)
                self.check(start is not None and start.get('category') == cat and start.get('kind') == row['kind'], f'{label}: matching prior debit')
                self.check(cid not in ends, f'{label}: unique terminal outcome')
                ends[cid] = row
                (completed if event == 'complete' else failed)[cat] += 1
                tr = terminal_by_id.get(cid, {})
                self.check(tr.get('status') == ('complete' if event == 'complete' else 'error'), f'{label}: terminal status pairing')
                for key in ('kind', 'category', 'internal', 'certification_counts', 'oracle_s'):
                    self.check(tr.get(key) == row.get(key), f'{label}: terminal/journal {key}')
                self.check(start is not None and tr.get('metadata') == start.get('metadata'), f'{label}: debit metadata preserved')
                self.check(finite(row['oracle_s']) and row['oracle_s'] >= 0, f'{label}: finite compute time')
                seconds += row['oracle_s']
                for k in INTERNAL:
                    v = row['internal']; count = v['counts'][k]; inc = v['inclusive_s'][k]; exc = v['exclusive_s'][k]
                    self.check(integer(count) and count >= 0 and finite(inc) and finite(exc) and 0 <= exc <= inc+1e-9,
                               f'{label}: internal {k} counts/timing')
                    icounts[k] += count; incl[k] += inc; excl[k] += exc
                self.check(sum(row['internal']['exclusive_s'].values()) <= row['oracle_s']+1e-6, f'{label}: exclusive internal time <= oracle time')
                for k in CERTS:
                    self.check(integer(row['certification_counts'][k]) and row['certification_counts'][k] >= 0,
                               f'{label}: certificate counter {k}')
                    cert[k] += row['certification_counts'][k]
                if event == 'complete':
                    completed_seen.add(cid)
                    self.check(row['certification_counts']['scalar_attempts'] == row['certification_counts']['scalar_completed'] == 1,
                               f'{label}: completed scalar certificate')
                    self.check(row['certification_counts']['export_attempts'] == row['certification_counts']['export_completed'] >= 1,
                               f'{label}: completed export certificate(s)')
                    if row['kind'] == 'full_H':
                        self.check(row['certification_counts']['export_completed'] == 1, f'{label}: full-H one export')
                    self.terminal_metrics(tr, f'terminal {cid}')
                    key = tr.get('cache_key_sha256')
                    if key is not None:
                        self.check(key not in self.by_cache, f'terminal {cid}: explicit key charged only once')
                        self.by_cache[key] = tr
                        self.requests.append(dict(key=key, category=cat, metadata=tr['metadata'], source_call_id=cid, cache_hit=False))
                else:
                    self.check(tr.get('error') == row.get('error'), f'{label}: error preserved and charged')
            elif event == 'cache_hit':
                journal_hits.append(row)
                cid = row['source_call_id']; source = terminal_by_id.get(cid, {})
                self.check(cid in completed_seen and source.get('kind') == 'full_H', f'{label}: cache source already completed full-H')
                self.check(source.get('cache_key_sha256') == row['cache_key_sha256'], f'{label}: source cache key')
                self.check(row['category'] in LIMITS, f'{label}: cache request category')
                self.requests.append(dict(key=row['cache_key_sha256'], category=row['category'], metadata=row['metadata'], source_call_id=cid, cache_hit=True))
            else:
                self.check(False, f'{label}: unknown event {event}')
        self.check(not pending, 'journal: no charged incomplete calls')
        self.check(set(starts) == set(ends) == set(terminal_by_id), 'journal/terminal exact debit/outcome pairing')
        self.check(journal_hits == hits, 'cache hit journal byte-content objects/order agree')
        self.terminals = terminal_by_id
        expected = dict(limits=LIMITS, total_limit=16384, used=dict(used), started=dict(used),
                        completed=dict(completed), failed=dict(failed), total_started=sum(used.values()),
                        total_completed=sum(completed.values()), total_failed=sum(failed.values()),
                        cache_hits=len(hits), internal_counts=dict(icounts), internal_inclusive_s=incl,
                        internal_exclusive_s=excl, certification_counts=dict(cert), oracle_s=seconds)
        summary = self.read('ledger_summary.json')
        self.resources = resources = self.read('resources.json')
        for label, saved in [('ledger_summary', summary), ('resources.ledger', resources['ledger'])]:
            for k, value in expected.items():
                if k in ('internal_inclusive_s', 'internal_exclusive_s'):
                    self.check(set(saved[k]) == set(value) and all(close(saved[k][n], v, 1e-6) for n, v in value.items()), f'{label}: {k} aggregate')
                elif k == 'oracle_s':
                    self.check(close(saved[k], value, 1e-6), f'{label}: compute-time aggregate')
                else:
                    self.check(saved.get(k) == value, f'{label}: {k} aggregate')
        self.check(sum(failed.values()) == 0, 'completed R1 must have zero failed calls (failed calls still charged)')
        return dict(**{k: expected[k] for k in ('used', 'total_started', 'total_completed', 'total_failed', 'cache_hits')},
                    internal_counts=dict(icounts), certification_counts=dict(cert), pending=sorted(pending))

    def saved_checkpoints(self):
        # Safe tensor-only loading; no network, optimizer or policy is created.
        import torch
        names = {'warm_base': 1024, 'warm_nav': 1024}
        names.update({f'{g}_round{r}': 1024+r*512 for g in GROUPS for r in (1, 2)})
        actual = {p.stem for p in (self.out/'checkpoints').glob('*.pt')}
        self.check(actual == set(names), 'checkpoints: exactly two warm/eight round endpoints')
        structures, configs, counts = [], [], []
        for name, steps in names.items():
            path = self.file(f'checkpoints/{name}.pt')
            payload = torch.load(path, map_location='cpu', weights_only=True)
            state = payload['state']; h = hashlib.sha256(); structure = []; count = 0
            self.check(isinstance(state, dict) and bool(state), f'{name}: tensor state dict')
            for key, tensor in sorted(state.items()):
                self.check(isinstance(key, str) and torch.is_tensor(tensor), f'{name}/{key}: tensor-only state')
                value = tensor.detach().cpu().contiguous()
                self.check(bool(torch.isfinite(value).all()), f'{name}/{key}: finite tensor')
                h.update(key.encode()); h.update(value.numpy().tobytes())
                structure.append((key, str(value.dtype), list(value.shape))); count += value.numel()
            actual_hash = h.hexdigest()
            self.check(payload['tensor_sha256'] == actual_hash, f'{name}: independently recomputed tensor hash')
            self.check(payload['optimizer_steps_including_inherited'] == steps, f'{name}: inherited logical update count')
            self.check(payload['protocol_sha256'] == self.protocol_hash and payload['seed'] == SEED, f'{name}: protocol/seed')
            representation = 'base' if name.startswith(('A', 'C', 'warm_base')) else 'nav'
            self.check(payload['representation'] == representation, f'{name}: representation tag')
            self.check('optimizer' not in payload and 'optimizer_state_dict' not in payload,
                       f'{name}: checkpoint optimizer-state scope changed; review verifier limitation')
            self.checkpoints[name] = dict(tensor_sha256=actual_hash, parameter_count=count, representation=representation,
                                          model_config=payload['model_config'])
            structures.append(structure); configs.append(payload['model_config']); counts.append(count)
        self.check(all(x == structures[0] for x in structures), 'all groups: identical tensor keys/shapes/dtypes')
        self.check(all(x == configs[0] for x in configs), 'all groups: identical architecture config')
        self.check(all(x == counts[0] for x in counts), 'all groups: identical parameter count')
        self.check(configs[0]['hidden'] == 64 and configs[0]['layers'] == 2, 'registered hidden64/two layers')
        self.notes['reference_hashes'] = {n: v['tensor_sha256'] for n, v in self.checkpoints.items()}
        self.notes['cross_representation_reference_hash_equal'] = {
            'warm': self.checkpoints['warm_base']['tensor_sha256'] == self.checkpoints['warm_nav']['tensor_sha256'],
            'round1': self.checkpoints['C_round1']['tensor_sha256'] == self.checkpoints['D_round1']['tensor_sha256']}
        return dict(count=len(names), parameter_count=counts[0], torch=str(torch.__version__),
                    checkpoint_optimizer_state_saved=False)

    def quality_rows(self):
        rows = self.rows('quality.jsonl')
        expected = set()
        for phase, methods in [('initial', INITIAL), ('round1', tuple(GROUPS)), ('round2', tuple(GROUPS))]:
            expected.update((phase, m, c, None, None) for m in methods for c in range(32))
        expected.update(('timing', m, c, mode, rep) for m in TIMING for c in range(32)
                        for mode in ('cold', 'hot') for rep in range(3))
        seen_ids = set()
        compare_keys = ('J', 'L', 'K', 'raw_J', 'raw_L', 'raw_K', 'raw_route', 'route', 'final_route',
                        'valid', 'plan_content_sha256', 'source_versions', 'internal', 'certification_counts',
                        'export_validation', 'independent_validation', 'oracle_s', 'kind', 'category')
        for n, row in enumerate(rows, 1):
            key = (row['phase'], row['method'], row['case'], row.get('cachemode'), row.get('rep'))
            self.check(integer(row['case']) and 0 <= row['case'] < 32, f'quality {n}: original case index')
            self.check(row['phase'] != 'timing' or (integer(row.get('rep')) and 0 <= row['rep'] < 3), f'quality {n}: timing repetition index')
            self.check(key not in self.quality, f'quality {n}: unique grid cell')
            self.quality[key] = row
            self.check(key in expected, f'quality {n}: registered grid cell')
            cid = row['call_id']; terminal = self.terminals.get(cid, {})
            self.check(cid not in seen_ids, f'quality {n}: unique billed request')
            seen_ids.add(cid)
            self.check(terminal.get('status') == 'complete' and row.get('charged') is True and row.get('cache_hit') is False,
                       f'quality {n}: completed uncached billed request')
            for k in compare_keys:
                self.check(row.get(k) == terminal.get(k), f'quality {n}: terminal {k}')
            metadata = {k: row.get(k) for k in ('case', 'phase', 'method', 'cachemode', 'rep', 'scene_sha256')}
            self.check(terminal.get('metadata') == metadata, f'quality {n}: journal request metadata')
            self.check(row['scene_sha256'] == self.scene_hashes['dev', row['case']], f'quality {n}: saved scene hash')
            self.check(row['category'] == 'validation', f'quality {n}: validation debit')
            self.check(row['kind'] == ('native' if row['method'] in ('fixed3', 'detour16') else 'full_H'), f'quality {n}: native/common-H scope')
            for t in ('oracle_s', 'preparation_s', 'total_s', 'measured_request_wall_s'):
                self.check(finite(row[t]) and row[t] >= 0, f'quality {n}: finite {t}')
            decision = row.get('decision_s', 0.)
            self.check(finite(decision) and decision >= 0, f'quality {n}: decision time')
            self.check(close(row['total_s'], row['preparation_s']+decision+row['oracle_s']), f'quality {n}: complete time arithmetic')
            self.check(close(row['bookkeeping_and_timer_residual_s'], row['measured_request_wall_s']-row['total_s']), f'quality {n}: observed wall residual')
        self.check(set(self.quality) == expected and len(rows) == 2208, 'quality: complete unique2208 grid (224+256+1728)')
        validation_ids = {cid for cid, r in self.terminals.items() if r['category'] == 'validation'}
        self.check(seen_ids == validation_ids, 'quality: every validation call represented exactly once')
        for method in TIMING:
            for case in range(32):
                endpoint = ('round2', method, case, None, None) if method in GROUPS else ('initial', method, case, None, None)
                base = self.quality[endpoint]
                for mode in ('cold', 'hot'):
                    for rep in range(3):
                        current = self.quality['timing', method, case, mode, rep]
                        self.check(all(current[k] == base[k] for k in ('J', 'L', 'K', 'raw_route', 'final_route')),
                                   f'timing/{case}/{method}/{mode}/{rep}: endpoint route and cost deterministic')
        return dict(rows=len(rows), phases=dict(Counter(r['phase'] for r in rows)))

    def saved_plans(self):
        expected = set(); verified = 0
        for key, row in self.quality.items():
            phase, method, case, mode, rep = key
            if phase == 'timing' and (mode != 'cold' or rep != 0):
                continue
            name = f'{phase}_{case:02}_{method}.json'; expected.add(name)
            plan = self.read('plans/'+name); scene = self.scenes['dev', case]
            self.check(content_hash(plan) == row['plan_content_sha256'], f'{name}: canonical content hash (not file bytes)')
            order = plan['visit_order_candidate_ids']; selected = plan['selected_candidate_ids']
            self.check([0]+[j+1 for j in order]+[0] == route(row), f'{name}: exported route order')
            self.check(len(set(order)) == len(order) and set(selected) == set(order) and len(selected) == len(set(selected)), f'{name}: selected/order sets')
            points = plan['route_polyline']; legs = plan['route_leg_costs_m']
            goodpoints = isinstance(points, list) and bool(points) and all(isinstance(v, list) and len(v) == 2 and all(finite(x) for x in v) for v in points)
            self.check(goodpoints, f'{name}: finite saved polyline')
            self.check(all(finite(x) and x >= 0 for x in legs) and len(legs) == len(order)+1, f'{name}: finite leg costs/count')
            length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
            cumulative = [0.]
            for a, b in zip(points, points[1:]):
                cumulative.append(cumulative[-1]+math.dist(a, b))
            self.check(close(length, plan['route_length_m']) and close(length, sum(legs)) and close(length, row['L']), f'{name}: polyline/leg/metric length arithmetic')
            self.check(close(plan['stop_penalty_equivalent_m'], .75) and close(plan['objective_length_plus_stops_m'], length+.75*len(order))
                       and close(plan['objective_length_plus_stops_m'], row['J']), f'{name}: saved objective arithmetic')
            self.check(math.dist(points[0], scene['start']) < 1e-8 and math.dist(points[-1], scene['start']) < 1e-8, f'{name}: closed saved polyline')
            docks = {x['id']: x for x in scene['candidates']}; taskids = {x['id'] for x in scene['tasks']}
            assignments = plan['assignments']; assigned = [a['task_id'] for a in assignments]
            self.check(len(assigned) == len(set(assigned)) and set(assigned) == taskids, f'{name}: all task IDs assigned exactly once')
            self.check(not plan['unserved_task_ids'] and plan.get('status') == 'complete', f'{name}: complete bookkeeping')
            self.check(all(a['candidate_id'] in selected and a['arm'] in ('L', 'R') for a in assignments), f'{name}: assignment dock/arm identifiers')
            stops = plan['service_stops']; arrival = 0.; cursor = 0
            self.check([s['candidate_id'] for s in stops] == order, f'{name}: saved service order')
            for i, j in enumerate(order):
                stop = stops[i]; arrival += legs[i]; dock = docks[j]
                self.check(stop['sequence'] == i+1 and close(stop['arrival_distance_m'], arrival), f'{name}: service {i} arrival')
                self.check(all(close(stop['pose'][k], dock[k], 1e-8, 0) for k in ('x', 'y', 'theta', 'h')), f'{name}: service {i} pose')
                self.check(sorted((a['task_id'], a['arm']) for a in stop['tasks']) == sorted((a['task_id'], a['arm']) for a in assignments if a['candidate_id'] == j), f'{name}: service {i} task bookkeeping')
                positions = [n for n in range(cursor, len(points)) if close(cumulative[n], arrival)]
                self.check(bool(positions), f'{name}: service {i} saved leg endpoint')
                if positions:
                    cursor = positions[0]
                    self.check(math.dist(points[cursor], [dock['x'], dock['y']]) < 1e-8, f'{name}: service {i} saved endpoint coordinate')
            self.check(plan['validation'] == row['export_validation'] == row['independent_validation'], f'{name}: stored double certificate identity')
            verified += 1
        actual = {p.name for p in (self.out/'plans').glob('*.json')}
        self.check(actual == expected and len(expected) == 768, 'saved plans: exactly768 expected deployment certificates')
        return dict(plans_verified=verified, scope='Arithmetic and identifiers only; no geometric/F/D recomputation')

    def cache_key(self, scene_hash, prefix, action, reference):
        root_key = signature(dict(scene=scene_hash, prefix=prefix, action=action,
                                  reference=reference, H=self.p['H_sha256']))
        return content_hash(root_key)

    def request_matches(self, row, action, label):
        key = self.cache_key(row['scene_sha256'], row['prefix'], action, row['reference_hash'])
        source = self.by_cache.get(key)
        self.check(source is not None, f'{label}: exact scene/prefix/action/reference/H cache-key source')
        if source is None:
            return None, key
        raw = route(source, True); p = row['prefix']
        self.check(raw[1:1+len(p)] == p and len(raw) > len(p)+2 and raw[len(p)+1] == action,
                   f'{label}: saved raw route preserves ordered prefix and forced action')
        self.check(source['category'] != 'validation', f'{label}: training/audit cache source scope')
        return source, key

    def labels(self):
        self.initial_labels = initial = self.rows('labels_initial.jsonl')
        self.round_labels = rounds = self.rows('labels_round.jsonl')
        promotion = self.read('promotion.json'); self.promotion = promotion
        native_rows = [r for r in self.terminals.values()
                       if r['category'] == 'init' and r['kind'] == 'native' and r['metadata'].get('tag') == 'train_referenceB']
        native = {r['metadata']['case']: r for r in native_rows}
        self.check(set(native) == set(range(32)) and len(native_rows) == 32, 'initial teacher:32 native Fixed3 references')
        self.check(all(r.get('method') == 'fixed3' for r in native_rows), 'initial teacher references retain native Fixed3 pipeline')
        self.teacher_call_ids = {r['call_id'] for r in native_rows}
        observed = Counter(); expected_requests = Counter(); grouped = defaultdict(dict)
        for n, rec in enumerate(initial+rounds):
            iswarm = n < len(initial); tag = 'warm' if iswarm else f'round{rec["round"]}_{rec["group"]}'
            cat = 'init' if iswarm else 'round'; case = rec['case']; label = f'{tag}/{case}/state{n}'
            p = rec['prefix']; actions = rec['actions']; ys = rec['Y']; targets = rec['target']
            scene = self.scenes['train', case]; m = len(scene['candidates'])
            self.check(integer(case) and 0 <= case < 32, f'{label}: case')
            self.check(isinstance(p, list) and len(p) == len(set(p)) and all(integer(a) and 1 <= a <= m for a in p), f'{label}: unique ordered prefix IDs')
            self.check(1 <= len(actions) <= 8 and len(actions) == len(set(actions)) and all(integer(a) and 1 <= a <= m and a not in p for a in actions), f'{label}: queried action IDs/count')
            self.check(len(ys) == len(actions) == len(targets) and all(finite(y) and y > 0 for y in ys), f'{label}: queried costs/target lengths')
            self.check(close(rec['Yref'], ys[0]), f'{label}: first reference action/target-zero')
            self.check(all(close(t, (rec['Yref']-y)/max(rec['Yref'], 1e-8), 1e-10) for t, y in zip(targets, ys)), f'{label}: independently recomputed normalized Delta')
            self.check(rec['scene_sha256'] == self.scene_hashes['train', case], f'{label}: scene bytes hash')
            if iswarm:
                expected_hash = signature(route(native[case])); refname = 'gB'
                stateid = (case, tuple(p)); bucket = ('warm',)
            else:
                g, r = rec['group'], rec['round']; representation = GROUPS[g]
                promoted = r == 2 and g in ('C', 'D') and promotion[representation]['promoted']
                name = f'{g}_round1' if promoted else 'warm_'+representation
                expected_hash = self.checkpoints[name]['tensor_sha256']
                refname = representation+('_g1' if promoted else '_g0')
                stateid = (case, tuple(p)); bucket = (r, g)
            self.check(rec['reference_hash'] == expected_hash and rec['reference'] == refname, f'{label}: frozen reference hash/name')
            self.check(stateid not in grouped[bucket], f'{label}: unique actual state')
            grouped[bucket][stateid] = rec
            for a, y in zip(actions, ys):
                source, key = self.request_matches(rec, a, label)
                if source is not None:
                    self.check(close(source['J'], y, 1e-10), f'{label}/action{a}: label equals completed terminal cost')
                expected_requests[key, cat, tag, case] += 1
            observed[bucket] += len(actions)
        self.check(0 < len(initial) <= 508 and {r['case'] for r in initial} == set(range(32)), 'initial: <=508 states and all32 primary cases')
        for case in range(32):
            self.check(1 <= sum(r['case'] == case for r in initial) <= (16 if case < 28 else 15), f'initial case{case}: actual threshold-state bound')
        for r in (1, 2):
            for g in GROUPS:
                data = grouped[r, g]
                self.check(1 <= len(data) <= 128 and {key[0] for key in data} == set(range(32)), f'round{r}/{g}: <=128 states and all32 cases')
                self.check(all(1 <= sum(key[0] == case for key in data) <= 4 for case in range(32)), f'round{r}/{g}: actual-state per-scene cap4')
            for fixed, api in [('A', 'C'), ('B', 'D')]:
                a, b = grouped[r, fixed], grouped[r, api]
                self.check(set(a) == set(b), f'round{r}/{fixed}={api}: actual prefix pools matched')
                for key in set(a).intersection(b):
                    aa, bb = a[key], b[key]
                    self.check(set(aa['actions']) == set(bb['actions']), f'round{r}/{fixed}={api}/{key}: queried sets matched')
                    if r == 1 or not promotion[GROUPS[fixed]]['promoted']:
                        self.check(aa['reference_hash'] == bb['reference_hash'], f'round{r}/{fixed}={api}/{key}: matched reference')
                        self.check(dict(zip(aa['actions'], aa['Y'])) == dict(zip(bb['actions'], bb['Y'])), f'round{r}/{fixed}={api}/{key}: matched exact queried costs')
                        self.check(dict(zip(aa['actions'], aa['target'])) == dict(zip(bb['actions'], bb['target'])), f'round{r}/{fixed}={api}/{key}: matched Delta')
        actual_requests = Counter()
        for req in self.requests:
            meta = req['metadata']; tag = meta.get('tag')
            if tag == 'warm' or (isinstance(tag, str) and re.fullmatch(r'round[12]_[ABCD]', tag)):
                actual_requests[req['key'], req['category'], tag, meta['case']] += 1
                self.check(meta.get('scene_sha256') == self.scene_hashes['train', meta['case']], 'label cache request: scene hash')
                self.check(req['key'] == self.cache_key(meta['scene_sha256'], meta['prefix'], meta['action'], meta['reference_tensor_sha256']), 'label cache request: exact material key')
                source = self.terminals[req['source_call_id']]
                raw = route(source, True); prefix = meta['prefix']
                self.check(raw[1:1+len(prefix)] == prefix and raw[len(prefix)+1] == meta['action'], 'label cache request: source raw ordered prefix/action')
        self.check(actual_requests == expected_requests, 'labels: every action use paired to exactly one completed call/cache-hit request')
        collection = self.read('initial_collection.json')
        self.check(collection['states'] == len(initial) and collection['menus_upper'] == 508 and collection['label_uses'] == sum(len(r['actions']) for r in initial), 'initial collection counts')
        uses = sum(len(r['actions']) for r in initial+rounds)
        self.check(self.resources['stats']['label_uses'] == uses, 'resources label uses exclude audit/quality and include all cache hits')
        self.label_groups = grouped
        return dict(initial_states=len(initial), round_states={f'round{r}_{g}': len(grouped[r, g]) for r in (1, 2) for g in GROUPS},
                    label_action_uses=uses, cache_key_scope='full scene bytes/ordered prefix/action/reference tensor/H; raw-route equality is additionally guarded by frozen oracle source')

    def audit_rows(self):
        rows = self.rows('audit.jsonl'); labels = {state_key(r): r for r in self.round_labels}
        seen = set(); expected_requests = Counter(); unknown = 0
        for n, row in enumerate(rows, 1):
            key = state_key(row); label = f'audit{n}'
            self.check(key in labels and key not in seen, f'{label}: unique actual labelled training state')
            seen.add(key); rec = labels[key]
            self.check(row['queried'] == rec['actions'] and row['queried_target'] == rec['target'] and row['reference_hash'] == rec['reference_hash'], f'{label}: frozen labels/reference')
            self.check(close(row['Yref'], rec['Yref']) and close(row['known_best'], min(rec['Y'])), f'{label}: reference/known best')
            action = row['selected']; unseen = action not in rec['actions']
            self.check(row['unknown'] is unseen and row['requested_new_audit'] is unseen, f'{label}: unknown-query accounting')
            self.check(integer(action) and action not in rec['prefix'] and 1 <= action <= len(self.scenes['train', row['case']]['candidates']), f'{label}: selected candidate identifier')
            self.check(finite(row['predicted']) and len(row['queried_prediction']) == len(rec['actions']) and all(finite(x) for x in row['queried_prediction']), f'{label}: finite stored predictions')
            if not unseen:
                pos = rec['actions'].index(action)
                self.check(close(row['Yselected'], rec['Y'][pos], 1e-10) and close(row['predicted'], row['queried_prediction'][pos]), f'{label}: queried selected action reused exactly')
            else:
                unknown += 1
                source, ckey = self.request_matches(rec, action, label)
                if source:
                    self.check(close(row['Yselected'], source['J'], 1e-10), f'{label}: unknown selected terminal cost')
                expected_requests[ckey, row['group'], row['round'], row['case']] += 1
        self.check(seen == set(labels), 'audit: one opportunity per actual round-labelled state, no upper-cap fill required')
        actual_requests = Counter()
        for req in self.requests:
            meta = req['metadata']
            if meta.get('tag') == 'actual_unknown_argmax':
                self.check(req['category'] == 'audit', 'unknown audit request category')
                actual_requests[req['key'], meta['group'], meta['round'], meta['case']] += 1
        self.check(actual_requests == expected_requests, 'unknown audit requests exactly match completed calls/cache hits')
        known_requests = sum((r['metadata'].get('tag') == 'warm' or re.fullmatch(r'round[12]_[ABCD]', str(r['metadata'].get('tag', ''))) is not None
                              or r['metadata'].get('tag') == 'actual_unknown_argmax') for r in self.requests)
        self.check(known_requests == len(self.requests), 'every explicit-key call/cache hit belongs to recorded labels or audit')
        attributed = (self.teacher_call_ids
                      | {cid for cid, r in self.terminals.items() if r['category'] == 'validation'}
                      | {r['source_call_id'] for r in self.requests})
        self.check(attributed == set(self.terminals), 'every billed terminal call attributed to native teacher, labels, audit or quality')
        return dict(opportunities=len(rows), unknown_selected_requests=unknown,
                    scope='Action generalization on saved training states; predictions/masks cannot be replayed without forbidden model/F/D work')

    def optimization(self):
        rows = self.rows('training.jsonl'); groups = defaultdict(list)
        expected = {'warm_base': 1024, 'warm_nav': 1024}
        expected.update({f'round{r}_{g}': 512 for r in (1, 2) for g in GROUPS})
        for n, row in enumerate(rows, 1):
            groups[row['tag']].append(row)
            self.check(row['tag'] in expected and integer(row['case']) and 0 <= row['case'] < 32, f'training{n}: tag/case')
            self.check(all(finite(row[k]) and row[k] >= -1e-12 for k in ('loss', 'regression', 'ranking', 'mse')), f'training{n}: finite nonnegative metrics')
            self.check(close(row['loss'], row['regression']+.1*row['ranking'], 1e-8, 2e-6), f'training{n}: regression+.1ranking')
        self.check(set(groups) == set(expected) and len(rows) == 6144, 'training:6144 actual optimizer-step rows')
        ordered_tags = ['warm_base', 'warm_nav']+[f'round{r}_{g}' for r in (1, 2) for g in GROUPS]
        self.check([r['tag'] for r in rows] == [tag for tag in ordered_tags for _ in range(expected[tag])], 'training: registered warm/round/group execution order')
        for tag, count in expected.items():
            rr = groups[tag]
            self.check([r['step'] for r in rr] == list(range(1, count+1)), f'{tag}: exact contiguous step count')
            # Scene selection uses an independent identical schedule for the
            # matched groups. Reproduce only RNG indices, never model work.
            rng = np.random.default_rng(SEED if tag.startswith('warm') else SEED+int(tag[5]))
            pool = self.initial_labels if tag.startswith('warm') else self.label_groups[int(tag[5]), tag[-1]].values()
            sizes = Counter(r['case'] for r in pool)
            cases = []
            for _ in range(count):
                case = int(rng.choice(list(range(32)))); cases.append(case)
                rng.integers(sizes[case], size=8)
            self.check([r['case'] for r in rr] == cases, f'{tag}: registered scene RNG sequence with eight index draws per step')
        stats = self.resources['stats']; calibration = self.read('calibration.json')
        self.check(stats['optimizer_updates'] == 6144 and stats['backward_probe_steps'] == 32, 'resource optimizer/probe counts')
        self.check(stats.get('training_or_probe_forward_batches') == 6176 and stats.get('training_or_probe_forward_rows') == 6176*8, 'training/probe forward batch/row counts')
        self.check(calibration['probe_steps'] == 32 and calibration['optimizer_updates'] == 0 and len(calibration['step_s']) == 32 and len(calibration['raw_completion_s']) == 6, 'calibration probes/zero optimizer updates')
        self.check(calibration['stats_before_optimizer']['optimizer_updates'] == 0 and calibration['stats_before_optimizer']['backward_probe_steps'] == 32, 'calibration pre-optimizer snapshot')
        self.check(len(calibration['prefix_checks']) == 6 and all(finite(r['prefix_equivalence_maxabs']) and 0 <= r['prefix_equivalence_maxabs'] < 1e-5 for r in calibration['prefix_checks']), 'stored full/incremental prefix probe tolerance')
        self.check(all(finite(x) and x > 0 for x in calibration['step_s']+calibration['raw_completion_s']), 'finite positive calibration samples')
        wall = min(7200, max(1800, math.ceil(4*(6144*max(calibration['step_s'])+12288*calibration['max_initial_H_s']
                                                     +1728*calibration['max_prepare_s']+4096*max(calibration['raw_completion_s']))+120)))
        self.check(wall == calibration['wall_limit_s'] == self.resources['wall_limit_s'], 'wall limit: exact registered calibration formula')
        initial_H = [r['oracle_s'] for r in self.terminals.values() if r['kind'] == 'full_H' and r['category'] == 'init']
        self.check(close(max(initial_H), calibration['max_initial_H_s']), 'calibration max initial full-H compute time')
        self.check(self.resources['stats']['scenes'] == 64 and self.resources['stats']['prepare_calls'] == 1824, 'scene/preparation request counts64/1824')
        self.check(calibration['parameter_count'] == self.checkpoints['warm_base']['parameter_count'] and calibration['architecture'] == self.checkpoints['warm_base']['model_config'], 'calibration/checkpoint parameter count/config')
        self.check(self.resources['logical_updates_per_configuration'] == 2048, 'logical2048 per configuration vs actual6144')
        return dict(actual_optimizer_steps=len(rows), steps_by_tag={k: len(v) for k, v in groups.items()},
                    logical_updates_per_configuration=2048, logical_sum_including_inherited_warm=8192,
                    wall_limit_s=wall, optimizer_state_tensor_audit_available=False)

    def promotion_and_invariants(self):
        promotions = self.read('promotion.json')
        for representation, api in [('base', 'C'), ('nav', 'D')]:
            initial = [self.quality['initial', 'warm_'+representation, i, None, None] for i in range(32)]
            candidate = [self.quality['round1', api, i, None, None] for i in range(32)]
            base = np.array([r['J'] for r in initial]); cand = np.array([r['J'] for r in candidate])
            values = 100*(base-cand)/base
            samples = np.random.default_rng(SEED).choice(values, size=(20000, 32), replace=True).mean(1)
            low, high = np.quantile(samples, [.025, .975])
            promote = bool(values.mean() >= .5 and low > 0 and all(r['valid'] for r in initial+candidate))
            saved = promotions[representation]
            self.check(saved['promoted'] is promote and close(saved['mean_gain_percent'], float(values.mean()), 1e-10)
                       and all(close(x, float(y), 1e-10) for x, y in zip(saved['CI95'], [low, high])) and len(saved['CI95']) == 2,
                       f'{representation}: independently recomputed all32 promotion rule/bootstrap')
            self.check(saved['candidate_tensor_hash'] == self.checkpoints[f'{api}_round1']['tensor_sha256']
                       and saved['reference_tensor_hash'] == self.checkpoints['warm_'+representation]['tensor_sha256'], f'{representation}: promotion tensor identities')
        for r in (1, 2):
            saved = self.read(f'round{r}_invariants.json')
            self.check(saved['passed'] is True, f'round{r}: stored invariant status')
            for fixed, api in [('A', 'C'), ('B', 'D')]:
                same = self.checkpoints[f'{fixed}_round{r}']['tensor_sha256'] == self.checkpoints[f'{api}_round{r}']['tensor_sha256']
                self.check(saved['pairs'][f'{fixed}={api}'] is same, f'round{r}/{fixed}={api}: exact tensor identity record')
                required = r == 1 or not promotions[GROUPS[fixed]]['promoted']
                if required:
                    self.check(same, f'round{r}/{fixed}={api}: required exact tensor invariant')
                    for i in range(32):
                        aa = self.quality[f'round{r}', fixed, i, None, None]; bb = self.quality[f'round{r}', api, i, None, None]
                        self.check(all(aa[k] == bb[k] for k in ('J', 'raw_route', 'final_route')), f'round{r}/{fixed}={api}/{i}: required exact deployment invariant')
        return dict(promoted={k: v['promoted'] for k, v in promotions.items()},
                    bootstrap_resamples=20000, bootstrap_seed=SEED, development_intervals='exploratory; same dev32 used for promotion and screening')

    def completion(self):
        r = self.resources
        self.check(r['status'] == 'completed' and not (self.out/'failure.json').exists(), 'full-run completed status without failure receipt')
        self.check(r['retained_test_queries'] == 0 and r['R2_started'] is False and r['R3_started'] is False, 'declared retained-test/R2/R3 boundary')
        self.check(finite(r['elapsed_s']) and finite(r['post_calibration_elapsed_s']) and 0 <= r['post_calibration_elapsed_s'] <= r['elapsed_s'], 'finite total/post-calibration elapsed times')
        # A final atomic solver/optimizer request can finish slightly beyond
        # the deadline: the source checks the fixed deadline between requests.
        self.notes['post_calibration_overrun_s'] = max(0., r['post_calibration_elapsed_s']-r['wall_limit_s'])
        return dict(status=r['status'], elapsed_s=r['elapsed_s'], post_calibration_elapsed_s=r['post_calibration_elapsed_s'],
                    retained_test_queries_declared=r['retained_test_queries'])

    def run(self):
        for name, fn in [('registration', self.registration), ('saved_scenes', self.saved_scenes),
                         ('call_accounting', self.journal), ('checkpoint_tensors', self.saved_checkpoints),
                         ('quality_grid', self.quality_rows), ('saved_plan_arithmetic', self.saved_plans),
                         ('labels_and_cache', self.labels), ('audit_opportunities', self.audit_rows),
                         ('optimization_accounting', self.optimization),
                         ('reference_promotion_and_invariants', self.promotion_and_invariants), ('completion', self.completion)]:
            self.section(name, fn)
        return dict(schema='rc3-r1-saved-evidence-qa-v1', created_unix=time.time(),
                    all_passed=self.failure_count == 0, verdict='PASS_SAVED_EVIDENCE_CONSISTENT' if not self.failure_count else 'FAIL_OR_INCOMPLETE_SAVED_EVIDENCE',
                    checks=self.check_count, failure_count=self.failure_count, failures=self.failures,
                    failures_truncated=self.failure_count > len(self.failures), sections=self.sections,
                    notes=self.notes, input_files=self.inputs, verifier_sha256=digest(__file__),
                    no_research_calls=True, no_model_instantiation=True,
                    scope='Saved evidence consistency and arithmetic only, not an independent geometric feasibility or performance experiment',
                    limitations=[
                        'No scenes are generated, no C/F/D or action masks are recomputed, and no route is replanned or geometrically revalidated. Feasibility relies on the two version-pinned stored certificates.',
                        'Raw label-route length/coverage and all-menu argmax cannot be independently recomputed from the compact journal. Ordered prefix/action, Delta, complete-call cost and cache provenance are checked.',
                        'Only deployment plans selected by the frozen saver have complete polylines; labels retain routes/metrics/certificate summaries and canonical plan hashes without complete plan payloads.',
                        'Adam state is held by the frozen runtime but is not saved. Step logs, inherited checkpoint metadata, matched RNG scene schedules and exact endpoint tensor identities are audited; Adam moment/step tensors cannot be audited.',
                        'Training logs save each selected scene, not the eight individual sampled label indices. Index draws are replayed only to advance the registered RNG and verify the saved scene sequence.',
                        'Equal base/nav reference tensor hashes are permitted. Frozen Oracle guards exact raw-route equality on explicit-key hits; this audit verifies each saved source prefix/action and every request-key mapping but cannot recover an unsaved duplicate raw completion.',
                        'Single-trajectory and batched forced-action completions can have floating-point differences. Their exact argmax equivalence cannot be independently established from these artifacts; no new cross-batch inference is performed.',
                        'The initially shared untrained tensors and unqueried-action masks/predictions are source-level conditions, not independently recoverable saved artifacts.',
                        'Reserved-test checks cover saved scene identities, output filenames and resource declarations; they cannot prove absence of unsaved reads.',
                        'The development set is shared by promotion and final screening; passing this audit does not establish R2 confirmation, learning generalization or deployment advantage.'
                    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent,
                        help='directory containing saved registered R1 evidence')
    args = parser.parse_args()
    if not args.out.is_dir():
        parser.error('--out must be an existing evidence directory')
    result = Audit(args.out).run()
    destination = args.out/'qa_saved.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(verdict=result['verdict'], checks=result['checks'], failures=result['failure_count'],
                          output=str(destination.resolve())), ensure_ascii=False))
    return 0 if result['all_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
