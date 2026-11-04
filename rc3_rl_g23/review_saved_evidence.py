"""Read-only reviewer arithmetic. Never imports execution/model/environment modules.

Inputs are completed saved records. Output is confined to this G23 review trace.
This script does not plan, query, repair, train, select, or forward a model.
"""
import gzip
import hashlib
import itertools
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch  # Artifact deserialization only; no model construction or forward.

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
TRACE = ROOT / 'paper/review-traces/experiment-result-to-claim/2026-11-04_run01'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def close(a, b, tol=1e-7):
    assert abs(a - b) < tol, (a, b)


def suffix_check(item):
    transitions, events = item['transitions'], item['events']
    total = sum(t['reward'] for t in transitions) + sum(e['reward'] for e in events)
    close(total, item['remaining_return'])
    quality = sum(t['quality'] for t in transitions) + sum(e['quality_reward'] for e in events)
    penalty = (sum(t['work'] for t in transitions) + sum(e['menu_work_units'] for e in events)) / 1e6
    close(total, quality - penalty)
    close(item['total_return'], 100 * (item['anchor_J'] - item['returned_J']) / item['anchor_J'] - item['work'] / 1e6)
    assert item['validation']['valid_model_plan']
    return len(transitions), len(events)


def main():
    p = read(OUT / 'protocol.json')
    execution_analysis = read(OUT / 'analysis.json')
    label_receipt = read(OUT / 'label_verification.json')
    assert label_receipt['all_passed']
    all_scenes = [case for group in p['splits'].values() for case in group]
    assert len(all_scenes) == len({case['seed'] for case in all_scenes})
    old_protocol = read(ROOT.parent / 'rc3_rl_g2/outputs/protocol.json')
    old_seeds = {case['seed'] for group in old_protocol['splits'].values() for case in group}
    assert not old_seeds.intersection(case['seed'] for case in all_scenes)
    assert sha(OUT / 'protocol.json') == (OUT / 'protocol.sha256').read_text().strip()
    assert sha(OUT / 'protocol.json') == read(OUT / 'verification_protocol.json')['base_protocol_sha256']
    sources = {name: sha(ROOT / name) == expected for name, expected in p['source_sha256'].items()}
    assert all(sources.values())
    frozen = read(OUT / 'freeze_before.json')
    frozen_count = 0
    for item in frozen['g2_and_legacy']['members']:
        if item['local_expected']:
            assert sha(ROOT.parent / item['path']) == item['local_sha256']
            frozen_count += 1
    for item in frozen['g2_and_legacy']['archives']:
        assert sha(ROOT.parent / item['path']) == item['sha256']
    for name, identity in frozen['g21_payloads'].items():
        assert sha(ROOT.parent / name) == identity['actual']
    for name, identity in frozen['g22_manifest']['files'].items():
        assert sha(ROOT.parent / name) == identity['sha256']
    assert sha(ROOT.parent / 'rc3_rl_g21_delivery.zip') == frozen['g21_archive_sha256']
    assert sha(ROOT.parent / 'rc3_rl_g22_delivery.zip') == frozen['g22_archive_sha']
    assert sha(ROOT.parent / 'rc3_rl_g21/MANIFEST.sha256.json') == frozen['g21_manifest_sha256']
    for name, existed in frozen['g21_gate_files_exist'].items():
        assert (ROOT.parent / 'rc3_rl_g21/outputs' / name).exists() == existed
    rows = read(OUT / 'test_results.json')
    methods = sorted(set(x['method'] for x in rows))
    assert len(rows) == 180 and len(methods) == 15
    index = {(x['case_id'], x['rep'], x['method']): x for x in rows}
    assert len(index) == len(rows)
    assert set(index) == set(itertools.product(range(6), range(2), methods))
    plan_checks = []
    for row in rows:
        registered_case = p['splits']['test'][row['case_id']]
        assert row['seed'] == registered_case['seed'] and row['mode'] == registered_case['mode']
        path = OUT / 'plans' / f"{row['case_id']}_{row['rep']}_{row['method']}.json"
        plan = read(path)
        xy = np.asarray(plan['route_polyline'], dtype=float)
        length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
        close(length, row['L'])
        close(length, sum(plan['route_leg_costs_m']))
        close(row['J'], length + 0.75 * row['K'])
        close(row['J'], plan['objective_length_plus_stops_m'])
        assert row['K'] == len(plan['service_stops'])
        assert row['valid'] and plan['validation']['valid_model_plan']
        assert not plan['unserved_task_ids'] and not plan['validation']['errors']
        order = plan['visit_order_candidate_ids']
        assert order == [s['candidate_id'] for s in plan['service_stops']]
        assigned = [(a['task_id'], a['candidate_id'], a['arm']) for a in plan['assignments']]
        stop_assigned = [(t['task_id'], s['candidate_id'], t['arm']) for s in plan['service_stops'] for t in s['tasks']]
        assert sorted(assigned) == sorted(stop_assigned)
        task_ids = [a[0] for a in assigned]
        assert len(task_ids) == len(set(task_ids))
        anchor = index[(row['case_id'], row['rep'], 'original_fixed3')]['J']
        assert row['J'] <= anchor + 1e-7
        assert row['overrun'] == (row['complete_s'] > p['complete_soft_s'])
        if row['return_'] is not None:
            close(row['return_'], 100 * (anchor - row['J']) / anchor - row['work'] / 1e6)
        plan_checks.append(dict(file=path.name, sha256=sha(path), polyline_length_m=length,
                                task_count=len(task_ids), arithmetic_and_assignments=True))

    summary = {}
    for method in methods:
        own = [r for r in rows if r['method'] == method]
        summary[method] = dict(n=len(own), mean_J=float(np.mean([r['J'] for r in own])),
            median_complete_ms=float(np.median([r['complete_s'] for r in own]) * 1000),
            p95_complete_ms=float(np.quantile([r['complete_s'] for r in own], .95) * 1000),
            overruns=sum(r['overrun'] for r in own), fallbacks=sum(r['fallback'] for r in own),
            calls=sum(r['calls'] for r in own), reasons=dict(Counter(r['reason'] for r in own)),
            work_budget_overruns=sum(r['work'] is not None and r['work'] > p['config']['work_budget'] for r in own))

    rng = np.random.default_rng(2026110423)
    exhaustive_indices = np.asarray(list(itertools.product(range(6), repeat=6)), dtype=np.int64)
    comparisons = {}
    for comparator in p['main_gate']['comparators']:
        scene_values = []
        per_seed = {}
        for case_id in range(6):
            values = []
            for seed in p['seeds']:
                for rep in range(2):
                    method = comparator if comparator in ('same_interface_rule', 'original_detour16') else f'{seed}_{comparator}'
                    base = index[(case_id, rep, method)]['J']
                    proposed = index[(case_id, rep, f'{seed}_multi_policy_iteration')]['J']
                    values.append(100 * (base - proposed) / base)
            scene_values.append(float(np.mean(values)))
        for seed in p['seeds']:
            own = []
            for case_id in range(6):
                for rep in range(2):
                    method = comparator if comparator in ('same_interface_rule', 'original_detour16') else f'{seed}_{comparator}'
                    base = index[(case_id, rep, method)]['J']
                    proposed = index[(case_id, rep, f'{seed}_multi_policy_iteration')]['J']
                    own.append(100 * (base - proposed) / base)
            per_seed[str(seed)] = float(np.mean(own))
        values = np.asarray(scene_values)
        ci = np.quantile(values[rng.integers(0, 6, (10000, 6))].mean(axis=1), [.025, .975])
        exact = np.quantile(values[exhaustive_indices].mean(axis=1), [.025, .975])
        published = execution_analysis['comparisons'][comparator]
        assert np.allclose(values, published['scene_values'], atol=1e-12, rtol=0)
        assert np.allclose(ci, published['ci95'], atol=1e-12, rtol=0)
        close(float(values.mean()), published['mean_gain_percent'])
        comparisons[comparator] = dict(scene_values_percent=scene_values, mean_gain_percent=float(values.mean()),
            bootstrap10000_ci95=ci.tolist(), exhaustive_scene_bootstrap_ci95=exact.tolist(), per_seed_mean_gain_percent=per_seed,
            passed=bool(values.mean() >= .3 and ci[0] > 0))

    alignment = read(OUT / 'alignment.json')
    prefix_lengths = []
    for item in alignment['cases']:
        original, aligned = item['original'], item['aligned']
        assert len(aligned) == item['matched_prefix'] and len(original) == 16 and item['valid']
        prefix_lengths.append(len(aligned))
        for k, step in enumerate(aligned):
            ref = original[k]
            assert step['action']['kind'] == 'improve'
            assert (step['action']['start'], step['action']['end']) == (ref['window']['start'], ref['window']['end'])
            assert step['route'] == ref['after']
            if k:
                assert aligned[k - 1]['route'] == ref['before']
            assert step['after_J'] <= step['before_J'] + 1e-7

    training = []
    saved_label_index = {}
    checkpoint_checks = []
    for seed in p['seeds']:
        initial = torch.load(OUT / f'model_{seed}_init.pt', map_location='cpu', weights_only=False)
        frozen_saved = torch.load(OUT / f'model_{seed}_frozen_init.pt', map_location='cpu', weights_only=False)
        assert initial['state'].keys() == frozen_saved['state'].keys()
        assert all(torch.equal(value, frozen_saved['state'][key]) for key, value in initial['state'].items())
        frozen_policy_id = digest({k: v.detach().numpy() for k, v in initial['state'].items()})
        checkpoint_checks.append(dict(seed=seed, frozen_policy_id=frozen_policy_id,
            init_and_frozen_init_tensors_identical=True, updates_init=initial['updates'], updates_frozen=frozen_saved['updates']))
        previous_keys = set()
        for round_id in range(2):
            path = OUT / f'seed{seed}_round{round_id}.pkl.gz'
            with gzip.open(path, 'rb') as f:
                data = pickle.load(f)
            keys = {s['observation']['state_key'] for s in data}
            policy_ids = sorted(set(s['policy_id'] for s in data))
            assert len(policy_ids) == 1
            assert policy_ids == ['rule'] if round_id == 0 else policy_ids != ['rule']
            if round_id:
                assert policy_ids == [frozen_policy_id]
            for item in data:
                obs = item['observation']
                assert obs['mask'][item['action']] and obs['actions'][item['action']]['kind'] != 'stop'
                assert len(obs['actions']) <= 17
                assert len(obs['edge_left']) == len(obs['edge_right']) == len(obs['edge_task'])
                assert np.all(obs['edge_left'] | obs['edge_right'])
                assert all(math.isfinite(float(item[k])) for k in ('one', 'multi', 'static', 'qstop', 'step_reward'))
                saved_label_index[(seed, round_id, item['case']['case_id'], digest(obs), item['action'])] = item
            training.append(dict(seed=seed, round=round_id, rows=len(data), distinct_scenes=len({s['case']['seed'] for s in data}),
                states=len(keys), new_menu_state_keys_vs_round0=len(keys - previous_keys) if round_id else None,
                policy_ids=policy_ids, labels_multi_static_differ=sum(abs(s['multi'] - s['static']) > 1e-8 for s in data),
                sha256=sha(path)))
            if round_id == 0:
                previous_keys = keys
    with gzip.open(OUT / 'verified_labels.json.gz', 'rt', encoding='utf-8') as f:
        proof = json.load(f)
    transitions = events = 0
    for item in proof:
        saved = saved_label_index[(item['seed'], item['round'], item['case'], item['state_key'], item['action'])]
        for field, actual in item['labels'].items():
            close(saved[field], actual)
        assert item['evaluated_policy'] == saved['policy_id']
        assert item['h'] == saved['remaining_h'] and item['remaining_work'] == saved['remaining_work']
        for suffix in ('stop', 'one', 'multi', 'static'):
            if item[suffix] is not None:
                a, b = suffix_check(item[suffix])
                assert a <= item['h']
                transitions += a
                events += b
        labels = item['labels']
        close(labels['qstop'], item['stop']['remaining_return'])
        close(labels['one'], item['one']['remaining_return'] - labels['qstop'])
        close(labels['multi'], item['multi']['remaining_return'] - labels['qstop'])
        close(labels['static'], item['multi' if item['static'] is None else 'static']['remaining_return'] - labels['qstop'])
        assert item['multi']['continuation_policy'] == item['evaluated_policy']
        if item['static'] is not None:
            assert item['static']['continuation_policy'] == 'rule'
            # Both arms fork immediately after the identical actual first transition.
            assert item['multi']['transitions'][0] == item['static']['transitions'][0]
        assert item['multi']['transitions'][0] == item['one']['transitions'][0]
    assert len(proof) == len(saved_label_index) == label_receipt['training_rows']

    log = [json.loads(x) for x in (OUT / 'resource.jsonl').read_text().splitlines()]
    counts = {k: 0 for k in p['limits']}
    phase_counts = defaultdict(Counter)
    for item in log:
        counts[item['kind']] += item['n']
        assert item['total'] == counts[item['kind']]
        phase_counts[item['phase']][item['kind']] += item['n']
    resource = read(OUT / 'resources.json')
    assert resource['elapsed_s'] <= p['wall_limit_s']
    assert counts == resource['counts'] == label_receipt['after_counts']
    assert all(counts[k] <= p['limits'][k] for k in counts)
    assert counts['optimizer_updates'] == 3840
    for seed in p['seeds']:
        assert sum(v['nonstop'] for key, v in phase_counts.items() if key.startswith(f'seed{seed}_')) <= 1536
    for seed in p['seeds']:
        fits = sorted(OUT.glob(f'fit_{seed}_*.json'))
        assert len(fits) == 5 and all(len(read(path)) == 256 for path in fits)
    v = phase_counts['verification']
    assert dict(v) == {k: n for k, n in label_receipt['verification_calls'].items() if n}

    audit_file = ROOT / 'paper/PAPER_CLAIM_AUDIT.json'
    result = dict(role='independent paper architect/reviewer', execution_prohibition='No execution/model modules imported; no planning, repair, forward, training, or queries',
        protocol_source_hashes_match=sources, frozen_dependencies=dict(g2_legacy_payloads=frozen_count,
            g21_payloads=len(frozen['g21_payloads']), g22_payloads=len(frozen['g22_manifest']['files']), all_hashes_match=True),
        plans=len(rows), independent_test_scenes=6, plan_checks=plan_checks,
        independent_scope='Saved polyline/cost/assignment arithmetic and saved transition ledgers; original scene collision/reachability validation is separately supplied by plan_validation.json',
        summary=summary, comparisons=comparisons, training=training, checkpoint_checks=checkpoint_checks,
        alignment=dict(cases=len(prefix_lengths), original_repairs=192, matched_prefixes=prefix_lengths, matched_repairs=sum(prefix_lengths),
                       semantic_windows_and_routes_match=True, full16_reproduction_claim=False),
        label_verification=dict(rows=len(proof), suffix_transitions_checked=transitions, termination_events_checked=events, arithmetic_passed=True),
        global_counts=counts, limits=p['limits'], phase_counts={k: dict(v) for k, v in phase_counts.items()},
        overruns=sum(r['overrun'] for r in rows), fallback_count=sum(r['fallback'] for r in rows),
        primary_passed=all(x['passed'] for x in comparisons.values()) and not any(r['overrun'] for r in rows),
        paper_claim_audit=read(audit_file).get('verdict', 'UNKNOWN') if audit_file.exists() else 'unavailable')
    TRACE.mkdir(parents=True, exist_ok=True)
    (TRACE / 'independent_recalculation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('plans','independent_test_scenes','comparisons','training','alignment','label_verification','global_counts','overruns','primary_passed')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
