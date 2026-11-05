"""Independent R0 audit: saved JSON/JSONL arithmetic and hashes only.

Does not import the project, prepare a scene, construct a completion, run H,
call a planner/model/optimizer, or issue a geometry certificate.
"""
from pathlib import Path
import collections
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'r0'
TRACE = Path(__file__).resolve().parent
TOL = 1e-7


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sig(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def close(x, y):
    assert math.isfinite(x) and math.isfinite(y)
    assert abs(x-y) <= TOL, (x, y)


def main():
    p = read(OUT/'protocol.json')
    summary = read(OUT/'execution_summary.json')
    verification = read(OUT/'verification.json')
    cases = read(OUT/'case_results.json')
    evaluations = lines(OUT/'evaluations.jsonl')
    checkpoints = lines(OUT/'checkpoints.jsonl')
    ledger = lines(OUT/'calls.jsonl')
    seeds = read(OUT/'seeds.json')
    wall = read(OUT/'wall_protocol.json')
    amendment = read(OUT/'registration_amendment.json')
    audit = read(OUT/'seed_audit.json')
    boundary = read(OUT/'seed_audit_boundary_sanity.json')
    inventory = lines(OUT/'seed_audit_inventory.jsonl')
    resources = read(OUT/'resources.json')
    analysis = read(OUT/'analysis.json')
    project = ROOT.parent

    assert len(cases) == len(seeds) == 32
    assert [x['case'] for x in cases] == list(range(32))
    modes = collections.Counter(x['mode'] for x in cases)
    assert modes == p['mode_counts'] == dict(regular=11, mixed=11, blocked=10)
    for i, (r, s) in enumerate(zip(cases, seeds)):
        assert s == dict(case=i, map_seed=8300000000+i,
                         task_seed=8400000000+i, mode=('regular', 'mixed', 'blocked')[i % 3])
        assert all(r[k] == s[k] for k in s)
        assert sha(OUT/'scenes'/f'case{i:02}.json') == r['scene_sha256']
        for method in ('native', 'root', 'teacher'):
            m = r[method]
            close(m['J'], m['L']+.75*m['K'])
            assert m['K'] >= 0
            plan = project/r['plans'][method]['path']
            assert sha(plan) == r['plans'][method]['sha256']
            close(read(plan)['objective_length_plus_stops_m'], m['J'])
            assert r['plans'][method]['valid']
        close(r['gain_J'], r['root']['J']-r['teacher']['J'])
        close(r['gain_percent'], 100*(r['root']['J']-r['teacher']['J'])/r['root']['J'])
        assert r['teacher']['J'] <= r['root']['J']+TOL
        assert r['valid']

    # Immutable inputs and the documented registration repair.
    assert all(sha(project/n) == h for n, h in p['source_sha256'].items())
    assert sha(project/p['proposal']) == p['proposal_sha256']
    assert sha(OUT/'seeds.json') == p['seeds_sha256']
    assert sha(OUT/'seed_audit.json') == p['seed_audit_sha256']
    assert sha(OUT/'seed_audit_boundary_sanity.json') == p['seed_boundary_sha256']
    assert sig(p['H_source_sha256']) == p['H_sha256']
    canonical = ''.join(f"{r['path']}\t{r['sha256']}\t{r['file_bytes']}\t{r['scanned_bytes']}\n"
                        for r in sorted(inventory, key=lambda r: r['path']))
    canonical_hash = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    assert canonical_hash == audit['scan']['inventory_sha256']
    assert canonical_hash != sha(OUT/'seed_audit_inventory.jsonl')
    for name, h in boundary['frozen_files_sha256'].items():
        assert sha(OUT/name) == h
    assert sha(OUT/'registration_attempt1'/'protocol.json') == amendment['original_protocol_sha256']
    assert sha(OUT/'registration_attempt1'/'r0_run.py') == amendment['original_source_sha256']
    assert sha(OUT/'protocol.json') == amendment['amended_protocol_sha256']
    assert sha(ROOT/'r0_run.py') == amendment['amended_source_sha256']
    original = read(OUT/'registration_attempt1'/'protocol.json')
    differences = sorted(k for k in set(original)|set(p) if original.get(k) != p.get(k))
    assert differences == ['registration_amendment', 'seed_boundary_sha256', 'source_sha256']
    assert {k: v for k, v in original['source_sha256'].items() if k != 'rc3_nav_value/r0_run.py'} == {
        k: v for k, v in p['source_sha256'].items() if k != 'rc3_nav_value/r0_run.py'}
    assert amendment['research_calls_before_amendment'] == 0
    assert amendment['scenes_generated_before_amendment'] == 0
    assert audit['verdict'] == 'PASS_NO_RECORDED_PRIOR_USE' and not audit['collisions']
    assert not audit['scan']['errors'] and boundary['all_passed']

    assert wall['protocol_sha256'] == sha(OUT/'protocol.json')
    assert read(OUT/'calibration_started.json')['protocol_sha256'] == sha(OUT/'protocol.json')
    assert read(OUT/'run_started.json')['wall_sha256'] == sha(OUT/'wall_protocol.json')
    assert wall['source_sha256'] == p['source_sha256']
    maximum = lambda name: max(x[name] for x in wall['samples'])
    formula_wall = min(1800, max(180, math.ceil(4*(32*maximum('prepare_s')+
        32*maximum('native_s')+2272*maximum('completeH_s')+64*maximum('export_s'))+60)))
    assert formula_wall == wall['wall_limit_s'] == summary['wall_limit_s'] == 180
    assert summary['run_elapsed_s'] < summary['wall_limit_s']

    # The append-only terminal ledger counts actual calls, not cached uses.
    starts = [x for x in ledger if x['event'] == 'start']
    ends = [x for x in ledger if x['event'] == 'complete']
    assert len(starts) == len(ends) == summary['total_terminal_calls']
    assert [x['call'] for x in starts] == list(range(1, len(starts)+1))
    assert len(ledger) == 2*len(starts)
    for a, b in zip(ledger[::2], ledger[1::2]):
        assert a['event'] == 'start' and b['event'] == 'complete'
        assert all(a[k] == b[k] for k in ('case', 'kind', 'call'))
        assert math.isfinite(b['elapsed_s']) and b['elapsed_s'] >= 0
    counts = dict(collections.Counter(x['kind'] for x in starts))
    assert counts == summary['terminal_started'] == summary['terminal_completed']
    assert counts == dict(native=32, root=32, query=2016, teacher=32, other=32)
    assert all(counts[k] <= p['top_level_caps'][k] for k in counts)
    assert len(starts) <= p['total_top_level_cap'] == 2304
    assert collections.Counter(x['kind'] for x in starts[:6]) == dict(native=3, root=3)
    assert [(x['case'], x['kind']) for x in starts[:6]] == [(i, k) for i in range(3) for k in ('native', 'root')]
    assert counts == resources['terminal_calls']
    evaluation_counts = collections.Counter(x['kind'] for x in evaluations)
    assert evaluation_counts == {k: v for k, v in counts.items() if k != 'native'}
    assert len(evaluations) == resources['full_H_calls'] == 2112
    raw_transitions = 0
    for r in evaluations:
        case = cases[r['case']]
        ident = dict(scene=case['scene_sha256'], B=case['B_hash'], g='g_B-v1',
                     H=p['H_sha256'], tol=TOL, prefix=r['prefix'], action=r['action'])
        assert sig(ident) == r['key']
        for field, route_field in (('raw', 'raw_route'), ('final', 'route')):
            m, route = r[field], r[route_field]
            assert route[0] == route[-1] == 0 and len(route[1:-1]) == len(set(route[1:-1]))
            assert len(route)-2 == m['K']
            close(m['J'], m['L']+.75*m['K'])
        assert r['final']['J'] <= r['raw']['J']+TOL
        assert r['raw_route'][1:1+len(r['prefix'])] == r['prefix']
        if r['action'] is not None:
            assert r['raw_route'][1+len(r['prefix'])] == r['action']
        raw_transitions += len(r['raw_route'])-2

    # Candidate-value lineage, selection, and continuity without reconstructing paths.
    assert len(checkpoints) == 256
    assert len({(r['case'], tuple(r['prefix'])) for r in checkpoints}) == len(checkpoints)
    query_uses = query_hits = opportunity_count = immediate_suboptimal = raw_suboptimal = 0
    opportunity_values, immediate_regrets, raw_regrets = [], [], []
    max_continuity_error = max_terminal_error = 0.0
    native_internal, h_internal = collections.Counter(), collections.Counter()
    for r in evaluations:
        h_internal.update(r['internal_calls'])
    for i, case in enumerate(cases):
        base = read(OUT/'baselines'/f'case{i:02}.json')
        assert sig(base['B']) == case['B_hash']
        assert base['B_hash'] == case['B_hash']
        assert not base['native_info']['failures'] and not base['native_info']['fallback_used']
        assert base['native_info']['method'] == 'fixed3' and base['native_info']['dp_calls'] == 0
        native_internal.update(base['native_internal_calls'])
        local = [r for r in evaluations if r['case'] == i]
        by_key = {}
        for r in local:
            by_key.setdefault(r['key'], r)
        root = next(r for r in local if r['kind'] == 'root')
        teacher = next(r for r in local if r['kind'] == 'teacher')
        repeat = next(r for r in local if r['kind'] == 'other')
        assert root['raw_route'] == repeat['raw_route'] and root['route'] == repeat['route']
        for name, saved in (('root', root), ('teacher', teacher)):
            for metric in ('J', 'L', 'K'):
                close(case[name][metric], saved['final'][metric])
        prefix_rows = [r for r in checkpoints if r['case'] == i]
        assert [r['checkpoint'] for r in prefix_rows] == list(range(8))
        assert case['checkpoints'] == len(prefix_rows)
        expected = root['final']['J']
        previous_crossing = -1
        for cp in prefix_rows:
            assert teacher['raw_route'][1:1+len(cp['prefix'])] == cp['prefix']
            assert teacher['raw_route'][1+len(cp['prefix'])] == cp['chosen_action']
            assert cp['crossed_eighths'] == list(range(previous_crossing+1, cp['crossed_eighths'][-1]+1))
            previous_crossing = cp['crossed_eighths'][-1]
            cs = cp['candidates']
            assert 1 <= len(cs) <= min(8, cp['legal_actions'])
            assert len(cs) == len({x['node'] for x in cs})
            assert cs[0]['node'] == cp['reference_action']
            for c in cs:
                saved = by_key[c['key']]
                assert saved['prefix'] == cp['prefix'] and saved['action'] == c['node']
                close(c['Y'], saved['final']['J'])
                close(c['raw_J'], saved['raw']['J'])
                assert c['route'] == saved['route']
                if c['cached']:
                    assert cp['checkpoint'] == 0 and c['node'] == cp['reference_action']
                    assert saved['kind'] == 'root'
            ref = next(c for c in cs if c['node'] == cp['reference_action'])
            max_continuity_error = max(max_continuity_error, abs(ref['Y']-expected))
            close(ref['Y'], expected)
            minimum = min(c['Y'] for c in cs)
            tied = [c for c in cs if c['Y'] <= minimum+TOL]
            chosen = next((c for c in tied if c['node'] == cp['reference_action']), min(tied, key=lambda c: c['node']))
            assert cp['chosen_action'] == chosen['node']
            close(cp['chosen_Y'], chosen['Y'])
            close(cp['best_Y'], minimum)
            assert chosen['Y'] <= expected+TOL
            expected = chosen['Y']
            close(cp['queried_opportunity_J'], ref['Y']-minimum)
            immediate = next(c for c in cs if c['node'] == cp['immediate_action'])
            close(cp['immediate_regret_J'], immediate['Y']-minimum)
            assert cp['immediate_vs_final_disagrees'] == (immediate['Y'] > minimum+TOL)
            raw = min(cs, key=lambda c: (c['raw_J'], c['node']))
            assert cp['raw_best_action'] == raw['node']
            opportunity_values.append(ref['Y']-minimum)
            immediate_regrets.append(immediate['Y']-minimum)
            raw_regrets.append(raw['Y']-minimum)
            opportunity_count += ref['Y']-minimum > TOL
            immediate_suboptimal += immediate['Y']-minimum > TOL
            raw_suboptimal += raw['Y']-minimum > TOL
            query_uses += len(cs)
            query_hits += sum(c['cached'] for c in cs)
        assert previous_crossing == 7
        close(teacher['final']['J'], expected)
        max_terminal_error = max(max_terminal_error, abs(teacher['final']['J']-expected))
        assert case['candidate_uses'] == sum(len(cp['candidates']) for cp in prefix_rows)
        assert case['cache_hits'] == 1
        # First three roots are reused from calibration in a fresh evaluator.
        assert case['cache_uses'] == case['candidate_uses']+2+(i >= 3)

    q = [100*(r['root']['J']-r['teacher']['J'])/r['root']['J'] for r in cases]
    positive = [r['root']['J']-r['teacher']['J'] > TOL for r in cases]
    positive_q = [x for x, pos in zip(q, positive) if pos]
    mean_gain = statistics.mean(q)
    share = max(positive_q)/sum(positive_q)
    feasible = all(r['valid'] for r in cases) and verification['all_passed']
    gate = dict(complete_cases=32, mean_gain_percent=mean_gain,
                positive_cases=sum(positive), max_positive_relative_share=share,
                mean_gain_pass=mean_gain >= .5, positive_count_pass=sum(positive) >= 12,
                concentration_pass=share <= .25, all_feasible=feasible)
    gate['passed'] = all(gate[k] for k in ('mean_gain_pass', 'positive_count_pass', 'concentration_pass', 'all_feasible'))
    for name, value in gate.items():
        if isinstance(value, float):
            close(value, analysis['gate'][name])
        else:
            assert value == analysis['gate'][name]
    assert query_uses == resources['query_label_uses'] == 2048
    assert query_hits == resources['query_cache_hits'] == 32
    assert query_uses-query_hits == counts['query']
    assert query_uses+32+32+32-query_hits == len(evaluations)
    assert dict(h_internal) == resources['internal_fullH_calls']
    assert dict(native_internal) == resources['internal_native_calls']
    assert verification['numeric_completeH_replays'] == len(evaluations)
    assert verification['raw_service_transitions'] == raw_transitions == 173740
    assert verification['checkpoint_queue_winner_checks'] == len(checkpoints)
    assert len(verification['certificates']) == 96
    assert all(c['valid'] and sha(project/c['path']) == c['sha256'] for c in verification['certificates'])
    assert all(verification[k] == 0 for k in ('new_H_calls', 'new_native_solver_calls', 'new_model_forwards', 'new_completion_trajectories'))
    assert all(summary[k] == 0 for k in ('neural_forwards', 'optimizer_updates', 'new_policy_samples', 'retained_test_queries'))
    assert not summary['R1_started']
    for name, n in [('checkpoints_with_final_opportunity', opportunity_count),
                    ('immediate_rule_suboptimal_checkpoints', immediate_suboptimal),
                    ('raw_min_suboptimal_after_H_checkpoints', raw_suboptimal)]:
        assert analysis['opportunities'][name] == n
    quality = {m: {k: statistics.mean(r[m][k] for r in cases) for k in ('J', 'L', 'K')}
               for m in ('native', 'root', 'teacher')}
    for m in quality:
        for k in quality[m]:
            close(quality[m][k], analysis['quality'][m][k])
    # Numerical sampling on saved scalars only; identical registered exploratory
    # bootstrap (NumPy RNG seed 114), without importing project code.
    import numpy as np
    bootstrap_inputs = dict(
        teacher_vs_gBH=q,
        teacher_vs_native_Fixed3=[100*(r['native']['J']-r['teacher']['J'])/r['native']['J'] for r in cases],
        root_gBH_vs_native_Fixed3=[100*(r['native']['J']-r['root']['J'])/r['native']['J'] for r in cases])
    bootstrap = {}
    for name, values in bootstrap_inputs.items():
        a = np.asarray(values)
        rng = np.random.default_rng(114)
        boot_means = a[rng.integers(32, size=(10000, 32))].mean(axis=1)
        ci = np.quantile(boot_means, [.025, .975]).tolist()
        close(statistics.mean(values), analysis['paired'][name]['mean'])
        for x, y in zip(ci, analysis['paired'][name]['ci95']):
            close(x, y)
        bootstrap[name] = dict(mean=statistics.mean(values), ci95=ci)
    result = dict(
        audit_scope='Saved-record arithmetic, hashes and static-source review only; no project imports, planning, H, geometry certification or learning.',
        passed=True, gate=gate, quality=quality, exploratory_scene_bootstrap=bootstrap,
        teacher_vs_native_mean_percent=statistics.mean(100*(r['native']['J']-r['teacher']['J'])/r['native']['J'] for r in cases),
        root_vs_native_mean_percent=statistics.mean(100*(r['native']['J']-r['root']['J'])/r['native']['J'] for r in cases),
        mode_counts=dict(modes), total_terminal_calls=len(starts), terminal_calls=counts,
        full_H_calls=len(evaluations), candidate_uses=query_uses, query_cache_hits=query_hits,
        fullH_result_uses=query_uses+32+32+32, native_internal_calls=dict(native_internal), fullH_internal_calls=dict(h_internal),
        checkpoints=len(checkpoints), checkpoints_with_queried_opportunity=opportunity_count,
        mean_queried_opportunity_J=statistics.mean(opportunity_values),
        immediate_rule_suboptimal_checkpoints=immediate_suboptimal,
        mean_immediate_rule_regret_J=statistics.mean(immediate_regrets),
        raw_min_suboptimal_after_H_checkpoints=raw_suboptimal,
        mean_final_regret_of_raw_min_J=statistics.mean(raw_regrets),
        maximum_reference_continuity_error_J=max_continuity_error,
        maximum_final_continuity_error_J=max_terminal_error,
        canonical_inventory_sha256=canonical_hash, inventory_raw_sha256=sha(OUT/'seed_audit_inventory.jsonl'),
        registration_protocol_changed_keys=differences, verified_plan_hashes=96,
        numerical_or_protocol_blockers=[],
        geometry_scope='Consumed stored verification and plan validity records; reviewer did not issue geometry certificates.',
        source_files={name: sha(OUT/name) for name in ('protocol.json', 'registration_amendment.json', 'wall_protocol.json',
            'execution_summary.json', 'case_results.json', 'calls.jsonl', 'evaluations.jsonl', 'checkpoints.jsonl',
            'verification.json', 'analysis.json', 'resources.json', 'seed_audit.json', 'seed_audit_boundary_sanity.json')})
    (TRACE/'numeric_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('passed', 'gate', 'total_terminal_calls', 'full_H_calls', 'checkpoints_with_queried_opportunity',
          'immediate_rule_suboptimal_checkpoints', 'raw_min_suboptimal_after_H_checkpoints', 'numerical_or_protocol_blockers')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
