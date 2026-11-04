"""Independent saved-data review. No planner, environment, or torch import.

Uses only the standard library and NumPy for trusted local teacher arrays.
Writes only this review trace's recomputed.json; never changes source records.
Run from the workspace with rc3_learning/.venv/Scripts/python.exe -B.
"""
from pathlib import Path
from collections import Counter
import gzip
import hashlib
import itertools
import json
import math
import pickle
import statistics as st
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
WORKSPACE = ROOT.parent
TRACE = Path(__file__).resolve().parent
METHODS = ('fixed3', 'detour', 'stop', 'detour_early', 'continued_ce', 'ce_gain')
MODELS = ('continued_ce', 'ce_gain')
TOL = 1e-8


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {str(k): native(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [native(v) for v in x]
    return x


def digest(x):
    return hashlib.sha256(json.dumps(native(x), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-7), (a, b)


def wins(values):
    return dict(wins=sum(v > TOL for v in values), ties=sum(abs(v) <= TOL for v in values), losses=sum(v < -TOL for v in values))


def distribution(values):
    return dict(n=len(values), mean=st.mean(values), median=st.median(values), min=min(values), max=max(values))


def compare_objects(a, b, where=''):
    assert type(a) is type(b) or isinstance(a, (int, float)) and isinstance(b, (int, float)), (where, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (where, a.keys(), b.keys())
        for k in a: compare_objects(a[k], b[k], where + '.' + k)
    elif isinstance(a, list):
        assert len(a) == len(b), where
        for i, (aa, bb) in enumerate(zip(a, b)): compare_objects(aa, bb, where + f'[{i}]')
    elif isinstance(a, float): close(a, b)
    else: assert a == b, (where, a, b)


protocol_path = ROOT / 'outputs/protocol.json'
p = read(protocol_path)
results_path = ROOT / 'outputs/evaluation/results.json'
rows = read(results_path)
saved_summary = read(ROOT / 'outputs/summary.json')
saved_feedback = read(ROOT / 'outputs/action_feedback.json')
audit = read(ROOT / 'outputs/plan_audit.json')
setup = read(ROOT / 'outputs/evaluation/model_setup.json')
receipt = read(ROOT / 'outputs/evaluation/receipt.json')
assert p['max_plans'] == 72 and not p['training_allowed'] and not p['ppo_allowed']
assert len(rows) == 72 and len({(r['case_id'], r['method']) for r in rows}) == 72
assert sha(protocol_path) == (ROOT / 'outputs/protocol.sha256').read_text().strip() == receipt['protocol_sha256']
expected = [dict(slot=i*6+j, case_id=c['case_id'], method=m)
            for i, c in enumerate(p['cases'])
            for j, m in enumerate(METHODS[i % 6:] + METHODS[:i % 6])]
assert read(ROOT / 'outputs/evaluation/schedule.json') == expected
assert [json.loads(line) for line in (ROOT / 'outputs/evaluation/attempts.jsonl').read_text().splitlines()] == expected
assert [dict(slot=r['slot'], case_id=r['case_id'], method=r['method']) for r in rows] == expected
assert Counter(c['mode'] for c in p['cases']) == dict(regular=4, mixed=4, blocked=4)
assert all(r['status'] == 'complete' and r['valid'] and not r['failures'] for r in rows)
assert audit['all_passed'] and audit['plans'] == 72
audit_map = {(v['case_id'], v['method']): v for v in audit['checks']}
by_case = {c['case_id']: {r['method']: r for r in rows if r['case_id'] == c['case_id']} for c in p['cases']}
with (WORKSPACE / 'rc3_rl_g2/outputs/teacher_extra/teacher.pkl').open('rb') as handle:
    teacher = pickle.load(handle)
assert sha(WORKSPACE / 'rc3_rl_g2/outputs/teacher_extra/teacher.pkl') == p['extra_teacher_sha256']
for method in MODELS:
    assert sha(WORKSPACE / p['models'][method]['path']) == p['models'][method]['sha256'] == setup[method]['checkpoint_sha256']

feedback = []
evidence_checks = []
all_events = []
policy_decisions = 0
snapshots = {}
for case in p['cases']:
    case_id = case['case_id']
    scene_path = WORKSPACE / case['scene_path']
    assert sha(scene_path) == case['scene_sha256']
    scene = read(scene_path)
    own = [s for s in teacher['samples'] if int(s['scene_id'].split(':', 1)[0]) == case_id]
    assert len(own) == 8
    teacher_map = {digest(s['environment_observation']): s for s in own}
    assert len(teacher_map) == 8
    for method in METHODS:
        r = by_case[case_id][method]
        destination = ROOT / f'outputs/evaluation/case_{case_id:02d}_{method}'
        evidence_path = destination / 'evidence.json.gz'
        e = json.loads(gzip.decompress(evidence_path.read_bytes()))
        plan = read(destination / 'plan.json')
        snapshots[(case_id, method)] = e
        assert plan == e['plan']
        checked = audit_map[(case_id, method)]
        assert sha(destination / 'plan.json') == checked['plan_sha256']
        assert sha(evidence_path) == checked['evidence_sha256']
        assert checked['valid'] and checked['independent_validation']['valid_model_plan']
        assert checked['independent_validation']['all_tasks_served']
        assert not checked['independent_validation']['errors']
        assert checked['independent_validation']['collision_count'] == checked['independent_validation']['point_collision_count'] == 0
        length = sum(math.dist(a, b) for a, b in zip(plan['route_polyline'], plan['route_polyline'][1:]))
        close(length, r['L'])
        close(sum(plan['route_leg_costs_m']), length)
        assert len(plan['service_stops']) == r['K']
        close(r['J'], length + .75 * r['K'])
        close(r['J'], plan['objective_length_plus_stops_m'])
        close(r['anchor_J'], by_case[case_id]['fixed3']['J'])
        close(r['quality_gain_percent'], 100 * (r['anchor_J'] - r['J']) / r['anchor_J'])
        assert r['J'] <= r['anchor_J'] + TOL
        assigned = [a['task_id'] for a in plan['assignments']]
        assert len(assigned) == len(set(assigned)) == len(scene['tasks'])
        assert set(assigned) == {t['id'] for t in scene['tasks']}
        assert not plan['unserved_task_ids'] and not plan['unserved_reasons']
        close(r['time_overrun_s'], max(0., r['complete_s'] - p['soft_complete_limit_s']))
        close(r['complete_plus_model_load_s'], r['complete_s'] + setup.get(method, {}).get('load_s', 0.))
        if r['shared_environment']:
            actions, events = e['actions'], e['events']
            assert len(actions) == r['steps']
            assert sum(a['proposal_calls'] for a in actions) == r['proposal_calls'] == e['counters']['proposal_calls']
            total = sum(a['reward'] for a in actions) + sum(ev['reward'] for ev in events)
            close(total, r['return_'])
            work = sum(a['after_work'] - a['before_work'] for a in actions) + sum(ev['menu_work_units'] for ev in events)
            assert work == r['work_used']
            close(total, r['quality_gain_percent'] - p['config']['lambda_work'] * work / p['config']['work_scale'])
            close(r['return_vs_stop'], total - by_case[case_id]['stop']['return_'])
            assert r['route_changes'] == sum(a['info']['route_changed'] for a in actions)
            assert r['best_improvements'] == sum(a['info']['best_improved'] for a in actions)
            close(e['finalization']['returned_cost'], r['J'])
            close(e['finalization']['best_cost'], min([r['anchor_J']] + [a['after_best_J'] for a in actions]))
            assert [node-1 for node in e['final_best'][1:-1]] == plan['visit_order_candidate_ids']
            assert not e['finalization']['fallback_used'] and not e['finalization']['failure_reason']
            close(e['finalization']['quality_correction'], 0.)
            running = 0.
            for a in actions:
                obs = a['environment_observation']
                idx = a['action_id']
                assert obs['mask'][idx] and a['action'] == obs['actions'][idx]
                s = teacher_map.get(digest(obs))
                label = 'state_unseen' if s is None else ('action_queried' if s['evidence_mask'][idx] else 'action_unqueried')
                assert a['teacher_status'] == label
                q = None if s is None or label != 'action_queried' else float(s['q_values'][idx])
                delta = None if q is None else q - float(s['q_values'][s['stop_id']])
                assert (a['known_Q'] is None) == (q is None)
                assert (a['known_delta_vs_stop'] is None) == (delta is None)
                if q is not None: close(q, a['known_Q']); close(delta, a['known_delta_vs_stop'])
                assert a['teacher_depth'] == (None if s is None else int(s['teacher_depth']))
                close(a['prefix_return'], running)
                close(a['immediate_best_quality_percent'], 100 * (a['before_best_J'] - a['after_best_J']) / r['anchor_J'])
                close(a['reward'], a['immediate_best_quality_percent'] - p['config']['lambda_work'] * (a['after_work'] - a['before_work']) / p['config']['work_scale'])
                close(a['reward'], a['info']['quality_reward'] - a['info']['work_penalty'] + a['info']['shaping_reward'])
                close(a['realized_policy_suffix_return'], total - running)
                close(a['realized_after_action_return'], total - running - a['reward'])
                stop_total = 100 * (r['anchor_J'] - a['before_best_J']) / r['anchor_J'] - p['config']['lambda_work'] * (a['before_work'] + obs['pending_menu_work']) / p['config']['work_scale']
                close(a['hypothetical_stop_total_return'], stop_total)
                item = dict(case_id=case_id, method=method, mode=case['mode'], step=a['step'], action_id=idx,
                    action_kind=a['action']['kind'], teacher_status=label, teacher_depth=a['teacher_depth'],
                    known_teacher_Q=q, known_teacher_delta=delta, proposal_calls=a['proposal_calls'],
                    accepted=a['info']['accepted'], route_changed=a['info']['route_changed'], best_improved=a['info']['best_improved'],
                    immediate_best_quality_percent=a['immediate_best_quality_percent'], reward=a['reward'],
                    realized_policy_suffix_return=total-running, realized_after_action_return=total-running-a['reward'],
                    prefix_return=running, after_work=a['after_work'], before_best_J=a['before_best_J'], after_best_J=a['after_best_J'],
                    hypothetical_stop_total_return=stop_total, final_episode_return=total, episode_return_vs_stop=r['return_vs_stop'],
                    suffix_scope='Observed frozen-policy trajectory, not optimal or teacher Q')
                feedback.append(item)
                running += a['reward']
            all_events.extend(dict(case_id=case_id, method=method, **ev) for ev in events)
            for choice in e['policy_observations']:
                if choice['executed']:
                    score = [v if v is not None else -math.inf for v in choice['scores']['logits']]
                    assert max(range(len(score)), key=score.__getitem__) == choice['action_id']
                    assert choice['observation']['mask'][choice['action_id']]
                    policy_decisions += 1
        else:
            assert r['return_'] is None and r['work_used'] is None
        evidence_checks.append(dict(case_id=case_id, method=method, recomputed_L=length, recomputed_J=length+.75*r['K'],
            plan_sha256=sha(destination/'plan.json'), evidence_sha256=sha(evidence_path)))

feedback.sort(key=lambda f: (next(r['slot'] for r in rows if r['case_id'] == f['case_id'] and r['method'] == f['method']), f['step']))
compare_objects(feedback, saved_feedback)
metrics = {}
for method in METHODS:
    own = [r for r in rows if r['method'] == method]
    vals = dict(n=len(own), mean_J=st.mean(r['J'] for r in own), mean_L=st.mean(r['L'] for r in own),
        mean_K=st.mean(r['K'] for r in own), mean_gain_percent=st.mean(r['quality_gain_percent'] for r in own),
        median_complete_ms=1000*st.median(r['complete_s'] for r in own),
        median_complete_plus_model_load_ms=1000*st.median(r['complete_plus_model_load_s'] for r in own),
        mean_proposals=st.mean(r['proposal_calls'] for r in own), total_proposals=sum(r['proposal_calls'] for r in own),
        quality_improved_cases=sum(r['quality_gain_percent'] > TOL for r in own), fallbacks=sum(r['fallback'] for r in own),
        stop_at_root=sum(r['stop_at_root'] for r in own), stop_reasons=dict(Counter(r['stop_reason'] for r in own)),
        overruns=sum(r['time_overrun_s'] > 0 for r in own), models_loaded_once=method in MODELS)
    if own[0]['shared_environment']:
        outcomes = wins([r['return_vs_stop'] for r in own])
        vals.update(mean_return=st.mean(r['return_'] for r in own), mean_return_vs_stop=st.mean(r['return_vs_stop'] for r in own),
            return_vs_stop_wins=outcomes['wins'], return_vs_stop_ties=outcomes['ties'], return_vs_stop_losses=outcomes['losses'],
            mean_work=st.mean(r['work_used'] for r in own))
    else: vals.update(mean_return=None, mean_return_vs_stop=None, return_vs_stop_wins=None, return_vs_stop_ties=None, return_vs_stop_losses=None, mean_work=None)
    compare_objects(vals, saved_summary['methods'][method])
    metrics[method] = dict(**vals, complete_ms=distribution([1000*r['complete_s'] for r in own]),
        complete_plus_model_load_ms=distribution([1000*r['complete_plus_model_load_s'] for r in own]),
        total_route_changes=sum(r['route_changes'] for r in own), total_best_improvements=sum(r['best_improvements'] for r in own),
        min_gain_percent=min(r['quality_gain_percent'] for r in own), max_gain_percent=max(r['quality_gain_percent'] for r in own))

paired = {}
for method, other in itertools.combinations(METHODS, 2):
    quality = [v[other]['J'] - v[method]['J'] for v in by_case.values()]
    speed = [v[other]['complete_s'] - v[method]['complete_s'] for v in by_case.values()]
    entry = dict(method=method, comparator=other, J_lower=wins(quality), time_lower=wins(speed),
        mean_J_improvement=st.mean(quality), mean_complete_ms_saved=1000*st.mean(speed),
        method_quality_and_time_dominates_cases=[i for i,v in by_case.items() if v[method]['J'] <= v[other]['J']+TOL and v[method]['complete_s'] <= v[other]['complete_s']+TOL and (v[method]['J'] < v[other]['J']-TOL or v[method]['complete_s'] < v[other]['complete_s']-TOL)],
        comparator_quality_and_time_dominates_cases=[i for i,v in by_case.items() if v[other]['J'] <= v[method]['J']+TOL and v[other]['complete_s'] <= v[method]['complete_s']+TOL and (v[other]['J'] < v[method]['J']-TOL or v[other]['complete_s'] < v[method]['complete_s']-TOL)])
    if by_case[0][method]['shared_environment'] and by_case[0][other]['shared_environment']:
        delta=[v[method]['return_']-v[other]['return_'] for v in by_case.values()]
        entry.update(return_higher=wins(delta), mean_return_difference=st.mean(delta))
    paired[f'{method}_vs_{other}'] = entry

mode_metrics = {}
for mode in ('regular', 'mixed', 'blocked'):
    mode_metrics[mode] = {}
    for method in METHODS:
        own=[r for r in rows if r['method'] == method and r['mode'] == mode]
        mode_metrics[mode][method] = dict(n=len(own), mean_J=st.mean(r['J'] for r in own),
            mean_gain_percent=st.mean(r['quality_gain_percent'] for r in own), median_complete_ms=1000*st.median(r['complete_s'] for r in own),
            mean_return_vs_stop=st.mean(r['return_vs_stop'] for r in own) if own[0]['shared_environment'] else None,
            return_vs_stop=wins([r['return_vs_stop'] for r in own]) if own[0]['shared_environment'] else None)

model_feedback={}
unknown_examples=[]
for method in MODELS:
    own=[a for a in feedback if a['method'] == method and a['proposal_calls'] > 0]
    model_feedback[method]={}
    for label in ('action_queried','action_unqueried','state_unseen'):
        subset=[a for a in own if a['teacher_status'] == label]
        val=dict(n=len(subset), immediate_best_improvements=sum(a['immediate_best_quality_percent'] > TOL for a in subset),
            positive_immediate_rewards=sum(a['reward'] > TOL for a in subset),
            positive_actual_policy_suffixes=sum(a['realized_policy_suffix_return'] > TOL for a in subset),
            mean_immediate_reward=st.mean(a['reward'] for a in subset) if subset else None,
            mean_actual_policy_suffix=st.mean(a['realized_policy_suffix_return'] for a in subset) if subset else None)
        compare_objects(val,saved_summary['model_feedback'][method][label])
        model_feedback[method][label]=dict(**val,
            visited_scene_count=len({a['case_id'] for a in subset}),
            positive_suffix_vs_same_state_stop=sum((a['final_episode_return']-a['hypothetical_stop_total_return']) > TOL for a in subset),
            mean_suffix_vs_same_state_stop=st.mean(a['final_episode_return']-a['hypothetical_stop_total_return'] for a in subset) if subset else None)
        if label != 'action_queried':
            for a in subset:
                unknown_examples.append(dict(**a, suffix_vs_same_state_stop=a['final_episode_return']-a['hypothetical_stop_total_return']))

late = []
for r in rows:
    if not r['shared_environment']:continue
    subset=[a for a in feedback if a['case_id']==r['case_id'] and a['method']==r['method']
        and a['step']>0 and a['before_best_J']<r['anchor_J']-TOL and a['hypothetical_stop_total_return']>r['return_']+TOL]
    if r['best_improvements']>0 and subset:
        late.append(dict(case_id=r['case_id'], method=r['method'], actual_return=r['return_'],
            retrospectively_better_visited_stop_return=max(a['hypothetical_stop_total_return'] for a in subset),
            scope='Accounting on already visited states, assumes prefix final validation no rollback; not deployed or optimal'))
compare_objects(late, saved_summary['retrospective_late_stop_diagnostics'])

sources={str(path.relative_to(ROOT)).replace('\\','/'):sha(path) for path in [protocol_path,results_path,ROOT/'outputs/summary.json',ROOT/'outputs/action_feedback.json',ROOT/'outputs/plan_audit.json',ROOT/'inputs/user_request.txt',ROOT/'paper/EXPERIMENT_PLAN.md']}
output=dict(review_scope='Independent saved-data arithmetic; no environment step, planner, branch query, model forward, training, PPO, or timing run',
    source_sha256=sources, checks=dict(all_passed=True, plans=len(evidence_checks), shared_actions=len(feedback),
        executed_policy_decisions=policy_decisions, terminal_events=len(all_events), shared_proposals=sum(r['proposal_calls'] for r in rows if r['shared_environment']),
        original_detour_proposals=sum(r['proposal_calls'] for r in rows if r['method']=='detour'),
        all_returned_routes_no_worse_than_anchor=True, arithmetic_matches_executor_summary_and_feedback=True,
        plan_audit_hashes_match_saved_data=True, model_checkpoint_hashes_match_protocol=True),
    methods=metrics, pairs=paired, modes=mode_metrics, model_feedback=model_feedback,
    unknown_action_examples=unknown_examples, retrospective_late_stop=late, model_setup=setup,
    evidence_checks=evidence_checks, paper_claim_audit='unavailable',
    audit_qualification='provisional — no paper claim audit available')
(TRACE/'recomputed.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({k:output[k] for k in ['review_scope','checks','methods','model_feedback']},ensure_ascii=False,indent=2))
