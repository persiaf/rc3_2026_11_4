"""Independent saved-result acceptance; no author metric/model/planner imports.

Default mode reads saved records only. --replay-calibration separately authorizes
one post-run checkpoint QA replay of missing calibration predictions and one dev
menu per model. It never generates scenes, navigation, H, labels or gradients.
Research resources remain immutable; QA resources are reported separately.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INPUTS = {}
CHECKS = Counter()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(name, lines=False):
    path = HERE / name
    INPUTS[path.relative_to(ROOT).as_posix()] = dict(sha256=digest(path), bytes=path.stat().st_size)
    content = path.read_text(encoding='utf-8')
    return [json.loads(x) for x in content.splitlines() if x.strip()] if lines else json.loads(content)


def write(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def check(condition, name):
    if not condition:
        raise AssertionError(name)
    CHECKS[name] += 1


def close(a, b, tol=1e-8):
    return bool(np.allclose(a, b, atol=tol, rtol=0))


def cover_matrix(scene):
    # Independent geometry-only replay of the registered spherical-sector proxy.
    tasks = np.asarray([[t['x'], t['y'], t['z']] for t in scene['tasks']])
    nodes = np.asarray([[c['x'], c['y'], c['theta']] for c in scene['candidates']])
    robot = scene['robot']
    delta = tasks[:, None, :2] - nodes[None, :, :2]
    co, si = np.cos(nodes[:, 2]), np.sin(nodes[:, 2])
    xx = co * delta[:, :, 0] + si * delta[:, :, 1] - robot['arm_forward_offset']
    yy = (-si * delta[:, :, 0] + co * delta[:, :, 1])[:, :, None] - np.array([1, -1]) * robot['arm_spacing'] / 2
    zz = tasks[:, None, 2] - robot['shoulder_height']
    distance = np.sqrt(xx[:, :, None] ** 2 + yy ** 2 + zz[:, :, None] ** 2)
    arms = ((distance >= robot['arm_min_reach']) & (distance <= robot['arm_max_reach'])
            & (np.abs(np.arctan2(yy, xx[:, :, None])) <= math.radians(robot['arm_sector_half_angle_deg'])))
    return arms, arms.any(axis=2)


def remaining_after(C, prefix, require_positive=False):
    remaining = np.ones(len(C), dtype=bool)
    for node in prefix:
        check(isinstance(node, int) and 1 <= node <= C.shape[1], 'route_node_id_valid')
        if require_positive:
            check(bool(C[remaining, node - 1].any()), 'prefix_step_positive_geometry_coverage')
        remaining &= ~C[:, node - 1]
    return remaining


def aggregate(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[row['case']].append(row)
    per_scene = [dict(case=case, states=len(group), **{k: float(np.mean([r[k] for r in group])) for k in keys})
                 for case, group in sorted(groups.items())]
    return {k: float(np.mean([r[k] for r in per_scene])) for k in keys}, per_scene


def interval(values):
    # Independently implement the registered scene-paired percentile bootstrap.
    x = np.array(values, dtype=float)
    draws = np.random.default_rng(11522026).integers(0, len(x), size=(20000, len(x)))
    means = x[draws].sum(axis=1) / len(x)
    return np.percentile(means, [2.5, 97.5]).tolist()


def checkpoint_acceptance(labels, protocol, replay=False):
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    fit = np.concatenate([np.asarray(r['features']) for r in labels if r['partition'] == 'fit'])
    expected_mean = fit.mean(axis=0).astype(np.float32)
    expected_std = np.maximum(fit.std(axis=0), .05).astype(np.float32)
    bundles, weights = {}, {}
    for kind in ('linear', 'neural'):
        path = HERE / 'weights' / (kind + '.pt')
        INPUTS[path.relative_to(ROOT).as_posix()] = dict(sha256=digest(path), bytes=path.stat().st_size)
        tick = time.perf_counter()
        bundle = torch.load(path, map_location='cpu', weights_only=False)
        loading_s = time.perf_counter() - tick
        check(bundle['protocol_sha256'] == digest(HERE / 'protocol.json'), 'checkpoint_protocol_hash')
        check(bundle['seeds'] == protocol['models']['stochastic_seeds'] and len(bundle['states']) == 3, 'checkpoint_all_three_seeds')
        check(bundle['optimizer_updates_each'] == 1024, 'checkpoint_fixed_update_metadata')
        check(len(bundle['features']) == 38, 'checkpoint_feature_dimension')
        check(np.array_equal(bundle['mean'], expected_mean) and np.array_equal(bundle['std'], expected_std), 'checkpoint_fit_only_standardization')
        states = []
        for seed, state in zip(bundle['seeds'], bundle['states']):
            check(all(bool(torch.isfinite(t).all()) for t in state.values()), 'checkpoint_finite_tensor_endpoint')
            count = sum(t.numel() for t in state.values())
            check(count == bundle['parameters'] == (38 if kind == 'linear' else 6721), 'checkpoint_parameter_count')
            states.append(dict(seed=seed, parameters=count, shapes={k: list(t.shape) for k, t in state.items()},
                               tensor_sha256=signature({k: hashlib.sha256(t.numpy().tobytes()).hexdigest() for k, t in state.items()})))
        check(len({x['tensor_sha256'] for x in states}) == 3, 'three_distinct_seed_endpoints')
        train = read(kind + '_training.json')
        check(train['seeds'] == bundle['seeds'] and train['parameters'] == bundle['parameters'], 'training_checkpoint_identity')
        for seed in train['seeds']:
            history = [h for h in train['history'] if h['seed'] == seed]
            check([h['step'] for h in history] == [1, 256, 512, 768, 1024], 'training_all_registered_endpoint_steps')
            check(all(np.isfinite([h['loss'], h['value'], h['ranking']]).all() for h in history), 'training_finite_saved_losses')
        weights[kind] = dict(sha256=digest(path), loading_s=loading_s, endpoints=states,
                             optimizer_updates_each=1024, optimizer_updates_total=3072,
                             standardization_fit_states=320, training_s=train['training_s'])
        bundles[kind] = bundle
    if replay:
        check(not (HERE / 'calibration_replay.json').exists(), 'no_implicit_QA_replay_overwrite')
        replay_rows, check_rows = {}, {}
        forward_s = {}
        with threadpool_limits(limits=1), torch.inference_mode():
            for kind, bundle in bundles.items():
                tick = time.perf_counter()
                replay_rows[kind] = []
                # Pure tensor implementation: no author Ranker/predict/evaluate.
                def predict(features):
                    x = torch.as_tensor((np.asarray(features, dtype=np.float32) - bundle['mean']) / bundle['std'])
                    members = []
                    for state in bundle['states']:
                        if kind == 'linear':
                            z = torch.nn.functional.linear(x, state['net.weight']).squeeze(-1)
                        else:
                            z = torch.nn.functional.linear(x, state['net.0.weight'], state['net.0.bias']).relu()
                            z = torch.nn.functional.linear(z, state['net.2.weight'], state['net.2.bias']).relu()
                            z = torch.nn.functional.linear(z, state['net.4.weight'], state['net.4.bias']).squeeze(-1)
                        members.append((z - z[0]).numpy())
                    z = np.stack(members)
                    return z.mean(axis=0), z.std(axis=0), z
                for r in (x for x in labels if x['partition'] == 'calibration'):
                    mean, sd, members = predict(r['features'])
                    replay_rows[kind].append(dict(case=r['case'], threshold=r['threshold'], prefix=r['prefix'],
                        menu=r['menu'], predicted_advantage_percent=mean.tolist(), ensemble_sd_percent=sd.tolist(),
                        member_advantage_percent=members.tolist()))
                dev = next(x for x in labels if x['partition'] == 'development')
                mean, sd, members = predict(dev['features'])
                saved = read(kind + '_actions.json')['rows'][0]
                check(saved['case'] == dev['case'] and saved['prefix'] == dev['prefix'], 'QA_dev_identity')
                max_error = float(np.max(np.abs(mean - saved['predicted_advantage_percent'])))
                check(max_error <= 1e-6 and close(sd, saved['ensemble_sd_percent'], 1e-6), 'QA_checkpoint_reproduces_saved_dev_prediction')
                check_rows[kind] = dict(case=dev['case'], threshold=dev['threshold'], menu=dev['menu'],
                    predicted_advantage_percent=mean.tolist(), ensemble_sd_percent=sd.tolist(), max_abs_error_pp=max_error)
                forward_s[kind] = time.perf_counter() - tick
        replay_record = dict(schema='T-post-run-saved-checkpoint-QA-v1', utc=datetime.now(timezone.utc).isoformat(),
            scope='Independent saved-checkpoint acceptance replay, NOT original research inference or new research sample.',
            protocol_sha256=digest(HERE / 'protocol.json'), labels_sha256=digest(HERE / 'labels.jsonl'),
            weights_sha256={k: v['sha256'] for k, v in weights.items()}, rows=replay_rows, dev_identity_check=check_rows,
            resources=dict(calibration_member_forward_calls=384, calibration_action_rows=3072,
                dev_identity_member_forward_calls=6, dev_identity_action_rows=48,
                total_member_forward_calls=390, total_action_rows=3120,
                checkpoint_loading_s={k: v['loading_s'] for k, v in weights.items()}, tensor_forward_loop_s=forward_s,
                elapsed_checkpoint_QA_s=time.perf_counter()-started, optimizer_updates=0, scene_generation=0,
                navigation_preparations=0, planner_calls=0, H_calls=0, teacher_label_queries=0,
                cpu_torch_threads=2, threadpool_limit=1),
            calibration_scene_max_abs_error_pp={})
        label_index = {(r['case'], tuple(r['prefix'])): r for r in labels if r['partition'] == 'calibration'}
        for kind, rows in replay_rows.items():
            maxima = defaultdict(float)
            for row in rows:
                source = label_index[(row['case'], tuple(row['prefix']))]
                truth = 100 * (source['Yref'] - np.array(source['Y'])) / source['Yref']
                maxima[row['case']] = max(maxima[row['case']], float(np.max(np.abs(np.array(row['predicted_advantage_percent']) - truth))))
            replay_record['calibration_scene_max_abs_error_pp'][kind] = dict(sorted(maxima.items()))
        write('calibration_replay.json', replay_record)
    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay-calibration', action='store_true')
    args = parser.parse_args()
    started = time.perf_counter()
    p = read('protocol.json')
    registration = read('registration_receipt.json')
    check(digest(HERE/'protocol.json') == registration['protocol_sha256'], 'registered_protocol_hash')
    check(all(registration[k] == 0 for k in ('optimizer_updates_before_registration', 'new_label_queries_before_registration', 'new_scene_generation_before_registration')), 'registration_zero_work_receipt')
    frozen = json.loads(subprocess.check_output(['git', 'show', p['frozen_research_commit'] + ':project_management/GIT_MANIFEST.json'], cwd=ROOT))
    source_results = {}
    for name, expected in p['source_sha256'].items():
        actual = digest(ROOT/name)
        check(actual == expected, 'all_registered_source_bytes_frozen')
        old = not name.startswith('rc3_teacher_distill/')
        if old:
            check(actual == frozen['files'][name]['sha256'], 'old_source_matches_frozen_git_manifest')
        source_results[name] = dict(sha256=actual, old_frozen_manifest_match=old)
    for key, filename in [('seeds_sha256', 'seeds.json'), ('seed_audit_sha256', 'seed_audit.json'),
                          ('T0_sha256', 't0_diagnostics.json'), ('prereg_review_sha256', 'PREREG_REVIEW.md'),
                          ('unit_receipt_sha256', 'unit_receipt.json')]:
        check(digest(HERE/filename) == p[key], 'registered_supporting_receipt_hash')
    unit = read('unit_receipt.json')
    check(unit['tests_passed'] == 19 and unit['failures'] == 0, 'saved_19_synthetic_tests_receipt')
    t0 = read('t0_diagnostics.json')
    check(all(t0[k] == 0 for k in ('new_scene_generation', 'new_planning', 'new_H', 'new_neural_inference', 'new_optimizer_updates', 'retained_test_reads')), 'T0_saved_zero_research_receipt')
    seeds = read('seeds.json')
    seed_audit = read('seed_audit.json')
    values = [r[k] for rows in seeds.values() for r in rows for k in ('map_seed', 'task_seed')]
    check(len(values) == len(set(values)) == 312, 'all_registered_map_and_task_seeds_distinct')
    check(not any(7700000000 <= x <= 7700000059 for x in values), 'protected_seed_range_not_used')
    check(seed_audit['verdict'] == 'PASS' and not seed_audit['literal_hits'] and not seed_audit['range_overlaps'], 'saved_historical_seed_audit_pass')
    check(Counter(r['partition'] for r in seeds['train']) == {'fit': 80, 'calibration': 16}, 'fit80_calibration16_seed_split')
    check(Counter(r['mode'] for r in seeds['development']) == {'regular': 8, 'mixed': 8, 'blocked': 8}, 'balanced_independent_dev24')
    labels = read('labels.jsonl', True)
    terminals = read('run/terminal.jsonl', True)
    journal = read('run/call_journal.jsonl', True)
    resources = read('resources.json')
    ledger = read('run/ledger_summary.json')
    check(resources['ledger'] == ledger, 'resource_ledger_exact_join')
    check(len(labels) == 480 and len(terminals) == 3960, 'saved_menu_and_terminal_counts')
    terminal_index = {r['call_id']: r for r in terminals}
    check(len(terminal_index) == 3960 and sorted(terminal_index) == list(range(1, 3961)), 'terminal_unique_contiguous_call_ids')
    starts = {r['call_id']: r for r in journal if r['event'] == 'start'}
    completions = {r['call_id']: r for r in journal if r['event'] == 'complete'}
    check(len(starts) == len(completions) == 3960 and len(journal) == 7920, 'journal_3960_start_complete_zero_failure_cache')
    categories, kinds = Counter(), Counter()
    internal, certifications = Counter(), Counter()
    inc, exc = defaultdict(float), defaultdict(float)
    source_versions = dict(H_sha256=p['H_sha256'], H_source_sha256=p['H_source_sha256'],
        oracle_source_sha256=p['source_sha256']['rc3_nav_value/r1/oracle.py'],
        runtime_sha256=p['source_sha256']['rc3_v4/runtime.py'], scalar_validator_sha256=p['source_sha256']['rc3_baseline/validator.py'])
    references = {}
    for row in terminals:
        cid = row['call_id']
        check(starts[cid]['metadata'] == row['metadata'] and starts[cid]['category'] == completions[cid]['category'] == row['category'], 'terminal_journal_metadata_join')
        check(row['source_versions'] == source_versions, 'terminal_all_source_versions')
        check(row['status'] == 'complete' and row['valid'] and row['charged'] and not row['cache_hit'], 'terminal_complete_valid_charged_no_cache')
        check(row['route'] == row['final_route'] and row['route'][0] == row['route'][-1] == 0, 'terminal_closed_final_route')
        check(close(row['J'], row['cost']) and close(row['J'], row['L']+.75*row['K']), 'terminal_final_J_L_K_identity')
        check(close(row['raw_J'], row['raw_L']+.75*row['raw_K']), 'terminal_raw_J_L_K_identity')
        check(row['K'] == len(row['route'])-2 and row['raw_K'] == len(row['raw_route'])-2, 'terminal_route_count_K')
        check(row['J'] <= row['raw_J'] + 1e-7, 'H_nonincreasing_saved_objective')
        check(np.isfinite([row[k] for k in ('J','L','K','raw_J','raw_L','raw_K')]).all(), 'terminal_finite_raw_final')
        for validator in ('export_validation', 'independent_validation'):
            cert = row[validator]
            check(cert['valid_model_plan'] and cert['all_tasks_served'] and cert['collision_count'] == cert['point_collision_count'] == 0 and not cert['errors'] and not cert['unserved_task_ids'], 'terminal_saved_dual_validation_pass')
            check(close(row['L'], cert['independent_polyline_length_m'], 1e-7), 'terminal_final_L_matches_validator_polyline')
        check(row['certification_counts'] == {k:1 for k in ('export_attempts','export_completed','scalar_attempts','scalar_completed')}, 'terminal_dual_validation_counts')
        check(row['internal'] == completions[cid]['internal'] and row['certification_counts'] == completions[cid]['certification_counts'], 'journal_internal_validation_join')
        categories[row['category']] += 1
        kinds[row['kind']] += 1
        internal.update(row['internal']['counts'])
        certifications.update(row['certification_counts'])
        for k,v in row['internal']['inclusive_s'].items(): inc[k] += v
        for k,v in row['internal']['exclusive_s'].items(): exc[k] += v
        if row['kind'] == 'native':
            key = (row['metadata']['split'], row['metadata']['case'])
            check(key not in references and row['method'] == 'fixed3' and not row['fallback'], 'one_native_fixed3_B_per_scene')
            references[key] = row
    check(dict(categories) == {'reference':120, 'train':3072, 'development':768}, 'terminal_call_category_counts')
    check(dict(kinds) == {'native':120, 'full_H':3840}, 'native_and_full_H_counts')
    check(all(categories[k] == ledger['used'][k] == ledger['completed'][k] and ledger['failed'][k] == 0 for k in ledger['used']), 'all_category_attempt_complete_failure_conservation')
    check(dict(internal) == ledger['internal_counts'] and dict(certifications) == ledger['certification_counts'], 'internal_and_certification_totals')
    check(all(close(inc[k], ledger['internal_inclusive_s'][k]) and close(exc[k], ledger['internal_exclusive_s'][k]) for k in inc), 'internal_time_conservation')
    check(close(sum(r['oracle_s'] for r in terminals), ledger['oracle_s']), 'oracle_time_conservation')
    scenes, covers, arms = {}, {}, {}
    for split in ('train','development'):
        for spec in seeds[split]:
            key = (split, spec['case'])
            filename = f"scenes/{split}_{spec['case']:02}.json"
            scene = read(filename)
            check((scene['seed'],scene['task_seed'],scene['complexity']) == (spec['map_seed'],spec['task_seed'],spec['mode']), 'saved_scene_registered_seed_task_mode')
            scenes[key] = scene
            arms[key], covers[key] = cover_matrix(scene)
    check(len(list((HERE/'scenes').glob('*.json'))) == 120 and not list((HERE/'scenes').glob('T2*')), 'only_T1_120_scenes_no_T2_files')
    state_keys, call_ids = set(), []
    prefix_counts, partition_counts = Counter(), Counter()
    label_index = {}
    for r in labels:
        key = (r['split'], r['case'])
        spec = seeds[r['split']][r['case']]
        C = covers[key]
        prefix = r['prefix']
        state_key = (*key, tuple(prefix))
        check(state_key not in state_keys, 'saved_full_prefix_states_unique')
        state_keys.add(state_key)
        label_index[(r['case'],tuple(prefix))] = r if r['partition']=='development' else label_index.get((r['case'],tuple(prefix)))
        check(r['partition'] == spec['partition'] and r['mode'] == spec['mode'], 'label_partition_and_mode')
        check(r['scene_sha256'] == digest(HERE/f"scenes/{r['split']}_{r['case']:02}.json"), 'label_exact_scene_byte_hash')
        B = references[key]['route'][1:-1]
        check(r['B'] == B and r['reference_sha256'] == signature(B), 'label_fixed_B_reference_hash')
        remaining = remaining_after(C, prefix, True)
        check(int(remaining.sum()) == r['remaining'], 'saved_remaining_geometry_replay')
        check(prefix == B[:len(prefix)], 'prefix_actual_gB_trajectory')
        fraction = 1-float(remaining.mean())
        check(fraction + 1e-12 >= r['threshold'], 'prefix_coverage_threshold_crossed')
        if prefix:
            prior = remaining_after(C, prefix[:-1])
            check(1-float(prior.mean()) + 1e-12 < r['threshold'], 'first_prefix_crossing_not_later_state')
        x = np.asarray(r['features'])
        menu = r['menu']
        check(x.shape == (8,38) and np.isfinite(x).all() and len(set(menu)) == 8, 'menu_eight_unique_actions_features38_finite')
        check(menu[0] == next(a for a in B if C[remaining,a-1].any()), 'reference_first_legal_positive_B')
        gains = C[remaining].sum(axis=0)
        ids = np.array(menu)-1
        check((gains[ids] > 0).all() and (x[:,3:5] >= 0).all(), 'menu_positive_gain_finite_saved_current_home_distance')
        check(close(x[:,0], gains[ids]/len(C), 1e-6) and close(x[:,1], gains[ids]/10, 1e-6), 'feature_geometry_gain_recomputed')
        check(close(x[:,21],fraction,1e-6) and close(x[:,22],len(prefix)/100,1e-6) and close(x[:,31],remaining.sum()/100,1e-6), 'feature_complete_prefix_summaries')
        check(close(x[:,32],r['legal_action_count']/C.shape[1],1e-6), 'feature_saved_legal_count')
        check(close(x[:,16],[float(a==menu[0]) for a in menu],1e-6), 'feature_reference_flag')
        reconstructed_ratio = gains[ids]/(x[:,3]*10+.75)
        check(close(x[:,5],reconstructed_ratio,2e-6), 'feature_saved_navigation_gain_ratio')
        rule_idx = menu.index(r['rule_action'])
        check(reconstructed_ratio[rule_idx] >= max(reconstructed_ratio)-2e-6, 'rule_maximizes_saved_menu_ratio')
        check(len(r['Y']) == len(r['call_ids']) == len(r['raw_routes']) == len(r['final_routes']) == 8 and close(r['Yref'],r['Y'][0]), 'all_menu_actions_queried_reference_Y')
        for i, cid in enumerate(r['call_ids']):
            call = terminal_index[cid]
            meta = call['metadata']
            check(call['kind']=='full_H' and call['category']==r['split'] and meta['split']==r['split'] and meta['case']==r['case'], 'label_terminal_split_case')
            check(meta['prefix']==prefix and meta['action']==menu[i] and meta['threshold']==r['threshold'], 'label_terminal_full_prefix_forced_action')
            check(meta['reference']=='gB-v1' and meta['reference_sha256']==r['reference_sha256'] and meta['scene_sha256']==r['scene_sha256'], 'label_terminal_teacher_scene_versions')
            check(call['raw_route']==r['raw_routes'][i] and call['route']==r['final_routes'][i] and close(call['J'],r['Y'][i]), 'label_raw_final_route_and_Y_exact_join')
            raw = r['raw_routes'][i]
            check(raw[:len(prefix)+2] == [0,*prefix,menu[i]], 'raw_full_ordered_prefix_then_forced_action')
            rem = remaining.copy()
            rem &= ~C[:,menu[i]-1]
            expected = [0,*prefix,menu[i]]
            for node in B:
                if C[rem,node-1].any():
                    expected.append(node)
                    rem &= ~C[:,node-1]
            expected.append(0)
            check(expected == raw and not rem.any(), 'independent_geometry_gB_completion_exact_raw_route')
            check(not remaining_after(C,call['route'][1:-1]).any(), 'final_route_geometry_all_tasks_covered')
            expected_key = dict(scene=r['scene_sha256'],prefix=prefix,action=menu[i],reference=r['reference_sha256'],H=p['H_sha256'])
            check(signature(expected_key)==call['cache_key_sha256'], 'exact_cache_key_full_prefix_action_versions')
            call_ids.append(cid)
        prefix_counts[key] += 1
        partition_counts[r['partition']] += 1
    check(len(set(call_ids)) == 3840, 'every_label_unique_real_terminal_no_cache')
    check(set(prefix_counts.values()) == {4} and len(prefix_counts)==120, 'four_distinct_threshold_states_all120_scenes')
    check(dict(partition_counts) == {'fit':320,'calibration':64,'development':96}, 'saved_state_partition_counts')
    scene_resources = read('scene_resources.jsonl', True)
    check(len(scene_resources)==120 and all(prefix_counts[(r['split'],r['case'])]==r['states'] for r in scene_resources), 'scene_resource_state_conservation')
    weights = checkpoint_acceptance(labels,p,args.replay_calibration)
    qa = read('calibration_replay.json')
    check(qa['protocol_sha256']==digest(HERE/'protocol.json') and qa['labels_sha256']==digest(HERE/'labels.jsonl') and qa['weights_sha256']=={k:v['sha256'] for k,v in weights.items()}, 'QA_replay_current_input_hashes')
    development = [r for r in labels if r['partition']=='development']
    cal_index = {(r['case'],tuple(r['prefix'])):r for r in labels if r['partition']=='calibration'}
    metrics, recalibration = {}, {}
    for kind in ('linear','neural'):
        saved = read(kind+'_actions.json')
        maxima = defaultdict(float)
        check(len(qa['rows'][kind])==64, 'QA_all64_calibration_states')
        for row in qa['rows'][kind]:
            r = cal_index[(row['case'],tuple(row['prefix']))]
            check(row['menu']==r['menu'] and row['threshold']==r['threshold'], 'QA_calibration_full_prefix_menu_join')
            pred = np.asarray(row['predicted_advantage_percent'])
            members = np.asarray(row['member_advantage_percent'],dtype=np.float32)
            check(close(pred,members.mean(axis=0),1e-7), 'QA_calibration_three_member_mean')
            true = 100*(r['Yref']-np.asarray(r['Y']))/r['Yref']
            maxima[r['case']] = max(maxima[r['case']], float(np.max(np.abs(pred-true))))
        check(len(maxima)==16, 'QA_16_independent_calibration_scene_maxima')
        q = sorted(maxima.values())[math.ceil(17*.9)-1]
        check(close(q,saved['calibration_radius_pp'],1e-7), 'independently_reconstructed_calibration_radius')
        recalibration[kind] = dict(scene_max_abs_error_pp=dict(sorted(maxima.items())),order_statistic=16,
            radius_pp=q, original_saved_radius_pp=saved['calibration_radius_pp'],
            max_abs_radius_difference_pp=abs(q-saved['calibration_radius_pp']))
        rows=[]
        check(len(saved['rows'])==96,'all96_saved_dev_prediction_states')
        for r,predrow in zip(development,saved['rows']):
            check(all(r[k]==predrow[k] for k in ('case','threshold','prefix','menu','Y','Yref')), 'prediction_saved_label_exact_join')
            pred = np.asarray(predrow['predicted_advantage_percent'])
            check(pred.shape==(8,) and np.isfinite(pred).all() and pred[0]==0, 'saved_finite_anchored_all8_action_predictions')
            y=np.asarray(r['Y']); best=float(min(y)); true=100*(r['Yref']-y)/r['Yref']
            i=sorted(range(8),key=lambda i:(-pred[i],r['menu'][i]))[0]
            regret=100*(y[i]-best)/best
            optimism=float(pred[i]-true[i])
            rule_i=r['menu'].index(r['rule_action'])
            rr=100*(y[rule_i]-best)/best
            row=dict(case=r['case'],top1_percent=100*float(y[i]<=best+1e-7),near_0_1_percent=100*float(regret<=.1+1e-10),
                near_0_5_percent=100*float(regret<=.5+1e-10),mean_regret_percent=float(regret),rule_mean_regret_percent=float(rr),
                regret_improvement_percent=float(rr-regret),selected_optimism_gt_1pp_percent=100*float(optimism>1),
                selected_interval_coverage_percent=100*float(abs(optimism)<=q+1e-10),
                all_cover=bool(np.max(np.abs(pred-true))<=q+1e-10), chosen_error_pp=optimism)
            check(predrow['chosen_index']==i and predrow['chosen']==r['menu'][i] and not predrow['unknown'], 'saved_argmax_numeric_tie_chosen_no_unknown')
            check(close(predrow['true_advantage_percent'],true) and close(predrow['regret_percent'],regret) and close(predrow['rule_regret_percent'],rr) and close(predrow['chosen_error_pp'],optimism), 'all_saved_dev_action_numeric_fields_recomputed')
            flags=dict(top1=row['top1_percent']==100,near_optimal=row['near_0_1_percent']==100,near_0_5=row['near_0_5_percent']==100,
                interval_covers=row['selected_interval_coverage_percent']==100,error_gt_1pp=optimism>1,all_actions_interval_cover=row['all_cover'])
            check(all(bool(predrow[k])==bool(v) for k,v in flags.items()), 'all_saved_dev_action_boolean_fields_recomputed')
            rows.append(row)
        keys=list(rows[0]);keys=[k for k in keys if k not in ('case','all_cover')]
        m,per_scene=aggregate(rows,keys)
        m['scene_simultaneous_coverage_percent']=100*float(np.mean([all(r['all_cover'] for r in rows if r['case']==c) for c in sorted({r['case'] for r in rows})]))
        m['regret_improvement_CI95']=interval([r['regret_improvement_percent'] for r in per_scene])
        m['calibration_radius_pp']=q
        for k in ('top1_percent','near_0_1_percent','near_0_5_percent','mean_regret_percent','rule_mean_regret_percent','regret_improvement_percent','selected_optimism_gt_1pp_percent','selected_interval_coverage_percent','scene_simultaneous_coverage_percent','regret_improvement_CI95'):
            check(close(m[k],saved[k]), 'independent_aggregate_matches_saved_metrics')
        m['checks']=dict(top1_ge60=m['top1_percent']>=60,near_ge75=m['near_0_1_percent']>=75,regret_le0_5=m['mean_regret_percent']<=.5,
            rule_increment_CI_positive=m['regret_improvement_CI95'][0]>0,optimistic_outliers_le10=m['selected_optimism_gt_1pp_percent']<=10,
            scene_simultaneous_coverage_ge80=m['scene_simultaneous_coverage_percent']>=80,unknown_zero=True,finite_complete=True)
        check(m['checks']==saved['checks'] and all(m['checks'].values())==saved['passed'], 'independent_all_gate_checks')
        m['passed']=all(m['checks'].values())
        m['scene_bootstrap_CI95_descriptive']={k:interval([x[k] for x in per_scene]) for k in keys}
        m['per_scene']=per_scene
        metrics[kind]=m
    baseline_results={}
    for kind in ('rule','reference_first'):
        rows=[]
        for r in development:
            y=np.asarray(r['Y']);best=y.min();i=r['menu'].index(r['rule_action']) if kind=='rule' else 0
            regret=100*(y[i]-best)/best
            rows.append(dict(case=r['case'],top1_percent=100*float(y[i]<=best+1e-7),near_0_1_percent=100*float(regret<=.1+1e-10),
                near_0_5_percent=100*float(regret<=.5+1e-10),mean_regret_percent=float(regret)))
        keys=[k for k in rows[0] if k!='case'];m,per_scene=aggregate(rows,keys)
        m['scene_bootstrap_CI95_descriptive']={k:interval([x[k] for x in per_scene]) for k in keys};m['per_scene']=per_scene
        baseline_results[kind]=m
    gate=read('T1_GATE.json');status=read('STATUS.json');not_run=read('T2_NOT_RUN.json');before=read('before_training.json');start=read('run_started.json')
    check(gate['checks']==metrics['neural']['checks'] and gate['passed']==metrics['neural']['passed']==False, 'T1_primary_failed_stop_independently_confirmed')
    check(gate['frozen_model_sha256']=={k:v['sha256'] for k,v in weights.items()} and gate['protocol_sha256']==registration['protocol_sha256'], 'gate_frozen_weights_protocol_hashes')
    check(status['T1']=='failed_stop' and status['T2']=='not_started_gate_failed' and status['T3']=='not_authorized' and status['retained60']=='sealed', 'status_frozen_stop_sealed')
    check(all(not_run[k]==0 for k in ('T2_scene_generation','T2_planner_calls','T2_teacher_queries')) and not not_run['T3_authorized'], 'T2_not_run_zero_resource_receipt')
    check(before['optimizer_updates']==0 and before['labels']==480 and before['terminal_calls']==3960 and before['protocol_sha256']==start['protocol_sha256']==registration['protocol_sha256'], 'before_training_labels_complete_and_protocol')
    check(start['utc']>registration['registered_utc'] and before['UTC']>start['utc'], 'receipt_UTC_order_registration_before_research_before_training')
    stats=resources['stats']
    raw_decisions=sum(len(r['raw_route'])-2-len(r['metadata'].get('prefix',[])) for r in terminals if r['kind']=='full_H')+sum(len(r['route'])-2 for r in references.values())
    expected_stats=dict(scene_generation=120,preparations=120,nav_cache_hits=0,hot_priming=0,raw_gB_trajectories=3960,
        raw_gB_decisions=raw_decisions,label_uses=3840,menus=480,feature_rows=3840,
        training_forward_batches=6144,training_rows=196608,optimizer_updates=6144,inference_forward_batches=960,inference_action_rows=7680,
        raw_student_trajectories=0,student_decisions=0,old_model_trajectories=0,old_model_decisions=0,T0_new_research_calls=0,retained_test_reads=0,T3_calls=0)
    check(stats==expected_stats,'all_research_stats_count_conservation')
    check(resources['elapsed_wall_s']<3600 and close(resources['deadline_remaining_s']+resources['elapsed_wall_s'],3600,1e-5), 'research_elapsed_within_T1_deadline')
    descriptions={}
    for partition in ('fit','calibration','development'):
        rows=[r for r in labels if r['partition']==partition]
        best_ties=[int(np.sum(np.asarray(r['Y'])<=min(r['Y'])+1e-7)) for r in rows]
        gaps=[100*(sorted(r['Y'])[1]-min(r['Y']))/min(r['Y']) for r in rows]
        descriptions[partition]=dict(states=len(rows),best_tie_states=sum(x>1 for x in best_ties),all_actions_tied_states=sum(x==8 for x in best_ties),
            first_second_gap_percent_mean=float(np.mean(gaps)),first_second_gap_percent_median=float(np.median(gaps)),
            mean_menu_to_saved_legal_fraction=float(np.mean([8/r['legal_action_count'] for r in rows])))
    claims=read('paper_claims.json')
    check(claims['teacher_identification_supported']==claims['route_compression_supported']==claims['independent_RL_gain_supported']==claims['T3_authorized']==False,'machine_claims_match_stop')
    result=dict(schema='T-independent-saved-result-audit-v1',utc=datetime.now(timezone.utc).isoformat(),
        reviewer='/root/t_final_independent_review', verdict='PASS_SAVED_NUMBERS_AND_FAILED_STOP',
        scope=dict(default='saved records, geometry coverage, hashes, checkpoint tensor inspection; no author metrics',
            QA_replay='Separately authorized post-run saved-checkpoint calibration QA; 390 member forwards, 3120 action rows; no research query/gradient.',
            original60='No content read, generation or query; only saved seed exclusion receipt and range checked.',
            old_T0='T0 file hash/version/zero-call receipts checked; full old R0/R1 audit NOT rerun.',
            navigation='No navigation generation. Finite current/home menu distances checked as saved features plus full saved certification; D matrix was not persisted.',
            update_evidence='6144 count agrees frozen loop, six endpoint metadata and sparse histories; no independent per-update event journal exists.'),
        checks=dict(sorted(CHECKS.items())),source_hashes=source_results,inputs=INPUTS,weights=weights,
        partitions=dict(partition_counts),menus=480,labels=3840,independent_scene_count=120,
        research_resources=dict(categories=dict(categories),kinds=dict(kinds),journal_entries=len(journal),internal_counts=dict(internal),
            certification_counts=dict(certifications),cache_hits=0,optimizer_updates=6144,raw_gB_decisions=raw_decisions,
            elapsed_wall_s=resources['elapsed_wall_s'],T2_calls=0),
        postrun_QA_resources=qa['resources'],calibration_recomputed=recalibration,model_metrics=metrics,baselines=baseline_results,
        posthoc_menu_description=dict(note='Post-hoc descriptive only; does not replace any registered gate or compare directly with R1.',partitions=descriptions),
        T1_passed=False,T2_started=False,T3_authorized=False,paper_claim_audit='unavailable' if not (ROOT/'paper/PAPER_CLAIM_AUDIT.json').exists() else 'exists_not_evaluated',
        downstream_writing='provisional — no paper claim audit available',
        audit_elapsed_s=time.perf_counter()-started)
    write('saved_result_audit.json',result)
    print(json.dumps(dict(verdict=result['verdict'],checks=sum(CHECKS.values()),linear=metrics['linear']['mean_regret_percent'],neural=metrics['neural']['mean_regret_percent'],T1=False,QA_forwards=qa['resources']['total_member_forward_calls']),ensure_ascii=False))


if __name__=='__main__':
    main()
