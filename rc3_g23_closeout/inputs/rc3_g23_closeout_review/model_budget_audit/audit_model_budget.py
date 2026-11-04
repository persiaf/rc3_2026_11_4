"""Read-only G2.3 budget and saved-label arithmetic audit (stdlib only).

Does not import project modules, unpickle data, load tensors, infer, plan, train,
measure planner runtime, or reexecute the historical branch-verification job.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(source):
    output = source / 'outputs'
    protocol = read(output / 'protocol.json')
    resources = read(output / 'resources.json')
    verification = read(output / 'label_verification.json')
    authority = read(output / 'verification_protocol.json')
    execution = read(output / 'verification_execution.json')
    receipt = read(output / 'run_receipt.json')
    failures = []

    def check(condition, label):
        if not condition:
            failures.append(label)

    def close(actual, expected, label):
        check(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-7), label)

    check(sha(output / 'protocol.json') == (output / 'protocol.sha256').read_text().strip(), 'Protocol checksum')
    for name, expected in protocol['source_sha256'].items():
        check(sha(source / name) == expected, 'Registered source checksum: ' + name)
    check(authority['base_protocol_sha256'] == sha(output / 'protocol.json'), 'Verification base protocol')
    check(execution['script_sha256'] == sha(source / 'verify_labels.py'), 'Verification execution script checksum')
    check({k: v for k, v in execution.items() if k != 'script_sha256'} == authority, 'Verification execution and registration records')

    ledger = [json.loads(line) for line in (output / 'resource.jsonl').read_text(encoding='utf-8').splitlines()]
    totals = {k: 0 for k in protocol['limits']}
    phases = defaultdict(Counter)
    snapshot_lines = []
    first_test = first_verification = None
    for line, row in enumerate(ledger, 1):
        kind, n, phase = row['kind'], row['n'], row['phase']
        check(kind in totals and isinstance(n, int) and n >= 0, f'Counter type at line {line}')
        totals[kind] += n
        phases[phase][kind] += n
        check(row['total'] == totals[kind], f'Counter continuity at line {line}')
        check(totals[kind] <= protocol['limits'][kind], f'Counter cap at line {line}')
        if totals == authority['at_registration_counts']:
            snapshot_lines.append({'line': line, 'phase': phase})
        if phase == 'final_test' and first_test is None:
            first_test = line
            before = totals.copy()
            before[kind] -= n
            check(before['optimizer_updates'] == protocol['limits']['optimizer_updates'], 'All registered updates precede final testing')
        if phase == 'verification' and first_verification is None:
            first_verification = line
            before = totals.copy()
            before[kind] -= n
            check(before == verification['before_counts'] == receipt['counts'], 'Before-verification snapshot')
    check(totals == resources['counts'] == verification['after_counts'], 'Final resource totals')
    check(resources['limits'] == protocol['limits'] == authority['global_limits'], 'Global resource limits')
    check({k: phases['verification'][k] for k in totals} == verification['verification_calls'], 'Verification resource delta')
    check(phases['final_test']['optimizer_updates'] == phases['verification']['optimizer_updates'] == 0, 'No updates after training')
    check(len(snapshot_lines) == 1 and snapshot_lines[0]['line'] < first_test, 'Recorded registration counts correspond to a pre-test ledger prefix')
    seed_calls = {}
    for seed in protocol['seeds']:
        amount = sum(values['nonstop'] for phase, values in phases.items() if phase.startswith(f'seed{seed}_'))
        seed_calls[str(seed)] = amount
        check(amount <= protocol['training']['train_calls_per_seed'], f'Training repair cap for {seed}')

    align = read(output / 'alignment.json')
    prefix_lengths = []
    original_steps = 0
    for case in align['cases']:
        original, aligned = case['original'], case['aligned']
        original_steps += len(original)
        prefix_lengths.append(len(aligned))
        check(case['matched_prefix'] == len(aligned) > 0, f"Alignment count case {case['case_id']}")
        check(len(aligned) <= len(original), f"Alignment bounded prefix case {case['case_id']}")
        for index, row in enumerate(original):
            if index:
                check(row['before'] == original[index - 1]['after'], f"Original route continuity {case['case_id']}/{index}")
        for index, row in enumerate(aligned):
            ref = original[index]
            check(row['action']['kind'] == 'improve', 'Aligned rule action type')
            check((row['action']['start'], row['action']['end']) == (ref['window']['start'], ref['window']['end']), 'Aligned rule semantic window')
            check(row['route'] == ref['after'], 'Aligned rule accepted route')
    check(original_steps == 192 and sum(prefix_lengths) == 104, 'Alignment step totals')
    check(original_steps + sum(prefix_lengths) == phases['alignment']['nonstop'], 'Alignment repair ledger')

    with gzip.open(output / 'verified_labels.json.gz', 'rt', encoding='utf-8') as stream:
        labels = json.load(stream)
    check(len(labels) == verification['training_rows'] == 132, 'Saved label count')
    grouped = defaultdict(list)
    branch_count = transition_count = 0
    for index, row in enumerate(labels):
        grouped[(row['seed'], row['round'])].append(row)
        targets = row['labels']
        close(targets['qstop'], row['stop']['remaining_return'], f'Actual STOP baseline label {index}')
        for branch in ('one', 'multi'):
            close(targets[branch], row[branch]['remaining_return'] - targets['qstop'], f'STOP-relative {branch} label {index}')
        static = row['multi'] if row['static'] is None else row['static']
        close(targets['static'], static['remaining_return'] - targets['qstop'], f'STOP-relative static label {index}')
        if row['round'] == 0:
            check(row['evaluated_policy'] == 'rule' and row['static'] is None, f'Round-0 continuation {index}')
        else:
            check(row['static'] is not None and row['evaluated_policy'] != 'rule', f'Round-1 continuation {index}')
        prefixes = []
        for branch in ('stop', 'one', 'multi', 'static'):
            item = row[branch]
            if item is None:
                continue
            branch_count += 1
            transitions = item['transitions']
            transition_count += len(transitions)
            events = item['events']
            close(sum(t['reward'] for t in transitions) + sum(e['reward'] for e in events), item['remaining_return'], f'Suffix reward sum {index}/{branch}')
            close(100 * (item['anchor_J'] - item['returned_J']) / item['anchor_J'] - item['work'] / 1e6, item['total_return'], f'Final-plan reward {index}/{branch}')
            check(len(transitions) <= row['h'], f'Finite horizon {index}/{branch}')
            close(item['finalization']['returned_cost'], item['returned_J'], f'Returned cost {index}/{branch}')
            check(item['validation']['valid_model_plan'], f'Recorded branch proxy certificate {index}/{branch}')
            check(not item['finalization']['fallback_used'], f'No recorded fallback {index}/{branch}')
            prefixes.append(item['total_return'] - item['remaining_return'])
            for t in transitions:
                close(t['reward'], t['quality'] - t['work'] / 1e6, f'Step quality/work reward {index}/{branch}')
            if not events:
                close(item['work'] - sum(t['work'] for t in transitions), protocol['config']['work_budget'] - row['remaining_work'], f'Suffix work sum {index}/{branch}')
            if branch != 'stop':
                check(transitions[0]['action'] == row['one']['transitions'][0]['action'], f'Shared first action {index}/{branch}')
                close(transitions[0]['reward'], targets['step_reward'], f'Shared first reward {index}/{branch}')
        check(max(prefixes) - min(prefixes) < 1e-7, f'Common parent return {index}')
    check(transition_count == 2067, 'Serialized suffix transition count')
    groups = []
    for (seed, round_), rows in sorted(grouped.items()):
        summary = {'seed': seed, 'round': round_, 'labels': len(rows), 'states': len({r['state_key'] for r in rows}),
                   'cases': sorted({r['case'] for r in rows}), 'remaining_h': dict(Counter(r['h'] for r in rows)),
                   'evaluated_policies': sorted({r['evaluated_policy'] for r in rows})}
        for target in ('one', 'multi', 'static'):
            values = [r['labels'][target] for r in rows]
            summary[target] = {'positive': sum(v > 1e-9 for v in values), 'negative': sum(v < -1e-9 for v in values),
                               'mean': statistics.mean(values), 'minimum': min(values), 'maximum': max(values)}
        if round_ == 1:
            differences = [r['labels']['multi'] - r['labels']['static'] for r in rows]
            summary['multi_minus_static'] = {'positive': sum(v > 1e-9 for v in differences),
                                            'negative': sum(v < -1e-9 for v in differences),
                                            'mean': statistics.mean(differences)}
        groups.append(summary)
    equivalent_keys = ('remaining_return', 'transitions', 'events', 'total_return', 'work', 'returned_J', 'route', 'finalization', 'anchor_J')
    seed2302_round1 = [r for r in labels if r['seed'] == 2302 and r['round'] == 1]
    equal_one_multi = all(all(r['one'][key] == r['multi'][key] for key in equivalent_keys) for r in seed2302_round1)

    fits = []
    fit_updates = 0
    for path in sorted(output.glob('fit_*.json')):
        records = read(path)
        fit_updates += len(records)
        check(len(records) == protocol['training']['updates_per_fit'], 'Fit update count ' + path.name)
        check([r['update'] for r in records] == list(range(1, len(records) + 1)), 'Fit update sequence ' + path.name)
        check(all(math.isfinite(r['loss']) and math.isfinite(r['grad']) for r in records), 'Finite fit log ' + path.name)
        fits.append({'file': path.name, 'updates': len(records), 'initial_loss': records[0]['loss'],
                     'final_loss': records[-1]['loss'], 'zero_gradient_records': sum(r['grad'] == 0 for r in records)})
    check(fit_updates == totals['optimizer_updates'] == 3840, 'Fit logs and optimizer ledger total')

    storage_hashes = {}
    for path in sorted(output.glob('model_*.pt')):
        with zipfile.ZipFile(path) as archive:
            storage_hashes[path.name] = {name.split('/data/')[-1]: hashlib.sha256(archive.read(name)).hexdigest()
                                        for name in archive.namelist() if '/data/' in name}
    comparisons = []
    for seed in protocol['seeds']:
        initial = storage_hashes[f'model_{seed}_init.pt']
        for name in ('frozen_init', 'matched_static_rule_value', 'multi_policy_iteration', 'one_step_repeated'):
            current = storage_hashes[f'model_{seed}_{name}.pt']
            check(current.keys() == initial.keys(), f'Storage numbering {seed}/{name}')
            changed = sum(current[k] != v for k, v in initial.items())
            check(changed == 0 if name == 'frozen_init' else changed > 0, f'Frozen or updated storage {seed}/{name}')
            comparisons.append({'seed': seed, 'model': name, 'storage_blocks': len(initial), 'changed_from_init': changed})

    tests = read(output / 'test_results.json')
    multi_test = []
    for seed in protocol['seeds']:
        rows = [r for r in tests if r['method'] == f'{seed}_multi_policy_iteration']
        multi_test.append({'seed': seed, 'measurements': len(rows), 'repair_calls': sum(r['calls'] for r in rows),
                           'reasons': dict(Counter(r['reason'] for r in rows)), 'overruns': sum(r['overrun'] for r in rows)})
    return {
        'schema': 'g23-independent-model-budget-audit-v1',
        'all_saved_record_checks_passed': not failures, 'failures': failures,
        'current_review_scope': {'new_plans': 0, 'new_environment_queries': 0, 'new_training_updates': 0,
                                'new_model_inferences': 0, 'new_performance_measurements': 0,
                                'imports': 'Python standard library only; project modules and pickle payloads are not executed.'},
        'resources': {'totals': totals, 'limits': protocol['limits'], 'by_phase': {k: dict(v) for k, v in phases.items()},
                      'seed_training_repairs': seed_calls, 'recorded_registration_counter_prefix': snapshot_lines,
                      'first_final_test_line': first_test, 'first_verification_line': first_verification,
                      'registration_limit': 'Snapshot counts match a pre-test ledger prefix; this does not independently authenticate historical file creation time.'},
        'alignment': {'cases': len(align['cases']), 'original_steps': original_steps, 'matched_prefix_steps': sum(prefix_lengths),
                      'prefix_lengths': prefix_lengths,
                      'scope': 'Saved semantic windows and accepted route equality; no new execution of either rule or repair.'},
        'labels': {'rows': len(labels), 'decision_states_by_seed_round': sum(g['states'] for g in groups),
                   'serialized_branches': branch_count, 'serialized_suffix_transitions': transition_count, 'groups': groups,
                   'seed2302_round1_one_and_multi_equal_in_saved_suffixes': equal_one_multi,
                   'equal_suffix_rows': len(seed2302_round1), 'compared_suffix_fields': equivalent_keys,
                   'scope': 'Recomputed from saved verification JSON, not independent reexecution against the original pickle training tables.'},
        'fits': {'optimizer_updates': fit_updates, 'logs': fits,
                 'multi_model_updates_per_seed': 512, 'total_updates_per_seed_across_shared_initialization_and_controls': 1280},
        'checkpoint_storage_comparisons': comparisons,
        'checkpoint_scope': 'Raw torch ZIP storage-block hashes only; no pickle decoding, tensor loading, model forward or parameter-name semantic reconstruction.',
        'multi_test_behavior': multi_test,
        'limits': [
            'Three training seeds also use different four-scene training subsets; observed variation combines data and optimization randomness.',
            'Menus and finalizations are instrumented shared-environment event counters, not all internal legacy solver scans or validator invocations.',
            'The 2067 stored suffix transitions include repeated parent STOP and shared prefixes; they are not 2067 independent labeled decision states.',
            'The 8192 repair budget is a ceiling; the planned two rounds and 3840 updates finished, so unused calls do not authorize more training.',
            'No external legacy dependencies, original teacher pickle, robot IK or physical validation were rerun.',
        ],
        'input_sha256': {name: sha(output / name) for name in ('protocol.json', 'resource.jsonl', 'resources.json', 'alignment.json',
                                                               'verified_labels.json.gz', 'label_verification.json', 'test_results.json')},
        'source_semantics_reviewed': {name: sha(source / name) for name in ('core.py', 'learning.py', 'trial.py', 'verify_labels.py', 'repair_counted.py')},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.is_relative_to(source):
        raise ValueError('Write review output outside the supplied source package')
    result = audit(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'all_saved_record_checks_passed': result['all_saved_record_checks_passed'], 'failures': result['failures'],
                      'repairs': result['resources']['totals']['nonstop'], 'saved_labels': result['labels']['rows'],
                      'serialized_suffix_transitions': result['labels']['serialized_suffix_transitions'], 'new_queries': 0}))
    if result['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
