"""Recompute teacher labels from saved terminal records, without new searches."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from rc3_rl_g2.teacher_dataset import load_teacher, semantic_action


def _close(a, b, message):
    if not np.isfinite(a) or not np.isfinite(b) or abs(a-b) > 1e-7:
        raise AssertionError(message+f': {a} != {b}')


def audit_teacher(directory):
    directory = Path(directory)
    spec = json.loads((directory/'collection_spec.json').read_text(encoding='utf-8'))
    summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    if hashlib.sha256((directory/'teacher.pkl').read_bytes()).hexdigest() != summary['teacher_sha256']:
        raise AssertionError('teacher artifact hash differs from recorded collection')
    payload = load_teacher(directory/'teacher.pkl')
    by_scene = defaultdict(list)
    for sample in payload['samples']:
        by_scene[sample['scene_id']].append(sample)
    checks, total_queries, evidence_count, unknown_count = [], 0, 0, 0
    label_counts = Counter()
    for index, case in enumerate(spec['cases']):
        target = directory/f'case_{index:04d}'
        result = json.loads((target/'summary.json').read_text(encoding='utf-8'))
        if result['status'] != 'complete':
            raise AssertionError('incomplete teacher case')
        with gzip.open(target/'queries.jsonl.gz', 'rt', encoding='utf-8') as f:
            queries = [json.loads(line) for line in f]
        with gzip.open(target/'menus.jsonl.gz', 'rt', encoding='utf-8') as f:
            menus = [json.loads(line) for line in f]
        query_by_path = {tuple(q['path']): q for q in queries}
        menu_by_path = {tuple(m['path']): m for m in menus}
        if len(query_by_path) != len(queries) or len(menu_by_path) != len(menus):
            raise AssertionError('duplicate path identities')
        scene_id = str(case.get('case_id', case.get('scene_id', index)))+':'+str(case['seed'])
        samples = by_scene[scene_id]
        sample_by_path = {tuple(s['path']): s for s in samples}
        if len(samples) != len(menus) or set(sample_by_path) != set(menu_by_path):
            raise AssertionError('sample state set differs from expanded menu set')
        source = json.loads((target/'input.json').read_text(encoding='utf-8'))
        anchor = source['anchor_cost']
        config = spec['config']
        counter = Counter()
        for q in queries:
            prefix = tuple(q['path'][:-1])
            menu = menu_by_path[prefix]
            if q['action_id'] not in menu['queried_action_ids']:
                raise AssertionError('query outside registered evidence subset')
            if q['action'] != menu['observation']['actions'][q['action_id']]:
                raise AssertionError('query changed ordered semantic action')
            if not menu['observation']['mask'][q['action_id']]:
                raise AssertionError('queried action is illegal')
            prefix_reward = 0. if not prefix else query_by_path[prefix]['total_reward']
            _close(q['prefix_reward'], prefix_reward, 'prefix cumulative reward')
            _close(q['total_reward'], prefix_reward+q['step_reward']+q['terminal_event_reward'],
                   'step plus exactly-once terminal event')
            if q['leaf']:
                expected = 100.*(anchor-q['finalization']['returned_cost'])/max(anchor, 1e-8)
                expected -= config['lambda_work']*q['work_used']/config['work_scale']
                _close(q['total_reward'], expected, 'closed episode return')
            # Recompute the actual best suffix rather than trust the supplied
            # best_observed_terminal_reward or winning path fields.
            path = tuple(q['path'])
            descendant_leaves = [r for p, r in query_by_path.items()
                                 if r['leaf'] and p[:len(path)] == path]
            observed_best = max(r['total_reward'] for r in descendant_leaves)
            _close(q['best_observed_terminal_reward'], observed_best, 'highest descendant return')
            _close(q['remaining_return'], observed_best-prefix_reward, 'remaining return label')
            counter['selected_edges'] += 1
            counter['proposal_calls'] += int(q['info']['proposal'] is not None)
            counter['proposal_work_units'] += q['info']['proposal_work_units']
        for path, sample in sample_by_path.items():
            menu = menu_by_path[path]
            if sample['teacher_depth'] != menu['remaining_depth']:
                raise AssertionError('finite teacher depth was discarded')
            if sample['ordered_actions'] != [semantic_action(a) for a in menu['observation']['actions']]:
                raise AssertionError('teacher semantic action mapping changed')
            expected_evidence = np.zeros(len(sample['legality_mask']), dtype=bool)
            expected_evidence[menu['queried_action_ids']] = True
            if not np.array_equal(expected_evidence, sample['evidence_mask']):
                raise AssertionError('unknown legal actions were mislabeled as evidence')
            evidence_count += int(expected_evidence.sum())
            unknown_count += int((sample['legality_mask'] & ~expected_evidence).sum())
            expected_q = np.zeros(len(expected_evidence))
            for action_id in menu['queried_action_ids']:
                expected_q[action_id] = query_by_path[path+(action_id,)]['remaining_return']
            if not np.allclose(expected_q, sample['q_values'], atol=1e-7, rtol=0):
                raise AssertionError('saved Q differs from independently recomputed suffix return')
            best_q = expected_q[expected_evidence].max()
            ties = expected_evidence & (expected_q >= best_q-spec['tie_tolerance'])
            if not np.array_equal(ties, sample['ties']):
                raise AssertionError('near-tie set differs')
            label_counts['states'] += 1
            label_counts['stop_in_near_tie_set'] += int(ties[sample['stop_id']])
            label_counts['multiple_near_ties'] += int(ties.sum() > 1)
            if not path:
                label_counts['root_states'] += 1
                label_counts['root_stop_in_near_tie_set'] += int(ties[sample['stop_id']])
        for key in ('selected_edges', 'proposal_calls', 'proposal_work_units'):
            if counter[key] != result['physical_oracle_work'][key]:
                raise AssertionError('actual query work count differs: '+key)
        _close(sum(s['weight'] for s in samples), 1./len(by_scene), 'equal scene mass')
        routes = defaultdict(list)
        for sample in samples:
            routes[tuple(sample['current_route'])].append(sample)
        for repeated in routes.values():
            _close(sum(s['weight'] for s in repeated), 1./len(by_scene)/len(routes),
                   'equal route mass within scene')
        checks.append(dict(case_id=case.get('case_id', index), seed=case['seed'], passed=True,
                           queries=len(queries), samples=len(samples)))
        total_queries += len(queries)
    if total_queries != summary['query_count']:
        raise AssertionError('aggregate query count differs')
    return dict(all_passed=True, cases=len(checks), samples=len(payload['samples']),
        queries=total_queries, observed_action_labels=evidence_count, unknown_legal_actions=unknown_count,
        label_description=dict(label_counts),
        checks=checks, teacher_sha256=summary['teacher_sha256'],
        scope='Offline independent computation from all raw query/menu records. Recomputes highest suffix returns, prefix removal, terminal fees/corrections, evidence masks, ties and scene/route weights; not a rerun of teacher search or PPO rollout.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    args = p.parse_args()
    report = audit_teacher(args.directory)
    (args.directory/'audit_receipt.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'checks'}, ensure_ascii=False))
