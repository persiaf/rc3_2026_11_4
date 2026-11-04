"""Summarize saved G2.2 timing fields only; never run or time a planner."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
from pathlib import Path


def stats(values):
    return {
        'n': len(values),
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'min': min(values),
        'max': max(values),
        'sum': sum(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result_path = args.source / 'outputs/evaluation/results.json'
    rows = json.loads(result_path.read_text(encoding='utf-8'))
    assert len(rows) == 72
    methods = {}
    for name in ('fixed3', 'detour', 'stop', 'detour_early', 'continued_ce', 'ce_gain'):
        group = [row for row in rows if row['method'] == name]
        assert len(group) == 12
        keys = [key for key, value in group[0]['timing'].items()
                if isinstance(value, (int, float))]
        item = {
            'complete_ms': stats([row['complete_s'] * 1000 for row in group]),
            'saved_component_ms': {
                key: stats([row['timing'][key] * 1000 for row in group]) for key in keys
            },
            'component_scope': 'Individually recorded fields; not assumed exclusive or exhaustive.',
        }
        if group[0]['shared_environment']:
            evidence = []
            for row in group:
                path = result_path.parent / f"case_{row['case_id']:02d}_{name}" / 'evidence.json.gz'
                with gzip.open(path, 'rt', encoding='utf-8') as stream:
                    evidence.append(json.load(stream))
            menu = sum(e['counters']['menu_work'] for e in evidence)
            proposal = sum(e['counters']['proposal_work'] for e in evidence)
            total_work = sum(row['work_used'] for row in group)
            assert menu + proposal == total_work
            calls = sum(len(e['policy_observations']) for e in evidence)
            item['work_proxy'] = {
                'menu_work': menu, 'proposal_work': proposal, 'total_work': total_work,
                'menu_fraction': menu / total_work,
                'scope': 'Deterministic accounting proxy, not a share of wall-clock time.',
            }
            item['policy_decisions'] = calls
            if calls:
                item['amortized_recorded_ms_per_decision'] = {
                    'graph': sum(row['timing']['graph_s'] for row in group) * 1000 / calls,
                    'inference': sum(row['timing']['inference_s'] for row in group) * 1000 / calls,
                    'scope': 'Existing recordings only; includes first-use effects; not a fresh benchmark.',
                }
        methods[name] = item
    output = {
        'scope': 'Read-only arithmetic over existing records; zero new plans, queries, inference, training or timing.',
        'source_results_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest(),
        'methods': methods,
        'interpretation_limits': [
            'Single timing per scene-method; 12 old development scenes.',
            'Do not infer stable speedup or a causal decomposition from group medians.',
            'Shared and original solvers use different preparation and validation paths.',
            'Training work reward does not account for all complete-planning CPU costs.',
            'Optimizations must be applied fairly to all methods in a new protocol.',
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'methods': len(methods), 'source_records': len(rows), 'new_benchmarks': 0}))


if __name__ == '__main__':
    main()
