"""Re-read saved plans and reconstruct the cache evidence, outside any timing."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from rc3_baseline.validator import validate


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    root = Path('rc3_v5/cache_results'); results = {}; records = []
    for directory, expected in (('equivalence', 36), ('benchmark', 108)):
        path = root/directory; manifest = read(path/'manifest.json'); summary = read(path/'summary.json')
        assert manifest['status'] == 'complete'
        assert all(sha(p) == h for p,h in manifest['source_sha256'].items())
        files = sorted((path/'plans').glob('scene_plan_*.json')); assert len(files) == 18
        count = 0
        for index,file in enumerate(files):
            item = read(file); reference = next(iter(item['plans'].values()))
            assert item['scene']['seed'] == manifest['cases'][index]['seed']
            for name,plan in item['plans'].items():
                assert plan == reference
                check = validate(item['scene'], plan, item['feasibility'])
                assert check['valid_model_plan'] and check['all_tasks_served']
                assert not item['runs'][name]['runtime']['failures']
                if directory == 'benchmark': records.append(item['runs'][name]['record'])
                count += 1
        assert count == expected == summary['saved_plans']
        results[directory] = dict(scenes=18, revalidated_saved_plans=count,
            all_complete_plan_objects_equal=True, all_independently_valid=True, source_hashes_match=True)

    heads = read(root/'equivalence/head_checks.json'); assert len(heads) == 144
    assert all(x['all_heads_bitwise_equal'] and x['max_output_error'] == 0.
               and x['original_action'] == x['cached_action'] for x in heads)
    results['equivalence']['bitwise_states_recounted'] = len(heads)
    csv_rows = list(csv.DictReader((root/'benchmark/repeat_metrics.csv').open(newline='',encoding='utf-8-sig')))
    indexed = {(int(r['scene']),r['method'],int(r['repeat'])):r for r in csv_rows}
    assert len(indexed) == len(records) == 108
    for r in records:
        row = indexed[(r['scene'],r['method'],r['repeat'])]
        for key,value in r.items():
            assert row[key] == str(value) or (isinstance(value,(int,float)) and float(row[key]) == value)
    group = {}; paired = []; reductions = []; faster = 0; slower = 0
    for index in range(18):
        medians = {}
        for method in ('original','cached'):
            selected = [r for r in records if r['scene'] == index and r['method'] == method]
            assert len(selected) == 3
            medians[method] = {key:float(np.median([r[key] for r in selected]))
                for key in ('planning_ms','model_ms','graph_ms','inference_ms')}
        ratio = medians['original']['planning_ms']/medians['cached']['planning_ms']
        paired.append(ratio); reductions.append(100*(1-1/ratio))
        faster += int(ratio > 1); slower += int(ratio < 1)
        group[index] = medians
    expected_group = {method:{key:float(np.median([r[method][key] for r in group.values()]))
                    for key in ('planning_ms','model_ms','graph_ms','inference_ms')}
                    for method in ('original','cached')}
    summary = read(root/'benchmark/summary.json')
    assert summary['summary'] == expected_group
    assert summary['median_paired_speedup'] == float(np.median(paired))
    assert abs(summary['median_paired_reduction_pct']-float(np.median(reductions))) < 1e-12
    assert summary['faster_scenes'] == faster and summary['slower_scenes'] == slower
    results['benchmark'].update(raw_records_match_csv=True, summaries_reconstructed=True,
        original_planning_ms=expected_group['original']['planning_ms'],
        cached_planning_ms=expected_group['cached']['planning_ms'],
        median_paired_speedup=float(np.median(paired)),
        median_paired_reduction_pct=float(np.median(reductions)), faster_scenes=faster,slower_scenes=slower)
    results['all_passed'] = True
    results['scope'] = 'Post-save readback audit, no new training, timing runs or scene counts.'
    (root/'artifact_reconstruction.json').write_text(json.dumps(results, indent=2),encoding='utf-8')
    print(json.dumps(results))


if __name__ == '__main__': main()
