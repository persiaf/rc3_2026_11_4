"""Registered final CPU comparisons and exploratory paired statistics.

Run only after the formal PPO receipt exists. This tool never selects a model
from validation scores and never reuses checkpoint-validation latency as a
deployment measurement. Production training/model modules are not changed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RULES = ('original_fixed3', 'original_detour', 'stop', 'detour_memory', 'ils', 'random')
PAIRS = ((8192, 512), (16384, 1024), (32768, 2048))
SCOPE = ('One training seed; the same 24 validation scenes are repeatedly evaluated. '
         'Scene-paired bootstrap intervals are exploratory, conditional on this seed '
         'and validation distribution; not independent confirmation or across-seed uncertainty.')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _key(row):
    return str(row['case_id']), row['seed'], row['mode']


def _index(rows, method):
    selected = {}
    for row in rows:
        if row['method'] != method:
            continue
        key = _key(row)
        if key in selected:
            raise ValueError('duplicate case/method row: ' + str((key, method)))
        selected[key] = row
    return selected


def _validate_rows(rows, cases, methods):
    expected = {(str(c.get('case_id', i)), c['seed'], c.get('mode', 'regular'))
                for i, c in enumerate(cases)}
    if len(expected) != len(cases) or len(rows) != len(cases)*len(methods):
        raise ValueError('evaluation has duplicate registered cases or incomplete rows')
    if {r['method'] for r in rows} != set(methods):
        raise ValueError('evaluation method identity differs')
    for method in methods:
        if set(_index(rows, method)) != expected:
            raise ValueError('evaluation row case identities differ')


def paired_statistics(rows_a, method_a, rows_b, method_b, *, bootstrap=10000, seed=117):
    """Positive improvement always favors A; failed/missing cases are counted."""
    a, b = _index(rows_a, method_a), _index(rows_b, method_b)
    common = sorted(set(a) & set(b))
    result = dict(method_a=method_a, method_b=method_b, n_a=len(a), n_b=len(b),
                  shared_cases=len(common), unmatched_a=len(set(a)-set(b)),
                  unmatched_b=len(set(b)-set(a)), scope=SCOPE, metrics={})
    for metric in ('J', 'total_reward', 'work_used', 'complete_s'):
        eligible = [k for k in common if a[k].get('status') == b[k].get('status') == 'complete'
                    and _number(a[k].get(metric)) and _number(b[k].get(metric))]
        if not eligible:
            result['metrics'][metric] = dict(n=0, excluded=len(common), reason='No comparable complete numeric pairs')
            continue
        raw = np.asarray([a[k][metric]-b[k][metric] for k in eligible], dtype=float)
        improvement = raw if metric == 'total_reward' else -raw
        rng = np.random.default_rng(seed)
        means = improvement[rng.integers(0, len(eligible), size=(bootstrap, len(eligible)))].mean(1)
        tolerance = 1e-9
        item = dict(n=len(eligible), excluded=len(common)-len(eligible),
                    a_minus_b_mean=float(raw.mean()), a_improvement_mean=float(improvement.mean()),
                    a_improvement_median=float(np.median(improvement)),
                    improvement_ci95_percentile=[float(x) for x in np.quantile(means, [.025, .975])],
                    wins=int((improvement > tolerance).sum()), ties=int((np.abs(improvement) <= tolerance).sum()),
                    losses=int((improvement < -tolerance).sum()), tolerance=tolerance,
                    bootstrap_replicates=bootstrap, bootstrap_seed=seed,
                    direction='higher return is better' if metric == 'total_reward' else 'lower is better')
        if metric == 'J':
            nonzero = [k for k in eligible if abs(b[k]['J']) > 1e-12]
            item['relative_J_reduction_percent_mean'] = (
                float(np.mean([100*(b[k]['J']-a[k]['J'])/b[k]['J'] for k in nonzero])) if nonzero else None)
            item['relative_J_pairs'] = len(nonzero)
        result['metrics'][metric] = item
    return result


def summarize_methods(rows):
    output = []
    for name in sorted({r['method'] for r in rows}):
        group = [r for r in rows if r['method'] == name]
        valid = [r for r in group if r.get('status') == 'complete']
        item = dict(method=name, attempted=len(group), complete=len(valid),
                    failed_or_runtime_error=len(group)-len(valid), metrics={},
                    invalid_plan_count=sum(not bool(r.get('valid')) for r in group),
                    fallback_count=sum(bool(r.get('fallback')) for r in group),
                    time_overrun_count=sum(_number(r.get('time_overrun_s')) and r['time_overrun_s'] > 0 for r in group),
                    work_overrun_count=sum(bool(r.get('work_overrun')) for r in group),
                    explicit_stop_count=sum(bool(r.get('explicit_stop')) for r in group),
                    stop_at_root_count=sum(bool(r.get('stop_at_root')) for r in group))
        for field in ('J', 'total_reward', 'work_used', 'complete_s', 'quality_gain_percent',
                      'proposal_calls', 'time_overrun_s', 'repeated_proposals', 'unchanged_proposals'):
            values = [r[field] for r in valid if _number(r.get(field))]
            item['metrics'][field] = (dict(n=len(values), mean=float(np.mean(values)),
                median=float(np.median(values)), p95=float(np.quantile(values, .95)),
                minimum=float(min(values)), maximum=float(max(values))) if values else dict(n=0))
        output.append(item)
    return output


def _export_csv(path, rows):
    flat = [{k: (json.dumps(v, sort_keys=True, separators=(',', ':')) if isinstance(v, (dict, list)) else v)
             for k, v in row.items()} for row in rows]
    fields = list(dict.fromkeys(k for row in flat for k in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)


def _run_or_read(cases, methods, destination, config, time_limit, policy_kwargs):
    from dataclasses import asdict
    from rc3_rl_g2.evaluate_g2 import evaluate_methods, _model_identity
    saved = destination / 'results.json'
    if not saved.exists():
        return evaluate_methods(cases, methods, destination, config=config,
                                time_limit_s=time_limit, policy_kwargs=policy_kwargs)
    request = _read(destination / 'request.json')
    if (request['cases'] != cases or request['config'] != asdict(config)
            or request['time_limit_s'] != time_limit or request['policy_kwargs'] != policy_kwargs
            or list(request['method_setup']) != list(methods)):
        raise ValueError('existing evaluation request differs: ' + str(destination))
    for name, model in methods.items():
        if model is not None and request['method_setup'][name]['model'] != _model_identity(model):
            raise ValueError('existing evaluation model differs: ' + name)
    rows = _read(saved)
    _validate_rows(rows, cases, methods)
    return rows


def build_summary(output, protocol, *, bootstrap=10000):
    sets = {name: _read(output/name/'results.json') for name in
            ('controls_fixed_work', 'final_fixed_work', 'final_soft_time_0400ms')}
    summary = dict(schema='rc3-g2-final-comparison-v1', scope=SCOPE,
        final_model_rule='PPO32768 and continued BC2048 are predeclared endpoints; no validation winner selection.',
        comparison_boundaries=dict(fixed_work='H16; 2,000,000 shared soft work; frozen originals have no comparable work/reward.',
            full_time='Complete scene-to-plan 0.4 s soft deadline, including preprocessing and final verification; overruns retained.',
            timing='Scene generation, disk model loading, JSON serialization outside complete_s; model loading reported separately.',
            intervals=f'Paired scene percentile bootstrap, {bootstrap} replicates; pooled estimates are unstratified; no multiple-comparison adjustment.',
            p95='Empirical linear-interpolation 95th percentile across scene rows.'),
        datasets={}, checkpoint_stage_comparisons=[], protocol_sha256=_sha(ROOT/'outputs/protocol.json'))
    combined = []
    for name, rows in sets.items():
        modes = sorted({r['mode'] for r in rows})
        data = dict(methods=summarize_methods(rows), by_mode={m: summarize_methods([r for r in rows if r['mode']==m]) for m in modes})
        if name.startswith('final_'):
            data['ppo_vs_methods'] = [paired_statistics(rows, 'ppo_final', rows, other, bootstrap=bootstrap,
                                                        seed=protocol['training_seed']+419)
                                      for other in sorted({r['method'] for r in rows}-{'ppo_final'})]
            data['ppo_vs_methods_by_mode'] = {m: [paired_statistics([r for r in rows if r['mode']==m], 'ppo_final',
                    [r for r in rows if r['mode']==m], other, bootstrap=bootstrap, seed=protocol['training_seed']+419)
                for other in sorted({r['method'] for r in rows}-{'ppo_final'})] for m in modes}
        summary['datasets'][name] = data
        combined.extend([dict(evaluation_set=name, **r) for r in rows])
        _export_csv(output/name/'rows.csv', rows)
    controls = sets['controls_fixed_work']
    for actions, updates in PAIRS:
        stage = ROOT/f'outputs/validation/ppo_{actions}'
        stage_rows = _read(stage/'results.json')
        request = _read(stage/'request.json')
        if request['cases'] != protocol['splits']['validation'] or request['config'] != protocol['config'] or request['time_limit_s'] is not None:
            raise ValueError('stage validation differs from registered cases/config/budget')
        ppo_name = f'ppo_{actions}'
        _validate_rows(stage_rows, protocol['splits']['validation'], [ppo_name])
        entry = dict(ppo_actual_actions=actions, continued_bc_updates=updates,
                     stage_results_sha256=_sha(stage/'results.json'),
                     scope='Stage quality/return diagnostics only; deployment time uses the fresh final evaluation directories.',
                     comparisons=[], comparisons_by_mode={})
        for comparator in ('bc_frozen', f'bc_continued_{updates}'):
            pair = paired_statistics(stage_rows, ppo_name, controls, comparator, bootstrap=bootstrap,
                                     seed=protocol['training_seed']+actions)
            # Stages were evaluated at different wall-clock points. Do not
            # misrepresent their timing as a matched deployment comparison.
            pair['metrics'].pop('complete_s', None)
            entry['comparisons'].append(pair)
        pair = paired_statistics(controls, f'bc_continued_{updates}', controls, 'bc_frozen',
                                 bootstrap=bootstrap, seed=protocol['training_seed']+actions)
        pair['metrics'].pop('complete_s', None)
        entry['comparisons'].append(pair)
        for mode in sorted({r['mode'] for r in stage_rows}):
            mode_stage = [r for r in stage_rows if r['mode'] == mode]
            mode_controls = [r for r in controls if r['mode'] == mode]
            mode_pairs = [paired_statistics(mode_stage, ppo_name, mode_controls, comparator,
                bootstrap=bootstrap, seed=protocol['training_seed']+actions)
                for comparator in ('bc_frozen', f'bc_continued_{updates}')]
            mode_pairs.append(paired_statistics(mode_controls, f'bc_continued_{updates}',
                mode_controls, 'bc_frozen', bootstrap=bootstrap, seed=protocol['training_seed']+actions))
            for comparison in mode_pairs:
                comparison['metrics'].pop('complete_s', None)
            entry['comparisons_by_mode'][mode] = mode_pairs
        summary['checkpoint_stage_comparisons'].append(entry)
    _export_csv(output/'rows.csv', combined)
    _write(output/'summary.json', summary)
    return summary


def run(output):
    from rc3_rl_g2.train_ppo import protocol as load_protocol, configure
    from rc3_rl_g2.checkpoints import load_model, source_identity
    from rc3_rl_research.types import SearchConfig
    protocol, digest = load_protocol()
    receipt = _read(ROOT/'outputs/training/ppo_receipt.json')
    if not receipt.get('completed') or receipt['actual_actions'] != 32768 or receipt['protocol_sha256'] != digest:
        raise ValueError('formal 32,768-action PPO completion receipt required')
    selection_path = ROOT/'outputs/rule_development/selected_ils.json'
    selection = _read(selection_path)
    if not selection.get('frozen') or selection.get('validation_used') or selection['selected'] != dict(stagnation_limit=2, max_perturbations=1):
        raise ValueError('expected frozen rule-development ILS2/1 selection')
    configure(protocol['training_seed'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = {'bc_frozen': ROOT/'outputs/training/bc_initial.pt',
             **{f'bc_continued_{n}': ROOT/f'outputs/training/bc_continued_{n:04d}.pt' for n in (512, 1024, 2048)},
             'ppo_final': ROOT/'outputs/training/ppo_32768.pt'}
    spec = dict(schema='rc3-g2-final-evaluation-request-v1', cases=protocol['splits']['validation'],
        config=protocol['config'], protocol_sha256=digest, scope=SCOPE, execution='single process CPU; torch thread count 1; rotated method order; fresh adapters',
        checkpoint_sha256={k:_sha(v) for k,v in paths.items()}, core_source_identity=source_identity(),
        tool_sha256=_sha(__file__), evaluator_sha256=_sha(ROOT/'evaluate_g2.py'),
        ils_selection_sha256=_sha(selection_path), selected_ils=selection['selected'],
        methods_controls=['bc_frozen','bc_continued_512','bc_continued_1024','bc_continued_2048',*RULES],
        methods_final=['ppo_final','bc_frozen','bc_continued_2048',*RULES],
        soft_time_limit_s=.4, bootstrap_replicates=10000, no_validation_model_selection=True,
        planned_evaluations=dict(controls=240, final_fixed=216, final_soft_time=216, total=672))
    registration = output/'evaluation_protocol.json'
    if registration.exists():
        if _read(registration) != spec:
            raise ValueError('final evaluation registration changed')
    else:
        _write(registration, spec)
    models = {}
    loads = []
    for name, path in paths.items():
        tick = time.perf_counter()
        model, payload = load_model(path, 'cpu')
        elapsed = time.perf_counter()-tick
        if payload['protocol_sha256'] != digest:
            raise ValueError('model protocol differs')
        if name != 'bc_frozen' and payload['parent_bc_sha256'] != spec['checkpoint_sha256']['bc_frozen']:
            raise ValueError('models do not share the registered BC initialization')
        models[name] = model.eval()
        loads.append(dict(method=name, path=str(path), sha256=_sha(path), disk_load_and_verify_s=elapsed,
            scope='Separate disk deserialization, identity checks and CPU construction; OS caches not flushed; process startup excluded.'))
    # Keep each invocation load record; never overwrite a prior timing attempt.
    load_path = output/f'model_loads_{time.time_ns()}.json'
    _write(load_path, loads)
    cases = protocol['splits']['validation']
    if len(cases) != 24:
        raise ValueError('registered validation must contain 24 cases')
    config = SearchConfig(**protocol['config'])
    kwargs = {'ils': selection['selected']}
    controls = {name: models.get(name) for name in spec['methods_controls']}
    final = {name: models.get(name) for name in spec['methods_final']}
    _run_or_read(cases, controls, output/'controls_fixed_work', config, None, kwargs)
    _run_or_read(cases, final, output/'final_fixed_work', config, None, kwargs)
    _run_or_read(cases, final, output/'final_soft_time_0400ms', config, .4, kwargs)
    return build_summary(output, protocol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/final_evaluation')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if args.summarize_only:
        result = build_summary(args.output, _read(ROOT/'outputs/protocol.json'))
    else:
        result = run(args.output)
    print(json.dumps(dict(scope=result['scope'], output=str(args.output), datasets=list(result['datasets']))))


if __name__ == '__main__':
    main()
