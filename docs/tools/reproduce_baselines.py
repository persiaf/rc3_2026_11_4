"""Two bounded engineering examples; never imports a trainer or reads saved datasets."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.relative_to(root)
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(
        scope='Two newly generated engineering plans, not a performance experiment',
        utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
        executable=sys.executable, platform=platform.platform(),
        processor=platform.processor(), cpu_count=os.cpu_count(),
        dependencies={n: importlib.metadata.version(n)
                      for n in ('numpy', 'scipy', 'threadpoolctl')},
        base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                             text=True).strip(),
        scene=dict(scale='4x3', mode='regular', seed=27182818,
                   origin='make_complex_scene; no saved input dataset'),
        commands=[], optimizer_updates=0, weight_loads=0,
        sealed60_semantic_reads=0, S3_semantic_reads=0,
        plan_requests_predebit=0, independent_validation_requests=0,
    )
    receipt_path = output / 'VERIFICATION.json'

    def save():
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2)
                                + '\n', encoding='utf-8')

    def run(module, arguments, kind):
        command = [sys.executable, '-B', '-X', 'utf8', '-m', module, *arguments]
        if kind == 'plan':
            receipt['plan_requests_predebit'] += 1
        else:
            receipt['independent_validation_requests'] += 1
        row = dict(argv=command, kind=kind, timeout_s=180, status='STARTED')
        receipt['commands'].append(row)
        save()
        started = time.perf_counter()
        try:
            result = subprocess.run(command, cwd=root, capture_output=True,
                                    text=True, encoding='utf-8', timeout=180)
            row.update(exit_code=result.returncode, stdout=result.stdout,
                       stderr=result.stderr, status='PASS' if result.returncode == 0 else 'FAIL')
        except subprocess.TimeoutExpired:
            row.update(status='TIMEOUT', exit_code=None)
            raise
        finally:
            row['elapsed_s'] = time.perf_counter() - started
            save()
        if result.returncode:
            raise RuntimeError(f'{module} failed; see {receipt_path}')

    started = time.perf_counter()
    for method, budget in (('fixed3', 0), ('detour', 16)):
        name = 'fixed3' if method == 'fixed3' else 'detour16'
        plan = output / f'{name}.json'
        report = output / f'{name}_validation.json'
        run('rc3_v5.plan', ['--method', method, '--budget', str(budget),
                          '--scale', '4x3', '--mode', 'regular', '--seed', '27182818',
                          '--output', str(plan)], 'plan')
        run('rc3_baseline.validator', [str(plan), '--report', str(report)], 'validate')
        data = json.loads(plan.read_text(encoding='utf-8-sig'))
        check = json.loads(report.read_text(encoding='utf-8-sig'))
        if not (check['all_valid'] and check['all_tasks_served']):
            raise RuntimeError('Saved plan failed independent validation')
        p = data['plans'][method]
        receipt[name] = dict(status=data['status'], tasks=len(data['scene']['tasks']),
                             candidates=len(data['scene']['candidates']),
                             J=p['objective_length_plus_stops_m'], L=p['route_length_m'],
                             K=len(p['selected_candidate_ids']),
                             dp_calls=data['diagnostics']['dp_calls'],
                             planning_timing=data['timing'], validation=check)
    first = json.loads((output / 'fixed3.json').read_text(encoding='utf-8-sig'))
    second = json.loads((output / 'detour16.json').read_text(encoding='utf-8-sig'))
    if first['scene'] != second['scene']:
        raise RuntimeError('Examples must use the same generated scene')
    receipt['distinct_new_engineering_scenes'] = 1
    receipt['total_elapsed_s'] = time.perf_counter() - started
    receipt['artifacts'] = {
        p.name: dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size)
        for p in sorted(output.iterdir()) if p.is_file() and p != receipt_path
    }
    receipt['status'] = 'PASS_TWO_PROXY_PLANS'
    save()
    print(json.dumps({k: receipt[k] for k in ('status', 'total_elapsed_s', 'fixed3', 'detour16')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
