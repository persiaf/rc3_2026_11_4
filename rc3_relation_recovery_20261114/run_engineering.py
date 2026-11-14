"""Predebit and archive bounded synthetic test invocations; no scientific imports."""
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HOME = Path(__file__).resolve().parent
ROOT = HOME.parent


def write(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())


def main():
    plan = json.loads((HOME / 'ENGINEERING_PLAN.json').read_text(encoding='utf8'))
    results = HOME / 'engineering_results'
    results.mkdir(exist_ok=True)
    debits = results / 'SUITE_DEBITS.jsonl'
    previous = [json.loads(s) for s in debits.read_text(encoding='utf8').splitlines()] if debits.exists() else []
    invocations = [x for x in previous if x['event'] == 'begin']
    if len(invocations) >= plan['limits']['test_suite_invocations']:
        raise RuntimeError('Registered suite invocation limit exhausted')
    spent = sum(x.get('wall_s', 0) for x in previous if x['event'] == 'end')
    used_transactions = sum(x.get('synthetic_begin_calls', 0) for x in previous if x['event'] == 'end')
    remaining_transactions = plan['limits']['synthetic_transactions_total'] - used_transactions
    if remaining_transactions <= 0:
        raise RuntimeError('Cumulative synthetic transaction budget exhausted')
    if len(invocations) != sum(x['event'] == 'end' for x in previous):
        raise RuntimeError('Prior unclosed engineering attempt; do not reset its debit')
    if spent >= plan['limits']['wall_s_total']:
        raise RuntimeError('Engineering wall exhausted')
    from .persistence import append, sha
    number = len(invocations) + 1
    source = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(HOME.glob('*.py'))}
    source[(HOME / 'ENGINEERING_PLAN.json').relative_to(ROOT).as_posix()] = sha(HOME / 'ENGINEERING_PLAN.json')
    binding_path = results / f'ATTEMPT_{number:02}_SOURCE.json'
    write(binding_path, dict(files=source, python=sys.version, executable=sys.executable,
                             platform=platform.platform(), user_authorization='Current engineering-only request'))
    append(debits, dict(event='begin', attempt=number, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        source_binding_sha256=sha(binding_path), maximum_scientific_calls=0))
    begin = time.perf_counter()
    output = results / f'ATTEMPT_{number:02}.json'
    child = subprocess.run([sys.executable, '-B', '-X', 'utf8', '-m', __package__ + '.run_engineering',
                            '--child', str(output)], cwd=ROOT, capture_output=True, text=True,
                           encoding='utf8', timeout=plan['limits']['wall_s_total'] - spent,
                           env=dict(os.environ, RC3_ENGINEERING_TRANSACTION_CAP=str(remaining_transactions)))
    wall = time.perf_counter() - begin
    (results / f'ATTEMPT_{number:02}_PROCESS.log').write_text(child.stdout + child.stderr, encoding='utf8')
    report = json.loads(output.read_text(encoding='utf8')) if output.exists() else {'success': False}
    append(debits, dict(event='end', attempt=number, wall_s=wall, exit_code=child.returncode,
                        success=report['success'], synthetic_begin_calls=report.get('synthetic_begin_calls', 0)))
    print(json.dumps({k: report.get(k) for k in ('tests', 'failures', 'errors', 'success', 'synthetic_begin_calls')}))
    print(report.get('transcript', 'No test report'))
    raise SystemExit(0 if report['success'] and child.returncode == 0 else 1)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--child':
        from .test_persistence import run_suite
        report = run_suite()
        write(Path(sys.argv[2]), report)
        raise SystemExit(0 if report['success'] else 1)
    main()
