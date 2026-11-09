"""Read-only delivery consistency checks; never instantiate scientific objects."""
import ast
import csv
import hashlib
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOME = ROOT / 'rc3_gl_v4_20261109'


def read(p):
    return json.loads(p.read_text(encoding='utf8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def write_new(p, obj):
    with p.open('x', encoding='utf8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    start = time.perf_counter()
    binding = read(HOME / 'SOURCE_BINDING.json')
    for name, h in binding['source_sha256'].items():
        need(sha(ROOT / name) == h, 'Frozen runtime source changed: ' + name)
    for p in (ROOT / 'project_management/gl_v4_delivery').glob('*.py'):
        ast.parse(p.read_text(encoding='utf8'), filename=str(p))
    required = ['DEPLOYMENT_EQUIVALENCE.md', 'DEPLOYMENT_BENCHMARK.csv',
        'START_OBJECTIVE_ABLATION.md', 'TRAINING_AND_ROUTE_RESULTS.csv',
        'RESOURCE_ACCOUNTING.json', 'INDEPENDENT_REVIEW.md', 'NEXT_STEP_DECISION.md',
        'IMPLEMENTATION_REVIEW.md', 'EXPERIMENT_REPORT.md', 'RESULT_TO_CLAIM.md',
        'METHOD_IDENTITY_AUDIT.json', 'METHOD_IDENTITY_CLARIFICATION.json', 'GIT_VERSION_AUDIT.json']
    for name in required:
        need((HOME / name).is_file(), 'Missing delivery: ' + name)
    links = 0
    for p in HOME.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)', p.read_text(encoding='utf8')):
            if re.match(r'\w+://', target) or target.startswith('#'):
                continue
            target = target.split('#')[0]
            need((p.parent / target).exists(), 'Broken link: ' + str(p) + ' -> ' + target)
            links += 1
    with (HOME / 'TRAINING_AND_ROUTE_RESULTS.csv').open(encoding='utf8', newline='') as f:
        rows = list(csv.DictReader(f))
    counts = Counter(r['row_type'] for r in rows)
    need(len(rows) == 600 and sorted(counts.values()) == [216, 384], 'Wrong row type counts')
    audit = read(HOME / 'INDEPENDENT_AUDIT.json')
    identity = read(HOME / 'METHOD_IDENTITY_AUDIT.json')
    need(audit['status'] == 'PASS' and identity['counts']['legacy_default_aliases'] == 108,
         'Missing disclosed identity finding')
    r = read(HOME / 'RESOURCE_ACCOUNTING.json')
    need(r['deployment_requests'] == 396, 'Wrong deployment count')
    for phase, cap in [('DEPLOYMENT', 1800), ('SCIENCE', 7200)]:
        p = r['phase_resources'][phase]
        need(p['status'] == 'COMPLETE' and p['error'] is None and
             math.isfinite(p['wall_s']) and 0 < p['wall_s'] <= cap, 'Unclosed/capped phase')
    abnormal = [p.relative_to(ROOT).as_posix() for p in (HOME / 'runs').rglob('*')
                if p.is_file() and any(k in p.name.lower() for k in ('pending', 'emergency', 'fault', 'retry'))]
    need(not abnormal, 'Unexpected execution fault evidence: ' + str(abnormal))
    summary = read(HOME / 'results/SUMMARY.json')
    need(summary['candidate_screen'] is False, 'Negative candidate conclusion changed')
    closure = dict(status='PASS_WITH_DISCLOSED_ADAPTER_METADATA_DEFECT',
        frozen_runtime_files=len(binding['source_sha256']), source_binding_sha256=sha(HOME / 'SOURCE_BINDING.json'),
        required_deliveries=required, relative_links_resolved=links, combined_csv_rows=dict(counts),
        phase_budgets='UNCHANGED_COMPLETE', incomplete_or_retry_receipts=abnormal,
        identity_clarification_sha256=sha(HOME / 'METHOD_IDENTITY_CLARIFICATION.json'),
        independent_audit_sha256=sha(HOME / 'INDEPENDENT_AUDIT.json'),
        optimizer_updates=0, model_inferences=0, navigation_solves=0, native_calls=0, H_calls=0,
        wall_s=time.perf_counter() - start)
    write_new(HOME / 'DELIVERY_VERIFICATION.json', closure)
    write_new(HOME / 'MAINTENANCE_RESOURCE_ACCOUNTING.json', dict(
        scope='Post-run analysis/audit/clarification/delivery validation, separate from sealed scientific phase wall; no budget refunds or scientific retries',
        read_only_analysis_wall_s=read(HOME / 'results/READ_ONLY_ANALYSIS_CLOSURE.json')['wall_s'],
        independent_audit_wall_s=audit['wall_s'], method_identity_audit_wall_s=identity['wall_s'],
        closeout_validation_wall_s=closure['wall_s'],
        phase_resources_preserved=r['phase_resources'], old_frozen_files_byte_verified=audit['old_frozen_files_unchanged'],
        scientific_execution_retries=0, scientific_failures=0, metadata_defects_disclosed=1,
        new_optimizer_updates=0, new_model_inferences=0, new_nav_solves=0, new_native_calls=0, new_H_calls=0,
        note='Reported intervals are measured suboperations, not a fabricated exact total for interactive report editing or Git transport. Git version/LFS work is maintenance, not an additional planning/performance experiment.'))
    print(dict(status=closure['status'], links=links, frozen_runtime=len(binding['source_sha256']), csv_rows=counts))


if __name__ == '__main__':
    main()
