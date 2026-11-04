"""Verify that cleanup retained every original file outside the approved roots."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    backup = ROOT / 'project_backups/2026-11-04_g23-closeout/full_before_cleanup.zip'
    plan = json.loads((ROOT / 'project_management/cleanup_plan.json').read_text(encoding='utf-8'))
    receipt = json.loads((ROOT / 'project_management/cleanup_receipt.json').read_text(encoding='utf-8'))
    with zipfile.ZipFile(backup) as z:
        manifest = json.loads(z.read('SNAPSHOT_MANIFEST.json'))
    roots = [entry['path'] for entry in plan['roots']]
    if any((ROOT / name).exists() for name in roots):
        raise ValueError('An approved residue target still exists')
    removed = checked = checked_bytes = 0
    failures = []
    for entry in manifest['files']:
        name = entry['path']
        if any(name == root or name.startswith(root + '/') for root in roots):
            removed += 1
            continue
        if name == 'project_backup_run.log':
            # Created by this backup operation itself and grew while the snapshot ran.
            continue
        path = ROOT / name
        if not path.is_file() or path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            failures.append(name)
        checked += 1
        checked_bytes += entry['bytes']
    if removed != plan['totals']['files'] or receipt['removed_file_count'] != removed:
        raise ValueError('Removed file counts disagree with approved plan and original snapshot')
    report = {'schema': 'rc3-cleanup-retention-check-v1', 'all_passed': not failures,
              'original_snapshot_files': len(manifest['files']), 'approved_removed_files': removed,
              'approved_removed_logical_bytes': plan['totals']['bytes'],
              'retained_original_files_checked': checked, 'retained_original_bytes_checked': checked_bytes,
              'failures': failures, 'all_approved_targets_absent': True,
              'excluded_generated_operation_log': 'project_backup_run.log',
              'local_python_environment_included_in_retention_check': True,
              'new_experiments_or_performance_measurements': 0,
              'scope': 'Every original snapshot file outside approved removal roots and the new operation log was checked against its original SHA256. New management files are covered by GIT_MANIFEST.json.'}
    (ROOT / 'project_management/retention_check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
