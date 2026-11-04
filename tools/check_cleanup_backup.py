"""Verify every proposed cleanup file against the completed full backup."""
from __future__ import annotations
from datetime import datetime, timezone
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
    plan_path = ROOT / 'project_management/cleanup_plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    receipt_path = ROOT / 'project_backups/2026-11-04_g23-closeout/snapshot_receipt.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    archive = (ROOT / receipt['archive']).resolve()
    if not archive.is_relative_to(ROOT / 'project_backups') or not receipt['all_objects_sha256_verified']:
        raise ValueError('Completed verified backup is required')
    if digest(archive) != receipt['sha256']:
        raise ValueError('Backup ZIP checksum mismatch')
    with zipfile.ZipFile(archive) as z:
        entries = {e['path']: e for e in json.loads(z.read('SNAPSHOT_MANIFEST.json'))['files']}
    protected = set()
    frozen_archives = list(ROOT.glob('rc3_*delivery*.zip')) + [
        ROOT / 'rc3_v5_1_maintenance.zip', ROOT / 'rc3_paper_figures_v1.zip',
        ROOT / 'rc3_rl_research/outputs/rc3_rl_research_g01.zip']
    for frozen in frozen_archives:
        with zipfile.ZipFile(frozen) as z:
            protected.update(name for name in z.namelist() if not name.endswith('/') and (ROOT / name).is_file())
    # Compare exact live files to backup; reject extra or altered files before deletion.
    checked = 0
    total = 0
    for entry in plan['roots']:
        path = (ROOT / entry['path']).resolve()
        if path == ROOT or not path.is_relative_to(ROOT) or path.is_symlink():
            raise ValueError('Cleanup target escapes workspace or is a link')
        if any(part in {'.git', '.venv', 'project_backups'} for part in path.relative_to(ROOT).parts):
            raise ValueError('Protected cleanup target')
        candidates = [path] if path.is_file() else [p for p in path.rglob('*') if p.is_file()]
        if len(candidates) != entry['files'] or sum(p.stat().st_size for p in candidates) != entry['bytes']:
            raise ValueError('Cleanup target content changed: ' + entry['path'])
        for file in candidates:
            if file.is_symlink() or not file.resolve().is_relative_to(path if path.is_dir() else path.parent):
                raise ValueError('Unexpected cleanup link')
            name = file.relative_to(ROOT).as_posix()
            if name in protected:
                raise ValueError('Refusing to delete frozen archive payload: ' + name)
            original = entries.get(name)
            if not original or file.stat().st_size != original['bytes'] or digest(file) != original['sha256']:
                raise ValueError('File is not safely backed up: ' + name)
            checked += 1; total += original['bytes']
    if checked != plan['totals']['files'] or total != plan['totals']['bytes']:
        raise ValueError('Cleanup totals disagree')
    authorization = {'schema': 'rc3-cleanup-precheck-v1', 'ready_for_cleanup': True,
                     'plan_sha256': digest(plan_path), 'backup_sha256': receipt['sha256'],
                     'backed_up_files_verified': checked, 'backed_up_bytes_verified': total,
                     'checked_utc': datetime.now(timezone.utc).isoformat()}
    (ROOT / 'project_management/cleanup_precheck.json').write_text(json.dumps(authorization, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(authorization), flush=True)


if __name__ == '__main__':
    main()
