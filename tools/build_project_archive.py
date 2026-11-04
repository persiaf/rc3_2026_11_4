"""Package the clean, committed project with REAL Git LFS file bytes (stdlib)."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'project_backups/2026-11-04_g23-closeout'
ARCHIVE = DEST / 'rc3_2026_11_4_project.zip'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    if git('status', '--porcelain', '-z'):
        raise ValueError('Commit all project changes before building its final archive')
    head = git('rev-parse', 'HEAD').decode().strip()
    names = sorted(n.decode('utf-8') for n in git('ls-files', '-z').split(b'\0') if n)
    manifest = json.loads((ROOT / 'project_management/GIT_MANIFEST.json').read_text(encoding='utf-8'))
    expected = dict(manifest['files'])
    for name in manifest['excluded_self_referential_metadata']:
        if name in names:
            expected[name] = {'sha256': digest(ROOT / name), 'bytes': (ROOT / name).stat().st_size}
    if set(names) != set(expected):
        raise ValueError('Tracked file set differs from final content manifest')
    snapshot = {'version': '2026.11.04-g23-closeout', 'git_commit': head,
                'tag': 'snapshot-2026-11-04-g23-closeout', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'file_count': len(names), 'contains_actual_lfs_payloads': True,
                'excluded': ['Git metadata', 'virtual environment', 'approved removed caches and migration copies', 'local backup directory'],
                'restore': 'Extract rc3_2026_11_4/ to a new directory; install runtime requirements as described in README.md.'}
    DEST.mkdir(parents=True, exist_ok=True)
    last = time.monotonic()
    compressed = {'.zip', '.pt', '.npz', '.png', '.jpg', '.jpeg', '.pdf', '.docx', '.gz'}
    with zipfile.ZipFile(ARCHIVE, 'x', zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for index, name in enumerate(names, 1):
            path = ROOT / name
            z.write(path, 'rc3_2026_11_4/' + name,
                    compress_type=zipfile.ZIP_STORED if path.suffix.lower() in compressed else zipfile.ZIP_DEFLATED)
            if time.monotonic() - last > 8:
                print(json.dumps({'phase': 'package', 'files_done': index, 'files_total': len(names)}), flush=True)
                last = time.monotonic()
        z.writestr('ARCHIVE_SNAPSHOT.json', json.dumps(snapshot, ensure_ascii=False, indent=2).encode('utf-8'))
    last = time.monotonic()
    with zipfile.ZipFile(ARCHIVE) as z:
        if len(z.namelist()) != len(names) + 1:
            raise ValueError('Unexpected final ZIP members')
        for index, (name, entry) in enumerate(expected.items(), 1):
            h = hashlib.sha256(); size = 0
            with z.open('rc3_2026_11_4/' + name) as stream:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    h.update(chunk); size += len(chunk)
            if h.hexdigest() != entry['sha256'] or size != entry['bytes']:
                raise ValueError('Final ZIP payload mismatch: ' + name)
            if time.monotonic() - last > 8:
                print(json.dumps({'phase': 'verify_package', 'files_done': index, 'files_total': len(names)}), flush=True)
                last = time.monotonic()
    receipt = {**snapshot, 'archive': str(ARCHIVE.relative_to(ROOT)).replace('\\', '/'),
               'archive_sha256': digest(ARCHIVE), 'archive_bytes': ARCHIVE.stat().st_size,
               'raw_project_bytes': sum(e['bytes'] for e in expected.values()),
               'all_members_crc_and_sha256_verified': True}
    (DEST / 'project_archive_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
