"""Create and verify a lossless content-deduplicated pre-cleanup ZIP snapshot."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'project_backups/2026-11-04_g23-closeout'
ARCHIVE = DEST / 'full_before_cleanup.zip'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def files():
    for current, dirs, names in os.walk(ROOT, followlinks=False):
        base = Path(current)
        dirs[:] = sorted(d for d in dirs if d != '.git' and not (base == ROOT and d == 'project_backups'))
        for name in sorted(names):
            path = base / name
            if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
                raise ValueError('Unexpected link or path outside workspace')
            yield path


def create():
    DEST.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists():
        raise FileExistsError('Refusing to overwrite an existing backup')
    paths = list(files())
    entries = []
    seen = set()
    total = sum(p.stat().st_size for p in paths)
    done = 0
    last = time.monotonic()
    print(json.dumps({'phase': 'snapshot', 'files': len(paths), 'bytes': total}), flush=True)
    with zipfile.ZipFile(ARCHIVE, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for index, path in enumerate(paths, 1):
            stat = path.stat()
            key = digest(path)
            if key not in seen:
                z.write(path, 'objects/' + key)
                seen.add(key)
            if path.stat().st_size != stat.st_size or path.stat().st_mtime_ns != stat.st_mtime_ns:
                raise RuntimeError('Source changed during snapshot: ' + str(path))
            entries.append({'path': path.relative_to(ROOT).as_posix(), 'bytes': stat.st_size,
                            'mtime_ns': stat.st_mtime_ns, 'sha256': key})
            done += stat.st_size
            if time.monotonic() - last > 8:
                print(json.dumps({'phase': 'snapshot', 'files_done': index, 'files_total': len(paths),
                                  'bytes_done': done, 'bytes_total': total, 'unique_objects': len(seen)}), flush=True)
                last = time.monotonic()
        manifest = {'schema': 'rc3-lossless-snapshot-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
                    'source_root': str(ROOT), 'format': 'objects/SHA256 plus this path manifest; restore with tools/restore_project_backup.py',
                    'excluded': ['.git/ metadata (not present before version initialization)', 'project_backups/ (this backup and operation logs)'],
                    'includes_local_python_environment': True, 'files': entries}
        z.writestr('SNAPSHOT_MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8'))
        z.write(ROOT / 'tools/restore_project_backup.py', 'RESTORE_PROJECT.py')
    print(json.dumps({'phase': 'snapshot_written', 'archive_bytes': ARCHIVE.stat().st_size,
                      'files': len(entries), 'unique_objects': len(seen)}), flush=True)
    verify()


def verify():
    with zipfile.ZipFile(ARCHIVE) as z:
        manifest = json.loads(z.read('SNAPSHOT_MANIFEST.json'))
        required = {entry['sha256']: entry['bytes'] for entry in manifest['files']}
        assert len({entry['path'] for entry in manifest['files']}) == len(manifest['files'])
        assert set(z.namelist()) == {'objects/' + key for key in required} | {'SNAPSHOT_MANIFEST.json', 'RESTORE_PROJECT.py'}
        last = time.monotonic()
        for index, (key, size) in enumerate(required.items(), 1):
            h = hashlib.sha256(); actual = 0
            with z.open('objects/' + key) as stream:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    h.update(chunk); actual += len(chunk)
            if h.hexdigest() != key or actual != size:
                raise ValueError('Backup object integrity failed: ' + key)
            if time.monotonic() - last > 8:
                print(json.dumps({'phase': 'verify', 'objects_checked': index, 'objects_total': len(required)}), flush=True)
                last = time.monotonic()
    receipt = {'schema': 'rc3-snapshot-receipt-v1', 'archive': str(ARCHIVE.relative_to(ROOT)).replace('\\', '/'),
               'sha256': digest(ARCHIVE), 'archive_bytes': ARCHIVE.stat().st_size,
               'original_files': len(manifest['files']), 'original_bytes': sum(e['bytes'] for e in manifest['files']),
               'unique_objects': len(required), 'all_objects_sha256_verified': True,
               'manifest_sha256': hashlib.sha256(json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8')).hexdigest(),
               'includes_local_python_environment': True, 'verified_utc': datetime.now(timezone.utc).isoformat()}
    (DEST / 'snapshot_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    verify() if args.verify_only else create()
