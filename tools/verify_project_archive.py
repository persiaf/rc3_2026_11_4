"""Write/verify the SHA256 manifest of the Git project without project imports."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'project_management/GIT_MANIFEST.json'
EXCLUDED = {'project_management/GIT_MANIFEST.json', 'project_management/PUBLISH_RECEIPT.json'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_manifest():
    output = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT)
    names = sorted(set(n.decode('utf-8') for n in output.split(b'\0') if n))
    records = {}
    for name in names:
        if name in EXCLUDED:
            continue
        path = ROOT / name
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError('Unexpected link or path outside repository')
        records[name] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    version = json.loads((ROOT / 'PROJECT_VERSION.json').read_text(encoding='utf-8'))['version']
    result = {'schema': 'rc3-git-content-manifest-v1', 'version': version,
              'files': records, 'excluded_self_referential_metadata': sorted(EXCLUDED),
              'file_count': len(records), 'total_bytes': sum(e['bytes'] for e in records.values()),
              'scope': 'Original working-tree bytes after Git LFS checkout, not LFS pointers; no research execution.'}
    MANIFEST.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def verify():
    data = json.loads(MANIFEST.read_text(encoding='utf-8'))
    failures = []
    for name, expected in data['files'].items():
        pure = PurePosixPath(name)
        path = (ROOT / Path(*pure.parts)).resolve()
        if pure.is_absolute() or '..' in pure.parts or not path.is_relative_to(ROOT):
            raise ValueError('Unsafe manifest path')
        if not path.is_file() or path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            failures.append(name)
    result = {'all_passed': not failures, 'files_checked': len(data['files']),
              'bytes_checked': data['total_bytes'], 'failures': failures,
              'hint': 'Run git lfs pull if large files are still LFS pointers.'}
    print(json.dumps(result, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    if args.write:
        write_manifest()
    verify()
