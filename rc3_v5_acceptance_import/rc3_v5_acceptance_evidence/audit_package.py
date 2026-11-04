"""Read-only delivery integrity and optional frozen-v4 archive comparison."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--v4-archive', type=Path)
    args = parser.parse_args()
    root = args.source_root.resolve()
    out = args.output_dir.resolve()
    if out.is_relative_to(root):
        raise ValueError('Audit outputs must be outside frozen source tree')
    manifest = json.loads((root/'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
    records = manifest['files']
    if not records:
        raise ValueError('Empty file manifest')
    checks, seen = [], set()
    for item in records:
        name = item['path']; rel = PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or ':' in name or '\\' in name or name in seen:
            raise ValueError('Unsafe or repeated manifest path')
        seen.add(name)
        target = root.joinpath(*rel.parts)
        match = target.is_file() and target.stat().st_size == item['bytes'] and sha(target.read_bytes()) == item['sha256']
        checks.append({'path': name, 'matches': match})
    integrity = {'checked': len(checks), 'matches': sum(x['matches'] for x in checks),
                 'all_passed': all(x['matches'] for x in checks), 'entries': checks}
    out.mkdir(parents=True, exist_ok=True)
    (out/'package_integrity.json').write_text(json.dumps(integrity, ensure_ascii=False, indent=2), encoding='utf-8')
    passed = integrity['all_passed']
    summary = {'package_files': len(checks), 'package_passed': passed}
    if args.v4_archive:
        if sha(args.v4_archive.read_bytes()) != manifest['base_archive_sha256']:
            raise ValueError('v4 archive SHA does not match v5 base archive declaration')
        previous = []
        with zipfile.ZipFile(args.v4_archive) as archive:
            for entry in archive.infolist():
                if entry.is_dir():
                    continue
                name = 'V4_PACKAGE_MANIFEST.json' if entry.filename == 'PACKAGE_MANIFEST.json' else entry.filename
                rel = PurePosixPath(name)
                if rel.is_absolute() or '..' in rel.parts or ':' in name or '\\' in name:
                    raise ValueError('Unsafe archive path')
                target = root.joinpath(*rel.parts)
                match = target.is_file() and sha(archive.read(entry)) == sha(target.read_bytes())
                previous.append({'v4_path': entry.filename, 'v5_path': name, 'same': match})
        v4 = {'checked': len(previous), 'identical': sum(x['same'] for x in previous),
              'different': [x for x in previous if not x['same']]}
        (out/'v4_archive_comparison.json').write_text(json.dumps({'summary': v4, 'checks': previous}, ensure_ascii=False, indent=2), encoding='utf-8')
        passed = passed and not v4['different']
        summary['v4_comparison'] = v4
    summary['all_passed'] = passed
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
