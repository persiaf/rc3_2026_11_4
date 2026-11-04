"""Fail-closed, read-only file-integrity audit for frozen experiment manifests.

This complements (does not replace) independent geometric plan validation.
Every absent, changed, unreadable or malformed expected file enters ``errors``;
the CLI writes ``all_passed: false`` and exits 2 on any such failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_reference(raw, root, relocate_from=None):
    """Resolve portable relative paths, with explicit old-root relocation only."""
    if not isinstance(raw, str) or not raw:
        raise ValueError('manifest path must be a nonempty string')
    root = Path(root).resolve()
    win = PureWindowsPath(raw)
    posix = PurePosixPath(raw.replace('\\', '/'))
    if '..' in posix.parts:
        raise ValueError('parent traversal in manifest path')
    if win.is_absolute() or posix.is_absolute():
        if relocate_from is not None:
            old = PureWindowsPath(relocate_from) if win.is_absolute() else PurePosixPath(relocate_from)
            source = win if win.is_absolute() else posix
            try:
                relative = source.relative_to(old)
            except ValueError as error:
                raise ValueError('absolute path is outside explicit original root') from error
            target = root.joinpath(*relative.parts)
        else:
            target = Path(raw)
    else:
        if win.drive or any(':' in part for part in posix.parts):
            raise ValueError('drive-relative or colon path is unsupported')
        target = root.joinpath(*posix.parts)
    target = target.resolve()
    if target != root and root not in target.parents:
        raise ValueError('resolved manifest path is outside audit root')
    return target


def manifest_entries(manifest):
    """Accept original results manifests or file-list package manifests."""
    if not isinstance(manifest, dict):
        raise ValueError('manifest must be an object')
    entries = []
    if 'files' in manifest:
        if not isinstance(manifest['files'], list):
            raise ValueError('files must be a list')
        for row in manifest['files']:
            if not isinstance(row, dict):
                raise ValueError('file entry must be an object')
            entries.append(('package', row['path'], row['sha256'], row.get('bytes')))
    elif 'source_sha256' in manifest:
        for group in ('source_sha256', 'data_sha256'):
            values = manifest.get(group, {})
            if not isinstance(values, dict):
                raise ValueError(f'{group} must be an object')
            entries.extend((group, path, digest, None) for path, digest in values.items())
        for path_key, hash_key, label in (
                ('model_path', 'model_sha256', 'model'),
                ('wall_config_path', 'wall_config_sha256', 'wall_config')):
            if path_key in manifest or hash_key in manifest:
                entries.append((label, manifest[path_key], manifest[hash_key], None))
        if not manifest.get('source_sha256'):
            raise ValueError('source hash manifest is empty')
    else:
        raise ValueError('unsupported manifest schema: expected files or source_sha256')
    if not entries:
        raise ValueError('manifest contains no integrity checks')
    return entries


def audit_manifest(manifest_path, root, prior_report=None, relocate_from=None):
    errors, checks = [], []
    report = dict(schema='rc3-integrity-audit-v1', all_passed=False, errors=errors,
                  manifest=str(manifest_path), root=str(Path(root).resolve()),
                  checks=checks, scope='File integrity only; no new geometric or physical validation.')
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
        report['manifest_sha256'] = sha256(manifest_path)
        entries = manifest_entries(manifest)
    except Exception as error:
        errors.append(f'manifest load/schema: {error}')
        entries = []
    for group, raw, expected, expected_bytes in entries:
        item = dict(group=group, path=raw, expected_sha256=expected, unchanged=False)
        checks.append(item)
        try:
            if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', expected):
                raise ValueError('expected SHA256 must contain 64 hexadecimal characters')
            if expected_bytes is not None and (type(expected_bytes) is not int or expected_bytes < 0):
                raise ValueError('expected bytes must be a nonnegative integer')
            path = resolve_reference(raw, root, relocate_from)
            item['resolved_path'] = str(path)
            if not path.is_file():
                raise ValueError('file is missing or is not a regular file')
            item['actual_sha256'] = sha256(path)
            item['actual_bytes'] = path.stat().st_size
            item['unchanged'] = (item['actual_sha256'] == expected.lower() and
                                 (expected_bytes is None or item['actual_bytes'] == expected_bytes))
            if not item['unchanged']:
                raise ValueError('SHA256 or byte length mismatch')
        except Exception as error:
            item['error'] = str(error)
            errors.append(f'{group}: {raw}: {error}')
    if prior_report is not None:
        report['prior_report'] = str(prior_report)
        try:
            prior = json.loads(Path(prior_report).read_text(encoding='utf-8'))
            if prior.get('all_passed') is not True or prior.get('errors') != []:
                errors.append('prior report does not declare an error-free pass')
            # v3 recorded false source_checks without aggregating them into errors.
            for group in ('source_checks', 'data_checks'):
                for row in prior.get(group, []):
                    if row.get('unchanged') is not True:
                        errors.append(f'prior report contains failed {group}: {row.get("path")}')
        except Exception as error:
            errors.append(f'prior report load/schema: {error}')
    report['checked_files'] = len(checks)
    report['matched_files'] = sum(item['unchanged'] for item in checks)
    report['all_passed'] = bool(checks) and not errors
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--relocate-from', help='Explicit old root for archived absolute paths')
    parser.add_argument('--prior-report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit_manifest(args.manifest, args.root, args.prior_report, args.relocate_from)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('all_passed', 'checked_files', 'matched_files', 'errors')}))
    return 0 if report['all_passed'] else 2


if __name__ == '__main__':
    sys.exit(main())
