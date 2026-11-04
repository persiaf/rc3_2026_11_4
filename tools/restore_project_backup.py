"""Restore full_before_cleanup.zip into a NEW empty directory (stdlib only).

Example: python RESTORE_PROJECT.py full_before_cleanup.zip D:\\rc3_restored
The ZIP contains a SHA256 object store plus a manifest with every original path.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import zipfile


def restore(archive, destination):
    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Restore destination must be new or empty; existing project will not be overwritten')
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('SNAPSHOT_MANIFEST.json'))
        targets = []
        seen = set()
        for entry in manifest['files']:
            name = PurePosixPath(entry['path'])
            if name.is_absolute() or '..' in name.parts or '\\' in entry['path'] or ':' in entry['path']:
                raise ValueError('Unsafe manifest path')
            target = (destination / Path(*name.parts)).resolve()
            if not target.is_relative_to(destination) or target == destination or str(target).casefold() in seen:
                raise ValueError('Unsafe or duplicate restore destination')
            seen.add(str(target).casefold())
            key = entry['sha256']
            if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
                raise ValueError('Invalid content hash')
            targets.append((entry, target))
        # Verify each object before creating any restored project file.
        for key, size in {e['sha256']: e['bytes'] for e, _ in targets}.items():
            digest = hashlib.sha256(); actual = 0
            with z.open('objects/' + key) as stream:
                for data in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    digest.update(data); actual += len(data)
            if digest.hexdigest() != key or actual != size:
                raise ValueError('Corrupt backup object: ' + key)
        for index, (entry, target) in enumerate(targets, 1):
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise ValueError('Refusing to overwrite restore file')
            with z.open('objects/' + entry['sha256']) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
            os.utime(target, ns=(entry['mtime_ns'], entry['mtime_ns']))
            if index % 1000 == 0:
                print(f'Restored {index}/{len(targets)} files', flush=True)
    print(json.dumps({'restored_files': len(targets), 'destination': str(destination)}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    restore(args.archive, args.destination)
