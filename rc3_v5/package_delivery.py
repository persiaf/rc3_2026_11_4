"""Build a self-contained v5 delivery while preserving all archived v4 bytes."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path,PurePosixPath
import zipfile


def sha(data):return hashlib.sha256(data).hexdigest()


def safe_name(name):
    path=PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or ':' in name or '\\' in name:
        raise ValueError('unsafe archive path: '+name)
    return name


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=Path('rc3_v4_delivery.zip'))
    p.add_argument('--output',type=Path,default=Path('rc3_v5_delivery.zip'));a=p.parse_args()
    root=Path.cwd();entries={};frozen=[]
    with zipfile.ZipFile(a.base) as old:
        assert old.testzip() is None
        manifest=json.loads(old.read('PACKAGE_MANIFEST.json'))
        for item in manifest['files']:
            name=safe_name(item['path']);data=old.read(name)
            assert sha(data)==item['sha256'] and len(data)==item['bytes'],name
            local=root/name
            matched=local.is_file() and sha(local.read_bytes())==item['sha256']
            frozen.append(dict(path=name,unchanged=matched))
            assert matched,'archived v4 file changed: '+name
            entries[name]=data
        entries['V4_PACKAGE_MANIFEST.json']=old.read('PACKAGE_MANIFEST.json')
    receipt=dict(all_passed=all(x['unchanged'] for x in frozen),files=len(frozen),
                 original_archive_sha256=sha(a.base.read_bytes()),checks=frozen)
    (root/'rc3_v5/frozen_v4_integrity.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    files=list((root/'rc3_v5').rglob('*'))+list((root/'rc3_v4_review_import/rc3_v4_review_evidence').rglob('*'))
    files+=list((root/'paper/review-traces/experiment-result-to-claim/2026-11-03_run03').rglob('*'))
    files += [root/p for p in ('paper/EXPERIMENT_PLAN_20261103_v9.md','paper/EXPERIMENT_TRACKER_20261103_v9.md',
        'rc3_improve/OPEN_SOURCE_NOTES.md')]
    for path in files:
        if not path.is_file() or '__pycache__' in path.parts or path.suffix in ('.pyc','.tmp'):continue
        name=safe_name(path.relative_to(root).as_posix());data=path.read_bytes()
        if name in entries and entries[name]!=data:raise ValueError('unapproved overwrite: '+name)
        entries[name]=data
    entries['README_V5.md']=b'Start here: rc3_v5/README.md\nResults: rc3_v5/RESULTS.md\nRobot input status: rc3_v5/ROBOT_INPUT_STATUS.md\n'
    contents=[dict(path=k,bytes=len(v),sha256=sha(v)) for k,v in sorted(entries.items())]
    pack=dict(schema='rc3-v5-delivery-v1',created_utc=datetime.now(timezone.utc).isoformat(),
              files=contents,base_archive_sha256=receipt['original_archive_sha256'],
              scope='Frozen v4 dependency/evidence snapshot plus isolated v5 maintenance and robot input preparation.')
    entries['PACKAGE_MANIFEST.json']=json.dumps(pack,indent=2,ensure_ascii=True).encode('utf-8')
    with zipfile.ZipFile(a.output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,data in sorted(entries.items()):z.writestr(name,data)
    with zipfile.ZipFile(a.output) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(entries)==len(set(z.namelist()))
        assert all(sha(z.read(row['path']))==row['sha256'] for row in contents)
    final=dict(path=str(a.output),files=len(entries),manifest_files=len(contents),bytes=a.output.stat().st_size,
               sha256=sha(a.output.read_bytes()),crc_passed=True,manifest_hashes_passed=True,
               frozen_v4_files_unchanged=len(frozen))
    a.output.with_suffix('.manifest.json').write_text(json.dumps(final,indent=2),encoding='utf-8')
    print(json.dumps(final))


if __name__=='__main__':main()
