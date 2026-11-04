"""Read-only frozen-v5 and acceptance-package integrity verification."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import zipfile

V5_SHA='0901d94faa949a2378af99b8314580b77669fe666a807fc55a89d7b3a97a4d81'
ACCEPTANCE_SHA='a47d8a06dd1355eb3f5d9d4294ffcb6c18bf72d9943b42d60806fb413bed30be'


def sha(data):return hashlib.sha256(data).hexdigest()


def relative(value):
    p=PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or ':' in value or '\\' in value:
        raise ValueError('unsafe manifest path: '+value)
    return p


def audit(root,base,acceptance):
    rows=[];live=[];imported=[];archive_only=[]
    assert sha(base.read_bytes())==V5_SHA,'frozen v5 archive changed'
    with zipfile.ZipFile(base) as z:
        assert z.testzip() is None
        names=z.namelist();assert len(names)==len(set(names))==542
        manifest=json.loads(z.read('PACKAGE_MANIFEST.json'))
        for item in manifest['files']:
            name=str(relative(item['path']));data=z.read(name)
            matched=sha(data)==item['sha256'] and len(data)==item['bytes']
            rows.append(dict(path=name,matched=matched));assert matched,name
            path=root.joinpath(*PurePosixPath(name).parts)
            if name in ('README_V5.md','V4_PACKAGE_MANIFEST.json'):
                archive_only.append(name)
            else:
                same=path.is_file() and sha(path.read_bytes())==item['sha256']
                live.append(dict(path=name,unchanged=same));assert same,'workspace differs: '+name
        assert len(rows)==541
        old=json.loads(z.read('V4_PACKAGE_MANIFEST.json'))
        old_rows=[]
        for item in old['files']:
            name=str(relative(item['path']))
            same=sha(z.read(name))==item['sha256']
            old_rows.append(dict(path=name,unchanged=same));assert same,name
        # The inherited manifest itself has a v5 payload checksum above.
        assert len(old_rows)==270
    evidence=acceptance/'rc3_v5_acceptance_evidence'
    m=json.loads((evidence/'MANIFEST.json').read_text(encoding='utf-8'))
    for item in m['files']:
        name=relative(item['path']);path=evidence.joinpath(*name.parts)
        same=path.is_file() and sha(path.read_bytes())==item['sha256'] and path.stat().st_size==item['bytes']
        imported.append(dict(path=str(name),matched=same));assert same,'acceptance differs: '+str(name)
    original_acceptance=root/'rc3_v5_acceptance_evidence.zip'
    archive_check=None
    if original_acceptance.exists():
        archive_check=sha(original_acceptance.read_bytes())==ACCEPTANCE_SHA
        assert archive_check,'acceptance archive changed'
    return dict(all_passed=True,scope='Integrity only; no training, timing or plan execution.',
        v5_archive_sha256=V5_SHA,v5_entries=542,v5_manifest_payloads=len(rows),
        live_payloads_unchanged=len(live),archive_only_payloads=archive_only,
        inherited_v4_payloads=len(old_rows),inherited_v4_entries_including_manifest=271,
        acceptance_payloads=len(imported),acceptance_archive_sha256=ACCEPTANCE_SHA,
        acceptance_archive_present_and_matches=archive_check,
        payload_checks=rows,workspace_checks=live,acceptance_checks=imported)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--base-archive',type=Path,default=Path('rc3_v5_delivery.zip'))
    p.add_argument('--acceptance-import',type=Path,default=Path('rc3_v5_acceptance_import'))
    p.add_argument('--output',type=Path,default=Path('rc3_v5_1/evidence/freeze_audit.json'))
    a=p.parse_args();r=audit(a.root,a.base_archive,a.acceptance_import)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in r.items() if not k.endswith('_checks')}))


if __name__=='__main__':main()
