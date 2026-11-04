"""Relocate a delivery and run its CPU-only entry points from the new root."""
import argparse
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import subprocess
import sys
import zipfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,default=Path('rc3_v5_delivery.zip'))
    p.add_argument('--destination',type=Path,default=Path('rc3_v5_bundle_smoke'))
    args=p.parse_args();dest=args.destination.resolve()
    if dest.exists():raise ValueError('use a fresh destination to avoid modifying existing evidence')
    records=[]
    with zipfile.ZipFile(args.archive) as z:
        assert z.testzip() is None
        manifest=json.loads(z.read('PACKAGE_MANIFEST.json'))
        for row in manifest['files']:
            name=PurePosixPath(row['path'])
            assert not name.is_absolute() and '..' not in name.parts and ':' not in str(name) and '\\' not in str(name)
            data=z.read(str(name));assert hashlib.sha256(data).hexdigest()==row['sha256']
            path=dest.joinpath(*name.parts);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        (dest/'PACKAGE_MANIFEST.json').write_bytes(z.read('PACKAGE_MANIFEST.json'))
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    commands=[
        ('portable_integration',[sys.executable,'-B','-m','rc3_v5.verify','--skip-unit','--output','smoke_outputs']),
        ('external_regressions',[sys.executable,'-B','-m','unittest','rc3_v5.test_external']),
        ('robot_assets',[sys.executable,'-B','-m','rc3_v5.robot_input','inventory','--manifest',
                         'rc3_v5/robot_assets/manifest.json','--output','smoke_outputs/inventory.json']),
        ('native_protocol_relocated',[sys.executable,'-B','-m','rc3_v5.audit_protocol','--input',
             'rc3_v5/mlp_budget_windows','--output','smoke_outputs/native_protocol.json']),
        ('uploaded_protocol_relocated',[sys.executable,'-B','-m','rc3_v5.audit_protocol','--input',
             'rc3_v4_review_import/rc3_v4_review_evidence/mlp_budget_supplement','--output','smoke_outputs/uploaded_protocol.json']),
    ]
    for name,command in commands:
        proc=subprocess.run(command,cwd=dest,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
        records.append(dict(name=name,command=command,returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr))
        assert proc.returncode==0,records[-1]
    result=dict(all_passed=True,relocated_root=str(dest),archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
                files_verified=len(manifest['files']),checks=records)
    args.archive.with_suffix('.smoke.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}))


if __name__=='__main__':main()
