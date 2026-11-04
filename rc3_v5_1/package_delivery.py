"""Package the isolated maintenance overlay and smoke-test beside frozen v5."""
import argparse
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import subprocess
import sys
import zipfile
from .audit_freeze import V5_SHA


def sha(data):return hashlib.sha256(data).hexdigest()


def safe(value):
    p=PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or ':' in value or '\\' in value:
        raise ValueError('unsafe path: '+value)
    return p


def extract(z,dest):
    for item in z.infolist():
        p=dest.joinpath(*safe(item.filename).parts)
        if item.is_dir():p.mkdir(parents=True,exist_ok=True);continue
        if p.exists():raise ValueError('overlay may not overwrite existing v5: '+str(p))
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(item))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('rc3_v5_1_maintenance.zip'))
    p.add_argument('--base',type=Path,default=Path('rc3_v5_delivery.zip'))
    p.add_argument('--smoke-directory',type=Path,default=Path('rc3_v5_1_bundle_smoke'))
    a=p.parse_args();base=a.base.resolve();dest=a.smoke_directory.resolve();root=Path.cwd()
    assert sha(base.read_bytes())==V5_SHA
    if dest.exists():raise ValueError('smoke directory must be new')
    paths=[]
    for folder in ('rc3_v5_1','rc3_v5_acceptance_import/rc3_v5_acceptance_evidence'):
        paths.extend(v for v in (root/folder).rglob('*') if v.is_file() and '__pycache__' not in v.parts and v.suffix!='.pyc')
    entries={v.relative_to(root).as_posix():v.read_bytes() for v in paths}
    manifest={'schema':'rc3-v5.1-maintenance-v1','required_base_sha256':V5_SHA,
              'overwrites_frozen_files':False,'files':[{'path':k,'bytes':len(v),'sha256':sha(v)} for k,v in sorted(entries.items())]}
    manifest_name='V5_1_MAINTENANCE_MANIFEST.json'
    entries[manifest_name]=json.dumps(manifest,indent=2).encode('utf-8')
    with zipfile.ZipFile(base) as z:assert not(set(entries)&set(z.namelist()))
    with zipfile.ZipFile(a.output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,data in sorted(entries.items()):z.writestr(str(safe(name)),data)
    with zipfile.ZipFile(a.output) as z:
        assert z.testzip() is None
        for item in manifest['files']:
            assert sha(z.read(item['path']))==item['sha256']
    receipt={'files':len(entries),'manifest_payloads':len(manifest['files']),'bytes':a.output.stat().st_size,
             'sha256':sha(a.output.read_bytes()),'base_sha256':V5_SHA,'crc_passed':True,'payload_hashes_passed':True,
             'overwrites_frozen_files':False}
    a.output.with_suffix('.manifest.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    with zipfile.ZipFile(base) as z:extract(z,dest)
    with zipfile.ZipFile(a.output) as z:extract(z,dest)
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    commands=[
        ('nineteen_regressions',[sys.executable,'-B','-m','unittest','rc3_v5_1.test_robot_input','-v'],0),
        ('patched_help',[sys.executable,'-B','-m','rc3_v5_1.robot_input','--help'],0),
        ('unchanged_empty_template',[sys.executable,'-B','-m','rc3_v5_1.robot_input','check',
          '--manifest','rc3_v5/robot_assets/manifest.json','--config','rc3_v5/robot_assets/runtime_config.template.json',
          '--output','maintenance_smoke/preflight.json'],2),
        ('relocated_freeze_audit',[sys.executable,'-B','-m','rc3_v5_1.audit_freeze','--base-archive',str(base),
          '--output','maintenance_smoke/freeze.json'],0),
    ]
    records=[]
    for name,cmd,expected in commands:
        proc=subprocess.run(cmd,cwd=dest,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
        records.append({'name':name,'expected_exit':expected,'exit':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr})
        assert proc.returncode==expected,records[-1]
    preflight=json.loads((dest/'maintenance_smoke/preflight.json').read_text(encoding='utf-8'))
    assert preflight['status']=='blocked_inputs' and len(preflight['blocked_inputs'])==135
    assert not preflight['physical_ik_verified'] and not preflight['collision_verified']
    assert 'rc3_v5_1.robot_input' in records[1]['stdout']
    # All shipped overlay bytes must still match after the relocated tests.
    assert all(sha((dest/item['path']).read_bytes())==item['sha256'] for item in manifest['files'])
    smoke={'all_passed':True,'relocated_root':str(dest),'archive_sha256':receipt['sha256'],
           'checks':records,'overlay_unchanged_after_tests':True,'real_calibration_or_ik_performed':False}
    a.output.with_suffix('.smoke.json').write_text(json.dumps(smoke,indent=2),encoding='utf-8')
    print(json.dumps(dict(receipt,relocated_smoke_passed=True)))


if __name__=='__main__':main()
