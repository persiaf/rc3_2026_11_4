"""Package the additive research directory, then verify it beside frozen code."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'outputs'/'rc3_rl_research_g01.zip'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    excluded={'__pycache__','portable_check'}
    files=[p for p in ROOT.rglob('*') if p.is_file()
           and not set(p.relative_to(ROOT).parts)&excluded
           and p.suffix not in ('.pyc','.zip')
           and p.name not in ('PACKAGE_MANIFEST.json','package_receipt.json','portable_receipt.json')]
    manifest=dict(format='additive-rc3-g01-v1',dependency='frozen v5 source modules in the parent extraction directory',
        files=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(files)])
    m=ROOT/'PACKAGE_MANIFEST.json';m.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    with zipfile.ZipFile(ARCHIVE,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(files+[m]):z.write(path,'rc3_rl_research/'+path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(ARCHIVE) as z:
        if z.testzip() is not None:raise ValueError('CRC failure')
        for row in manifest['files']:
            b=z.read('rc3_rl_research/'+row['path'])
            if hashlib.sha256(b).hexdigest()!=row['sha256']:raise ValueError('hash mismatch')
    receipt=dict(archive=str(ARCHIVE),sha256=digest(ARCHIVE),bytes=ARCHIVE.stat().st_size,
        payloads=len(files),entries=len(files)+1,all_manifest_hashes_match=True,
        scope='Additive research package; frozen v5 dependency not duplicated.')
    (ROOT/'outputs'/'package_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False))


def safe_extract(z,member,target):
    output=(target/member).resolve()
    if not output.is_relative_to(target.resolve()):raise ValueError('unsafe ZIP path')
    z.extract(member,target)


def portable():
    target=ROOT/'outputs'/'portable_check'
    if target.exists():raise ValueError('portable check directory already exists; keep previous verification evidence')
    target.mkdir()
    with zipfile.ZipFile(ARCHIVE) as z:
        for name in z.namelist():safe_extract(z,name,target)
    base=ROOT.parent/'rc3_v5_delivery.zip'
    dependency_roots={'rc3_baseline','rc3_fast','rc3_improve','rc3_v3','rc3_v5'}
    copied=[]
    with zipfile.ZipFile(base) as z:
        for name in z.namelist():
            if name.split('/')[0] in dependency_roots and name.endswith('.py'):
                safe_extract(z,name,target);copied.append(name)
    checks=[]
    commands=[['-m','unittest','rc3_rl_research.test_proposal','rc3_rl_research.test_env','rc3_rl_research.test_adapter','-q'],
        ['-m','rc3_rl_research.runtime_rl','--replay','rc3_rl_research/outputs/g0/case_01/selected_replay.json']]
    for command in commands:
        run=subprocess.run([sys.executable,'-B',*command],cwd=target,capture_output=True,
                           text=True,encoding='utf-8',errors='replace')
        checks.append(dict(command=command,returncode=run.returncode,stdout=run.stdout,stderr=run.stderr))
    receipt=dict(all_passed=all(c['returncode']==0 for c in checks),archive_sha256=digest(ARCHIVE),
        dependency_archive_sha256=digest(base),copied_frozen_python_files=len(copied),checks=checks,
        scope='Fresh extraction of the delivered supplement beside only original v5 Python dependencies; no training or timing benchmark.')
    (ROOT/'outputs'/'portable_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(all_passed=receipt['all_passed'],copied_files=len(copied))))
    if not receipt['all_passed']:raise SystemExit(2)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--portable',action='store_true');a=p.parse_args()
    portable() if a.portable else build()
