"""Incremental artifact packaging; portable checks never execute a repair."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path,PurePosixPath
from zipfile import ZipFile,ZIP_DEFLATED
from .core import ROOT,WS,read,write,sha

def payloads():
    return [p for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='MANIFEST.sha256.json' and '__pycache__' not in p.parts
        and p.suffix not in ('.pyc','.pending') and p.relative_to(ROOT).parts[:2]!=('outputs','portable_check')]

def build():
    p=read(ROOT/'outputs/protocol.json');a=read(ROOT/'outputs/analysis.json')
    assert read(ROOT/'outputs/label_verification.json')['all_passed']
    assert read(ROOT/'outputs/freeze_after.json')['all_passed']
    for name,expected in p['source_sha256'].items():assert sha(ROOT/name)==expected
    counts=read(ROOT/'outputs/resources.json')['counts'];assert counts==a['resource_counts']
    files=payloads();manifest=dict(schema='g23-delivery-manifest-v1',date='2026-11-04',
        files={f.relative_to(WS).as_posix():dict(sha256=sha(f),bytes=f.stat().st_size) for f in files},
        scope='Additive; dependencies from frozen before snapshot. Manifest covered by external ZIP receipt.')
    write(ROOT/'MANIFEST.sha256.json',manifest);files.append(ROOT/'MANIFEST.sha256.json')
    destination=WS/'rc3_rl_g23_delivery.zip';pending=destination.with_suffix('.pending')
    with ZipFile(pending,'w',ZIP_DEFLATED,compresslevel=6) as z:
        for f in files:z.write(f,f.relative_to(WS).as_posix())
    pending.replace(destination)
    with ZipFile(destination) as z:
        assert z.testzip() is None
        for name,x in manifest['files'].items():assert hashlib.sha256(z.read(name)).hexdigest()==x['sha256']
    receipt=dict(archive=destination.name,sha256=sha(destination),bytes=destination.stat().st_size,members=len(files),
        manifest_payloads=len(manifest['files']),all_payloads_match=True,primary_passed=a['primary_passed'],
        counts=counts,scope='Three seeds; capped negative result; no default deployment replacement',smoke='rc3_rl_g23_delivery.smoke.json')
    write(WS/'rc3_rl_g23_delivery_receipt.json',receipt);print(receipt)

def smoke():
    receipt=read(WS/'rc3_rl_g23_delivery_receipt.json');archive=WS/receipt['archive'];assert sha(archive)==receipt['sha256']
    stage=ROOT/'outputs/portable_check'/str(time.time_ns());stage.mkdir(parents=True,exist_ok=False)
    frozen=read(ROOT/'outputs/freeze_before.json');deps=set()
    for r in frozen['g2_and_legacy']['members']:
        if r['local_expected']:deps.add(r['path'])
    deps.update(r['path'] for r in frozen['g2_and_legacy']['archives'])
    deps.update(frozen['g21_payloads']);deps.update(frozen['g22_manifest']['files'])
    deps.update(('rc3_rl_g21/MANIFEST.sha256.json','rc3_rl_g21_delivery.zip','rc3_rl_g22_delivery.zip'))
    for name in sorted(deps):
        src=WS/name;dst=(stage/name).resolve()
        if not dst.is_relative_to(stage.resolve()):raise ValueError('Dependency path escapes stage')
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    with ZipFile(archive) as z:
        for info in z.infolist():
            name=PurePosixPath(info.filename)
            if name.is_absolute() or '..' in name.parts:raise ValueError('Unsafe archive member')
            dst=stage.joinpath(*name.parts).resolve()
            if not dst.is_relative_to(stage.resolve()):raise ValueError('ZIP path escapes stage')
            dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(z.read(info))
    manifest=read(stage/'rc3_rl_g23/MANIFEST.sha256.json')
    for name,identity in manifest['files'].items():assert sha(stage/name)==identity['sha256']
    checks=[];env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
    commands=[('unit',['-m','unittest','rc3_rl_g23.test_contract','-q'],0,'Ran 4 tests'),
        ('independent_saved_review',['rc3_rl_g23/review_saved_evidence.py'],0,'"primary_passed": false'),
        ('saved_plan_analysis',['-m','rc3_rl_g23.analyze'],0,"'primary_passed': False"),
        ('refuse_trial_replay',['-m','rc3_rl_g23.trial'],1,'No implicit resume/retry or budget expansion'),
        ('refuse_branch_requery',['-m','rc3_rl_g23.verify_labels'],1,'Verification already saved')]
    for name,args,code,needle in commands:
        result=subprocess.run([sys.executable,'-B',*args],cwd=stage,env=env,capture_output=True,text=True,encoding='utf-8',timeout=180)
        log=result.stdout+result.stderr;(stage/(name+'.log')).write_text(log,encoding='utf-8')
        passed=result.returncode==code and needle in log
        checks.append(dict(name=name,passed=passed,exit_code=result.returncode))
        if not passed:raise RuntimeError('Portable check failed: '+str(stage/(name+'.log')))
    assert read(stage/'rc3_rl_g23/outputs/resources.json')['counts']==receipt['counts']
    assert sha(archive)==receipt['sha256']
    result=dict(all_passed=True,archive_sha256=receipt['sha256'],stage=stage.relative_to(WS).as_posix(),
        payloads_verified=len(manifest['files']),copied_dependency_files=len(deps),checks=checks,
        additional_repairs=0,additional_training_updates=0,scope='Fresh isolated artifact/dependency copies, saved-record checks only; query guards reject reexecution')
    write(WS/'rc3_rl_g23_delivery.smoke.json',result);print(result)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('build','smoke'));args=parser.parse_args()
    build() if args.mode=='build' else smoke()
