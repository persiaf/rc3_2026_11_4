"""Package and verify the actual additive ZIP; smoke never executes a plan."""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path,PurePosixPath
import re
import subprocess
import sys
import time
from zipfile import ZipFile,ZIP_DEFLATED
from .common import ROOT,WORKSPACE,read,write,sha

DEPENDENCIES={
    'rc3_rl_g2_delivery.zip':'2fe860781980ccf7bffbe2fdc0d7945efc5b8941a64c7fe0d0ac3b5bfddc2908',
    'rc3_rl_g21_delivery.zip':'79b0985a1ec459f2a069e6b1c4d4841a6a54d7c609df9f020c25e7414086c50b'}

def files():
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or path.name=='MANIFEST.sha256.json':continue
        rel=path.relative_to(ROOT)
        if '__pycache__' in rel.parts or path.suffix in ('.pyc','.pending'):continue
        if rel.parts[:2]==('outputs','portable_check'):continue
        yield path

def build():
    if not read(ROOT/'outputs/final_checks.json')['all_passed']:raise ValueError('Checks incomplete')
    payload=list(files())
    manifest=dict(schema='rc3-g22-additive-manifest-v1',client_date='2026-11-04',dependencies=DEPENDENCIES,
        scope='G22 only. The external receipt covers this manifest and the ZIP.',
        files={p.relative_to(WORKSPACE).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in payload})
    manifest_path=ROOT/'MANIFEST.sha256.json';write(manifest_path,manifest);payload.append(manifest_path)
    destination=WORKSPACE/'rc3_rl_g22_delivery.zip';pending=destination.with_suffix('.pending')
    with ZipFile(pending,'w',ZIP_DEFLATED,compresslevel=6) as archive:
        for path in payload:archive.write(path,path.relative_to(WORKSPACE).as_posix())
    pending.replace(destination)
    with ZipFile(destination) as archive:
        if archive.testzip() is not None:raise ValueError('ZIP CRC failed')
        if any(not n.startswith('rc3_rl_g22/') for n in archive.namelist()):raise ValueError('Old paths in incremental ZIP')
        for name,identity in manifest['files'].items():
            raw=archive.read(name)
            if len(raw)!=identity['bytes'] or hashlib.sha256(raw).hexdigest()!=identity['sha256']:
                raise ValueError('ZIP payload changed')
    for name,expected in DEPENDENCIES.items():
        if sha(WORKSPACE/name)!=expected:raise ValueError('Frozen dependency changed')
    result=dict(schema='rc3-g22-delivery-receipt-v1',client_date='2026-11-04',archive=destination.name,
        archive_bytes=destination.stat().st_size,archive_sha256=sha(destination),members=len(payload),
        manifest_payloads=len(manifest['files']),manifest_sha256=sha(manifest_path),dependencies=DEPENDENCIES,
        all_payloads_match=True,crc_passed=True,scope='72 saved frozen-policy plans and feedback; zero training/PPO',
        final_checks_sha256=sha(ROOT/'outputs/final_checks.json'),smoke_receipt='rc3_rl_g22_delivery.smoke.json')
    write(WORKSPACE/'rc3_rl_g22_delivery_receipt.json',result);print(result)

def extract(archive,directory,predicate):
    count=0
    for item in archive.infolist():
        if item.is_dir() or not predicate(item.filename):continue
        name=PurePosixPath(item.filename)
        if name.is_absolute() or '..' in name.parts or any(':' in p for p in name.parts):raise ValueError('Unsafe ZIP path')
        path=directory.joinpath(*name.parts).resolve()
        if not path.is_relative_to(directory.resolve()):raise ValueError('ZIP path escapes stage')
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(archive.read(item));count+=1
    return count

def smoke():
    receipt=read(WORKSPACE/'rc3_rl_g22_delivery_receipt.json');archive=WORKSPACE/receipt['archive']
    if sha(archive)!=receipt['archive_sha256']:raise ValueError('ZIP identity changed')
    stage=ROOT/'outputs/portable_check'/str(time.time_ns());stage.mkdir(parents=True,exist_ok=False)
    selected={}
    for name,expected in DEPENDENCIES.items():
        path=WORKSPACE/name
        if sha(path)!=expected:raise ValueError('Dependency ZIP changed')
        predicate=(lambda n:True) if 'g21' in name else (lambda n:n.endswith('.py') or n in (
            'rc3_rl_g2/outputs/teacher_extra/teacher.pkl','rc3_rl_g2/outputs/teacher_extra/summary.json',
            'rc3_rl_g2/outputs/teacher_extra/audit_receipt.json','rc3_rl_g2/outputs/protocol.json','rc3_rl_g2/outputs/protocol.sha256'))
        with ZipFile(path) as z:selected[name]=extract(z,stage,predicate)
    with ZipFile(archive) as z:new_count=extract(z,stage,lambda _:True)
    manifest=read(stage/'rc3_rl_g22/MANIFEST.sha256.json')
    for name,identity in manifest['files'].items():
        if sha(stage/name)!=identity['sha256'] or (stage/name).stat().st_size!=identity['bytes']:
            raise ValueError('Extracted new payload identity mismatch')
    commands=[
        ('new_tests',['-m','unittest','rc3_rl_g22.test_feedback','-q'],0,None),
        ('saved_plan_audit',['-m','rc3_rl_g22.audit'],0,None),
        ('model_input_replay',['-m','rc3_rl_g22.replay'],0,None),
        ('evaluation_only_authority',['-c',"from rc3_rl_g22.evaluate import authorize; p=authorize(); assert p['training_allowed'] is False and p['ppo_allowed'] is False; print('evaluation-only')"],0,'evaluation-only'),
        ('refuse_duplicate_plans',['-m','rc3_rl_g22.evaluate'],1,'No automatic replay or plan retries'),
        ('old_g21_gate_still_closed',['-c','from rc3_rl_g21.evaluate_g21 import require_opportunity_gate; require_opportunity_gate()'],1,'opportunity_gate.json')]
    outcomes=[];env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
    for name,args,code,needle in commands:
        result=subprocess.run([sys.executable,'-B',*args],cwd=stage,env=env,capture_output=True,text=True,encoding='utf-8',timeout=120)
        log=result.stdout+result.stderr;(stage/(name+'.log')).write_text(log,encoding='utf-8')
        passed=result.returncode==code and (needle is None or needle in log)
        detail={}
        if name=='new_tests':
            match=re.search(r'Ran (\d+) tests',log);detail['tests']=int(match.group(1)) if match else None
            passed=passed and detail['tests']==16 and log.rstrip().endswith('OK')
        if name=='saved_plan_audit':passed=passed and read(stage/'rc3_rl_g22/outputs/plan_audit.json')['all_passed']
        if name=='model_input_replay':passed=passed and read(stage/'rc3_rl_g22/outputs/inference_replay.json')['all_passed']
        outcomes.append(dict(name=name,passed=bool(passed),exit_code=result.returncode,**detail))
        if not passed:raise ValueError('Portable check failed: '+str(stage/(name+'.log')))
    if len((stage/'rc3_rl_g22/outputs/evaluation/attempts.jsonl').read_text().splitlines())!=72:
        raise ValueError('Duplicate planning slot was executed')
    if (stage/'rc3_rl_g21/outputs/ppo_authorization.json').exists() or (stage/'rc3_rl_g21/outputs/opportunity_gate.json').exists():
        raise ValueError('Old gate was changed')
    result=dict(schema='rc3-g22-portable-smoke-v1',archive_sha256=receipt['archive_sha256'],all_passed=True,
        dependencies=DEPENDENCIES,selected_dependency_files=selected,new_members=new_count,
        verified_new_payloads=len(manifest['files']),checks=outcomes,stage=stage.relative_to(WORKSPACE).as_posix(),
        additional_plans=0,additional_search_proposals=0,optimizer_updates=0,ppo_actions=0,
        scope='Fresh ZIP extraction, synthetic tests, saved plan and model-input verification, duplicate-plan rejection, and unchanged old gate; no scene planning re-executed',
        archive_unchanged=sha(archive)==receipt['archive_sha256'])
    write(WORKSPACE/'rc3_rl_g22_delivery.smoke.json',result);print(result)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('build','smoke'))
    args=parser.parse_args();build() if args.mode=='build' else smoke()
