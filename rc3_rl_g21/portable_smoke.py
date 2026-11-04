"""Verify the actual delivery ZIP in a fresh isolated directory, without queries."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import re
import subprocess
import sys
import time
from zipfile import ZipFile
from .build_delivery import ROOT,WORKSPACE,sha,read

def target(root,name):
    parts=PurePosixPath(name)
    if parts.is_absolute() or '..' in parts.parts or any(':' in p for p in parts.parts):
        raise ValueError('Archive path escapes staging root')
    destination=root.joinpath(*parts.parts).resolve()
    if not destination.is_relative_to(root.resolve()):raise ValueError('Archive path escapes staging root')
    return destination

def extract(archive,root,predicate):
    count=0
    for entry in archive.infolist():
        if entry.is_dir() or not predicate(entry.filename):continue
        path=target(root,entry.filename);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(archive.read(entry));count+=1
    return count

def frozen_needed(name):
    if name.endswith('.py'):return True
    if name in ('rc3_rl_g2/outputs/protocol.json','rc3_rl_g2/outputs/protocol.sha256',
                'rc3_rl_g2/outputs/training/bc_initial.pt'):return True
    if name.startswith(('rc3_rl_g2/outputs/teacher_initial/','rc3_rl_g2/outputs/teacher_extra/')):
        return name.endswith(('/teacher.pkl','/summary.json','/audit_receipt.json','/menus.jsonl.gz'))
    return False

def main():
    archive=WORKSPACE/'rc3_rl_g21_delivery.zip';receipt=read(WORKSPACE/'rc3_rl_g21_delivery_receipt.json')
    if sha(archive)!=receipt['archive_sha256']:raise ValueError('Delivery identity mismatch')
    dependency=WORKSPACE/receipt['dependency_archive']
    if sha(dependency)!=receipt['dependency_archive_sha256']:raise ValueError('Frozen dependency identity mismatch')
    stage=ROOT/'outputs/portable_check'/time.strftime('%Y%m%d_%H%M%S')
    stage.mkdir(parents=True,exist_ok=False)
    with ZipFile(dependency) as old:
        old_count=extract(old,stage,frozen_needed)
        old_names=set(old.namelist())
    with ZipFile(archive) as new:
        if old_names&set(new.namelist()):raise ValueError('Additive paths overlap')
        new_count=extract(new,stage,lambda _:True)
    manifest=read(stage/'rc3_rl_g21/MANIFEST.sha256.json')
    for name,identity in manifest['files'].items():
        path=target(stage,name)
        if path.stat().st_size!=identity['bytes'] or sha(path)!=identity['sha256']:
            raise ValueError('Extracted payload mismatch: '+name)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
    commands=[
        (['-m','unittest','discover','-s','rc3_rl_g21','-t','.','-p','test_*.py','-q'],0,'unit_tests'),
        (['-m','rc3_rl_g21.audit_results','--require-all'],0,'results_audit'),
        (['-m','rc3_rl_g21.replay_checkpoints','--output-dir','rc3_rl_g21/outputs/portable_replay'],0,'checkpoint_replay'),
        (['-m','rc3_rl_g21.train_ppo_pilot','--device','cpu'],1,'ppo_gate_rejection')]
    checks=[]
    for args,expected,label in commands:
        before=time.perf_counter()
        result=subprocess.run([sys.executable,'-B',*args],cwd=stage,env=env,capture_output=True,text=True,encoding='utf-8',timeout=120)
        (stage/(label+'.log')).write_text(result.stdout+result.stderr,encoding='utf-8')
        passed=result.returncode==expected
        extra={}
        if label=='unit_tests':
            match=re.search(r'Ran (\d+) tests',result.stderr)
            extra['tests']=int(match.group(1)) if match else None
            passed=passed and extra['tests']==73 and result.stderr.strip().endswith('OK')
        elif label=='results_audit':passed=passed and read(stage/'rc3_rl_g21/outputs/results_audit.json')['all_passed']
        elif label=='checkpoint_replay':passed=passed and read(stage/'rc3_rl_g21/outputs/portable_replay/checkpoint_replay.json')['all_matches']
        elif label=='ppo_gate_rejection':passed=passed and 'PPO is gated: outputs/ppo_authorization.json is absent' in result.stderr
        checks.append(dict(name=label,passed=bool(passed),exit_code=result.returncode,elapsed_s=time.perf_counter()-before,**extra))
        if not passed:raise RuntimeError('Portable check failed; see '+str(stage/(label+'.log')))
    forbidden=['rc3_rl_g21/outputs/ppo_pilot','rc3_rl_g21/outputs/prepared_train',
               'rc3_rl_g21/outputs/prepared_cache','rc3_rl_g2/outputs/prepared_train']
    untouched=all(not (stage/name).exists() for name in forbidden)
    if not untouched:raise RuntimeError('Unexpected interaction/cache output')
    result=dict(schema='rc3-g21-portable-smoke-v1',archive_sha256=receipt['archive_sha256'],
        dependency_sha256=receipt['dependency_archive_sha256'],stage=stage.relative_to(WORKSPACE).as_posix(),
        selected_frozen_files=old_count,new_members=new_count,manifest_payloads_verified=len(manifest['files']),
        checks=checks,all_passed=all(c['passed'] for c in checks) and untouched,
        no_interaction_outputs=untouched,new_environment_queries=0,new_optimizer_updates=0,
        scope='Fresh ZIP extraction; synthetic tests, saved-record audit, zero-update checkpoint replay and rejection of gated PPO. No planner or worker end-to-end run.',
        source_archive_unchanged=sha(archive)==receipt['archive_sha256'])
    (WORKSPACE/'rc3_rl_g21_delivery.smoke.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
