"""Package the additive G2.1 artifact; never rewrites any frozen dependency."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parent
MANIFEST=ROOT/'MANIFEST.sha256.json'

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def payloads():
    result=[]
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or path==MANIFEST:continue
        relative=path.relative_to(ROOT)
        if '__pycache__' in relative.parts or path.suffix in ('.pyc','.pending'):continue
        if relative.parts[:2] in (('outputs','portable_check'),('outputs','reproductions')):continue
        result.append(path)
    return result

def main():
    decision=read(ROOT/'outputs/decision.json')
    checks=read(ROOT/'outputs/final_checks.json')
    if decision['opportunity_passed'] or decision['new_ppo_committed_actions']!=0:
        raise ValueError('This delivery is the bounded offline outcome')
    if not checks['all_passed']:raise ValueError('Final checks incomplete')
    if not read(ROOT/'outputs/freeze_after.json')['all_passed']:raise ValueError('Frozen inputs changed')
    if not read(ROOT/'outputs/checkpoint_replay.json')['all_matches']:raise ValueError('Checkpoint replay failed')
    files=payloads()
    manifest=dict(schema='rc3-g21-additive-payload-manifest-v1',client_delivery_date='2026-11-04',
        scope='G2.1 only. MANIFEST itself is covered by the external ZIP receipt.',
        dependency_archive='rc3_rl_g2_delivery.zip',
        dependency_archive_sha256='2fe860781980ccf7bffbe2fdc0d7945efc5b8941a64c7fe0d0ac3b5bfddc2908',
        files={p.relative_to(WORKSPACE).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files})
    MANIFEST.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    files.append(MANIFEST)
    destination=WORKSPACE/'rc3_rl_g21_delivery.zip'
    pending=destination.with_suffix('.pending')
    with ZipFile(pending,'w',compression=ZIP_DEFLATED,compresslevel=6) as archive:
        for path in files:archive.write(path,path.relative_to(WORKSPACE).as_posix())
    pending.replace(destination)
    with ZipFile(destination) as archive:
        if archive.testzip() is not None:raise ValueError('Archive CRC failed')
        names=set(archive.namelist())
        if any(not name.startswith('rc3_rl_g21/') for name in names):raise ValueError('Old paths leaked into additive package')
        for name,identity in manifest['files'].items():
            raw=archive.read(name)
            if len(raw)!=identity['bytes'] or hashlib.sha256(raw).hexdigest()!=identity['sha256']:
                raise ValueError('Archived payload mismatch: '+name)
    with ZipFile(WORKSPACE/'rc3_rl_g2_delivery.zip') as previous:
        overlaps=sorted(names&set(previous.namelist()))
    if overlaps:raise ValueError('Package overwrites frozen paths')
    receipt=dict(schema='rc3-g21-delivery-receipt-v1',client_delivery_date='2026-11-04',
        archive=destination.name,archive_sha256=sha(destination),archive_bytes=destination.stat().st_size,
        members=len(files),manifest_payloads=len(manifest['files']),manifest_sha256=sha(MANIFEST),
        payloads_match=True,crc_passed=True,frozen_path_overlaps=overlaps,
        dependency_archive=manifest['dependency_archive'],dependency_archive_sha256=manifest['dependency_archive_sha256'],
        artifact_scope='Additive offline training evidence; all new models fail the operational opportunity gate',
        final_checks_sha256=sha(ROOT/'outputs/final_checks.json'),smoke_receipt='rc3_rl_g21_delivery.smoke.json')
    (WORKSPACE/'rc3_rl_g21_delivery_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))

if __name__=='__main__':main()
