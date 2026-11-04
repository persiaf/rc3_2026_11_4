"""Pack reproducible v4 code/weights/evidence, retaining every frozen result."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED


def main():
    root=Path('.').resolve();archive=root/'rc3_v4_delivery.zip'
    if archive.exists():raise ValueError('refuse to overwrite an earlier delivery')
    required=['rc3_v4/results/verification.json','rc3_v4/verification/summary.json','rc3_v4/training/parity.json']
    for name in required:
        if not json.loads((root/name).read_text(encoding='utf-8'))['all_passed']:
            raise ValueError('required verification did not pass: '+name)
    files=[p for p in (root/'rc3_v4').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
    for folder in ('rc3_baseline','rc3_fast','rc3_improve','rc3_v3'):
        files.extend((root/folder).glob('*.py'))
    files += [root/'rc3_v3/deployment/policy.npz',root/'rc3_v3/data_v3_20261102/data.pkl',
              root/'rc3_v3/data_v3_20261102/manifest.json',root/'rc3_v3/data_v3_20261102/validation_cases.json',root/'rc3_v3_audit_evidence.zip',
              root/'paper/EXPERIMENT_PLAN_20261103_v8.md',root/'paper/EXPERIMENT_TRACKER_20261103_v8.md']
    trace=root/'paper/review-traces/experiment-result-to-claim/2026-11-03_run02'
    files += [p for p in trace.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files=sorted(set(files));entries=[]
    with ZipFile(archive,'w',compression=ZIP_DEFLATED,compresslevel=6) as z:
        for path in files:
            relative=path.relative_to(root).as_posix();blob=path.read_bytes()
            entries.append(dict(path=relative,bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest()));z.writestr(relative,blob)
        z.writestr('PACKAGE_MANIFEST.json',json.dumps(dict(schema='rc3-v4-delivery-v1',files=entries),indent=2))
    with ZipFile(archive) as z:
        bad=z.testzip()
        if bad:raise ValueError('zip CRC failed: '+bad)
    result=dict(path=archive.name,files=len(entries)+1,bytes=archive.stat().st_size,
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),crc_passed=True)
    (root/'rc3_v4_delivery.manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
