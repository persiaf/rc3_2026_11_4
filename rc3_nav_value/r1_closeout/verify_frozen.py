"""Byte-only check of frozen R0/R1 evidence and their kernel/weight sources."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
BASE='df8cf28b0ece2271bd86d5eceee8ea5ca71a65d0'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    payload=subprocess.check_output(['git','show',BASE+':project_management/GIT_MANIFEST.json'],cwd=ROOT)
    manifest=json.loads(payload)
    protocol=json.loads((ROOT/'rc3_nav_value/r1/protocol.json').read_text(encoding='utf-8'))
    prefixes=('rc3_nav_value/r0/','rc3_nav_value/r1/')
    explicit={'rc3_nav_value/REPORT_R0.md','rc3_nav_value/REPORT_R1.md',
        'rc3_nav_value/r0_run.py','rc3_nav_value/r0_analyze.py',
        'rc3_constructive/outputs/checkpoints/update0128.pt',
        'RC3_研究模型与实验计划_20261105.docx','RC3_研究模型与实验计划_20261105_评审修订版.docx'}
    explicit.update(protocol['source_sha256'])
    names=sorted(n for n in manifest['files'] if n.startswith(prefixes) or n in explicit)
    assert names and explicit<=set(manifest['files'])
    failures=[];total=0
    for name in names:
        old=manifest['files'][name];p=ROOT/name;total+=old['bytes']
        if not p.is_file() or p.stat().st_size!=old['bytes'] or digest(p)!=old['sha256']:
            failures.append(name)
    result=dict(schema='rc3-r1-closeout-frozen-bytes-v1',frozen_commit=BASE,
        base_manifest_sha256=hashlib.sha256(payload).hexdigest(),
        scope=dict(prefixes=list(prefixes),explicit_files=sorted(explicit),
            exclusion='New r1_closeout maintenance directory and project status files are outside the frozen evidence scope'),
        files_checked=len(names),bytes_checked=total,all_identical=not failures,failures=failures,
        files={n:manifest['files'][n] for n in names},
        research_calls=0,million_check_verifier_rerun=False,
        meaning='File bytes match the frozen Git content manifest; no performance or feasibility experiment repeated')
    out=Path(__file__).resolve().parent/'frozen_integrity.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:result[k] for k in ('frozen_commit','files_checked','bytes_checked','all_identical','failures','research_calls')}))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
