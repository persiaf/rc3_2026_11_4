"""Verify all pre-existing registered bytes, without research execution."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
BASE='5d5fe25754ffa032f97937a3fb54f15901c2fb7d'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    started=time.perf_counter()
    payload=subprocess.check_output(['git','show',BASE+':project_management/GIT_MANIFEST.json'],cwd=ROOT)
    data=json.loads(payload);failures=[]
    for name,expected in data['files'].items():
        p=ROOT/name
        if not p.is_file() or p.stat().st_size!=expected['bytes'] or digest(p)!=expected['sha256']:
            failures.append(name)
    refs={ref:subprocess.check_output(['git','rev-parse',ref],cwd=ROOT,text=True).strip() for ref in ('main','origin/main')}
    all_passed=not failures and all(s==BASE for s in refs.values())
    result=dict(schema='rc3-T-closeout-old-bytes-v1',maintenance_base=BASE,
        frozen_R1='df8cf28b0ece2271bd86d5eceee8ea5ca71a65d0',
        base_manifest_sha256=hashlib.sha256(payload).hexdigest(),files_checked=len(data['files']),
        bytes_checked=data['total_bytes'],all_passed=all_passed,failures=failures,
        original_main_refs=refs,elapsed_s=time.perf_counter()-started,
        scope='Only byte hashes against maintenance-base manifest; no scene interpretation, generation, navigation, planning, H, inference, training or test queries. New T directory and self-referential project manifest metadata are outside old-file scope.',
        new_research_calls=0,full_historical_QA_rerun=False,pushed=False)
    (OUT/'baseline_integrity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,ensure_ascii=False))
    if not all_passed:raise SystemExit(1)


if __name__=='__main__':main()
