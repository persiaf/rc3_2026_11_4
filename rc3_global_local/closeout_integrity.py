"""Check previous registered bytes; only current root status metadata may change."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
BASE='6f595cfa89f74d7f7a0e39faaadf5b8655d83f33'
ALLOWED={'README.md','PROJECT_VERSION.json'}


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    tick=time.perf_counter();payload=subprocess.check_output(['git','show',BASE+':project_management/GIT_MANIFEST.json'],cwd=ROOT)
    data=json.loads(payload);changed={};failures=[]
    for name,expected in data['files'].items():
        p=ROOT/name
        actual=dict(sha256=digest(p),bytes=p.stat().st_size) if p.is_file() else None
        if actual!=expected:
            if name in ALLOWED:changed[name]=dict(before=expected,after=actual)
            else:failures.append(name)
    result=dict(schema='rc3-S0-prior-byte-check-v1',base=BASE,files_checked=len(data['files']),
        old_bytes_checked=data['total_bytes'],unchanged=len(data['files'])-len(changed)-len(failures),
        allowed_current_status_metadata_changes=changed,failures=failures,all_frozen_study_bytes_unchanged=not failures,
        base_manifest_sha256=hashlib.sha256(payload).hexdigest(),elapsed_s=time.perf_counter()-tick,
        old_tags_and_main_refs={r:subprocess.check_output(['git','rev-parse',r],cwd=ROOT,text=True).strip() for r in ('main','origin/main')},
        scope='Byte hashes only, no scene interpretation or new planning/H/model/gradient/optimizer/test queries; project manifest self metadata excluded',pushed=False)
    (Path(__file__).resolve().parent/'baseline_integrity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k!='allowed_current_status_metadata_changes'}))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
