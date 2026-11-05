"""Compare every historical file to the Git base; allow only four status files."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
BASE='129ba6e916ed5a44ecb92998553c082b4510ecd1'
ALLOWED={'PROJECT_VERSION.json','README.md','rc3_nav_value/README.md',
         'research_proposals/20261105_nav_value_review/paper/EXPERIMENT_TRACKER.md'}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    payload=subprocess.check_output(['git','show',BASE+':project_management/GIT_MANIFEST.json'],cwd=ROOT)
    original=json.loads(payload);changes=[];failures=[]
    for name,old in original['files'].items():
        p=ROOT/name;now=digest(p) if p.is_file() else None
        if now!=old['sha256']:
            entry=dict(path=name,before_sha256=old['sha256'],after_sha256=now)
            changes.append(entry)
            if name not in ALLOWED:failures.append(entry)
    result=dict(base_commit=BASE,base_manifest_sha256=hashlib.sha256(payload).hexdigest(),
        original_files=len(original['files']),allowed_status_paths=sorted(ALLOWED),actual_changes=changes,
        frozen_files_checked=len(original['files'])-len(ALLOWED),frozen_files_all_identical=not failures,
        failures=failures,scope='Original file bytes only; new R1 files and self-referential manifest excluded. No research computation.')
    (Path(__file__).resolve().parent/'frozen_integrity_closeout.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(frozen_files_checked=result['frozen_files_checked'],all_identical=not failures,status_files_changed=len(changes))))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
