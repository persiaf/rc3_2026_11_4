"""Byte-only preservation audit of all baseline evidence (no task semantics)."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rc3_s1b_results_20261106.audit_s1b import digest,write_new,HERE

def main():
    started=time.perf_counter()
    base='ffcfcabc4ccc0b071a368b5e6e49684be3de4d64'
    manifest=json.loads(subprocess.check_output(['git','show',base+':project_management/GIT_MANIFEST.json'],cwd=ROOT))
    mutable={'README.md','PROJECT_VERSION.json','rc3_global_local_execution_v3/README.md','rc3_global_local_execution_v3/ACTIVE_EXECUTION_STATUS.json'}
    user_file='第二章与第三章修订稿_20261029.docx'
    checked=0;total=0;failures=[]
    for name,r in manifest['files'].items():
        if name in mutable or name==user_file:continue
        checked+=1;total+=r['bytes'];p=ROOT/name
        if not p.is_file() or p.stat().st_size!=r['bytes'] or digest(p)!=r['sha256']:
            failures.append(name)
    # Worktree user's unrelated in-progress Word edit stays untouched; the
    # committed Git blob is the release content, not the live Office buffer.
    blob=subprocess.check_output(['git','show',base+':'+user_file],cwd=ROOT)
    user_expected=manifest['files'][user_file]
    release_hash=hashlib.sha256(blob).hexdigest()
    preserved=release_hash==user_expected['sha256'] and len(blob)==user_expected['bytes']
    result=dict(schema='rc3-s1b-baseline-byte-preservation-v1',status='PASS' if not failures and preserved else 'FAIL',
        base_commit=base,baseline_files_checked=checked,baseline_bytes_checked=total,failures=failures,
        intentional_current_status_files_excluded=sorted(mutable),
        unrelated_user_worktree_file_excluded=user_file,release_blob_matches_frozen_manifest=preserved,
        release_user_file_sha256=release_hash,live_user_file_sha256=digest(ROOT/user_file),
        semantic_task_reads=0,optimizer_updates=0,teacher_queries=0,wall_s=time.perf_counter()-started)
    write_new(HERE/'FROZEN_EVIDENCE_AUDIT.json',result)
    print(json.dumps(result,ensure_ascii=True))
    if result['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
