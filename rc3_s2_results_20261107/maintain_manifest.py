"""Update only task-owned release hashes, preserving frozen and user content."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'project_management/GIT_MANIFEST.json'
PREFIXES=('rc3_s2_results_20261107/','rc3_global_local_execution_v3/authorizations/20261107_S2/',
    'rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/',
    'paper/review-traces/experiment-result-to-claim/20261107_run01/')
MUTABLE=('.gitattributes','README.md','PROJECT_VERSION.json','rc3_global_local_execution_v3/README.md',
    'rc3_global_local_execution_v3/ACTIVE_EXECUTION_STATUS.json','EXPERIMENT_RESULT_TO_CLAIM.md')

def entry(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return dict(sha256=h.hexdigest(),bytes=p.stat().st_size)

def main():
    m=json.loads(PATH.read_text(encoding='utf8'));files=m['files'];added=0
    for prefix in PREFIXES:
        for p in (ROOT/prefix).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts or p.name.endswith(('.pyc','.tmp')):continue
            name=p.relative_to(ROOT).as_posix()
            if prefix==PREFIXES[2] and name in files:continue
            added+=name not in files;files[name]=entry(p)
    for name in MUTABLE:files[name]=entry(ROOT/name)
    m.update(version='2026.11.07-s2-closeout',files=dict(sorted(files.items())),file_count=len(files),
        total_bytes=sum(r['bytes'] for r in files.values()),
        S2_scientific_execution=dict(report='rc3_s2_results_20261107/README.md',stage_status='rc3_s2_results_20261107/STAGE_STATUS.json',
            experiment_id='rc3_v3_global_local_20261106_01',protocol_sha256='0c5bc7d7a9557edf2af76f51a9f418550f0b4ce1ec6a8b47ddea95c6256e78bb',
            source_manifest_sha256='0f0ffd2d2dda2fefefcfcb5cb26dc5197b5d11bd1332b0a5546992409049bfe0',
            source_approved=True,S2_authorized=True,S3_authorized=False,original60='SEALED',main_merged=False),
        S2_manifest_scope='Only new S2 run/authority/report files and explicit mutable Git/status pointers refreshed. Old scientific bytes and unrelated live Word edits not replaced.')
    PATH.write_bytes((json.dumps(m,ensure_ascii=False,indent=2)+'\n').replace('\n','\r\n').encode('utf8'))
    print(json.dumps(dict(files=m['file_count'],bytes=m['total_bytes'],added=added)))

if __name__=='__main__':main()
