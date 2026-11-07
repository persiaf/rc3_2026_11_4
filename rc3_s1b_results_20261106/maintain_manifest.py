"""Maintain release content hashes without staging a user's unrelated Word edits.

No research code is imported. Old baseline records stay byte-scoped; new work
is restricted to this task's reports/receipts/run and explicit mutable pointers.
"""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'project_management/GIT_MANIFEST.json'
PREFIXES=('rc3_s1b_results_20261106/',
    'rc3_global_local_execution_v3/authorizations/20261106_S1_B/',
    'rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/')
MUTABLE=('README.md','PROJECT_VERSION.json','rc3_global_local_execution_v3/README.md',
    'rc3_global_local_execution_v3/ACTIVE_EXECUTION_STATUS.json')

def record(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return dict(sha256=h.hexdigest(),bytes=p.stat().st_size)

def main():
    data=json.loads(MANIFEST.read_text(encoding='utf8'))
    files=data['files']
    new_count=0
    for prefix in PREFIXES:
        for p in (ROOT/prefix).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts or p.name.endswith(('.tmp','.pyc')):continue
            name=p.relative_to(ROOT).as_posix()
            # The old paid S1-A files have already passed the full preservation
            # audit. Avoid rehashing their large caches on each publication edit.
            if prefix==PREFIXES[2] and name in files:continue
            new_count+=name not in files
            files[name]=record(p)
    for name in MUTABLE:files[name]=record(ROOT/name)
    data['files']=dict(sorted(files.items()))
    data.update(version='2026.11.06-s1b-three-seed-complete',file_count=len(files),
        total_bytes=sum(r['bytes'] for r in files.values()),
        S1_B_scientific_execution=dict(status='COMPLETE_ALL3_SEEDS',experiment_id='rc3_v3_global_local_20261106_01',
            branch='codex/global-local-preflight',approved_source_manifest_sha256='0f0ffd2d2dda2fefefcfcb5cb26dc5197b5d11bd1332b0a5546992409049bfe0',
            seeds=[11622026,11622027,11622028],updates_per_seed=256,total_optimizer_updates=768,
            new_scene_navigation_native_H_teacher_queries=0,S1_A_original_files_unchanged=1482,
            report='rc3_s1b_results_20261106/README.md',S2_technical='GO',S2_authorized=False,S2_executed=False,
            S3='SEALED',original60='SEALED',main_merged=False,
            metadata_scope='Earlier phase objects preserve original-turn status; current state is this record and ACTIVE_EXECUTION_STATUS.'),
        S1_B_manifest_scope='Only task-owned reports/receipts/new run artifacts and explicit current pointers refreshed; unrelated live Word edit excluded and committed release blob retained.')
    MANIFEST.write_bytes((json.dumps(data,ensure_ascii=False,indent=2)+'\n').replace('\n','\r\n').encode('utf8'))
    print(json.dumps(dict(files=data['file_count'],bytes=data['total_bytes'],new_records=new_count)))

if __name__=='__main__':main()
