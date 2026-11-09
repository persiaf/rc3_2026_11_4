"""Locate authenticated navigation artifacts; never generates tasks or navigation."""
import json
import subprocess
import time
from pathlib import Path
from rc3_s3_evaluator.common import ROOT,read,digest,identity,write_new,label,require,now
from rc3_fast.planner import map_key
from rc3_global_local_impl_review.runtime_guards import current_navigation_sources

HOME=Path(__file__).resolve().parent
INPUT=ROOT/'rc3_s3_evaluator/runs/rc3_v3_global_local_20261106_01/continuations/persistence_r1/inputs/S3_042'


def main():
    start=time.perf_counter()
    write_new(HOME/'SEARCH_START.json',dict(utc=now(),navigation_call_cap=0,original60_semantic_reads=0))
    receipt=read(INPUT/'RECEIPT.json');scene=read(INPUT/'scene.json')
    require(digest(INPUT/'scene.json')==receipt['scene_sha256'] and
        digest(INPUT/'arrays.npz')==receipt['arrays_sha256'] and identity(scene)==receipt['content']['scene'],'Saved input changed')
    key=map_key(scene);sources=current_navigation_sources()
    cmd=['rg','--files','--hidden','--no-ignore','-g','*.npz','-g','*.pkl','-g','!**/.venv/**',
        '-g','!**/.git/**','-g','!project_backups/**','-g','!**/__pycache__/**','-g','!**/node_modules/**']
    listing=subprocess.check_output(cmd,cwd=ROOT).decode('utf8').splitlines()
    paths=[Path(n.replace('\\','/')) for n in listing]
    # S1/S2 navigation is map-key named; S0 is explicitly receipt-labelled.
    candidates=[p for p in paths if 'navigation' in p.parts or p.stem==key or
        (p.parts[:3]==('rc3_global_local','outputs','parents') and p.name.endswith('_arrays.npz'))]
    checked=[];matches=[]
    for p in candidates:
        full=ROOT/p;side=full.with_suffix('.json')
        if not side.is_file():
            checked.append(dict(path=p.as_posix(),status='NO_NAVIGATION_PROVENANCE_RECEIPT'));continue
        r=read(side);k=r.get('map_key',r.get('geometry_key'))
        src=r.get('source_sha256',r.get('navigation_source_sha256'))
        record=dict(path=p.as_posix(),receipt=label(side),receipt_sha256=digest(side),map_key=k,
            same_physical_map=(k==key),same_navigation_sources=(src==sources))
        checked.append(record)
        if k==key and src==sources:matches.append(record)
    result=dict(schema='rc3-s3-existing-navigation-search-v1',saved_input=label(INPUT),
        input_files={p.name:dict(path=label(p),sha256=digest(p),bytes=p.stat().st_size)
            for p in (INPUT/'scene.json',INPUT/'arrays.npz',INPUT/'RECEIPT.json')},
        physical_map_key=key,navigation_sources=sources,inventory_paths=[p.as_posix() for p in paths],
        inventory_count=len(paths),candidate_receipts_checked=checked,authenticated_matches=matches,
        protected60_semantic_reads=0,scope='Filename inventory only outside labelled navigation/S0 artifacts; no protected task contents or model payloads read',
        actual_navigation_calls=0,actual_generation_model_native_H_teacher_optimizer=0)
    write_new(HOME/'NAVIGATION_SEARCH.json',result)
    write_new(HOME/'SEARCH_RESOURCE.json',dict(read_only_search_wall_s=time.perf_counter()-start,
        input_file_hash_reads=3,navigation_metadata_receipts=len(checked),actual_navigation_calls=0,
        result_sha256=digest(HOME/'NAVIGATION_SEARCH.json'),matches=len(matches)))
    print(dict(map_key=key,inventory=len(paths),candidates=len(checked),matches=matches,
        wall_s=read(HOME/'SEARCH_RESOURCE.json')['read_only_search_wall_s']))


if __name__=='__main__':main()
