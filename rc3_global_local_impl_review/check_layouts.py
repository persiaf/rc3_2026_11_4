"""Execute only the v2 physical-geometry independence gate; no S3 task draws."""
import hashlib
import json
import os
from pathlib import Path
import time
from .geometry_only import make_layout,physical_payload,map_key

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def append(p,d):
    with p.open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(d,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())


def register():
    target=OUT/'GEOMETRY_CHECK_PROTOCOL.json'
    if target.exists():raise FileExistsError('No implicit replacement of geometry protocol')
    p=read(ROOT/'rc3_global_local/S2_S3_protocol_v2.json')
    write(target,dict(schema='implementation-preflight-geometry-only-v1',base='057830d584b6c71c79ef5118087c681c98135fe2',
        active_v2_sha256=sha(ROOT/'rc3_global_local/S2_S3_protocol_v2.json'),seed_registry_sha256=sha(ROOT/'rc3_global_local/future_seeds_v2.json'),
        source_sha256={n:sha(ROOT/n) for n in ('rc3_global_local_impl_review/geometry_only.py','rc3_global_local_impl_review/check_layouts.py',
            'rc3_fast/scenes.py','rc3_fast/planner.py','rc3_baseline/baseline.py')},
        limits=dict(reference_layout_reconstructions=3,declared_new_physical_layouts=111,target_task_generations=0,navigation_preparations=0,
            model_forwards=0,backward_calls=0,optimizer_updates=0,S3_task_queries=0,original60_reads=0,wall_s=600),
        scope=p['layout_independence_gate'],failure='Stop on source drift or collision; no seed/mode replacement; keep failed record'))


def run():
    p=read(OUT/'GEOMETRY_CHECK_PROTOCOL.json')
    for n,h in p['source_sha256'].items():assert sha(ROOT/n)==h,n
    started=time.perf_counter();journal=OUT/'geometry_journal.jsonl'
    if journal.exists():raise FileExistsError('Geometry gate already attempted; no implicit replay')
    registry=read(ROOT/'rc3_global_local/future_seeds_v2.json');records=[];references=[];seen_keys={}
    counts=dict(reference_layout_attempts=0,reference_layouts_completed=0,layout_attempts=0,layouts_completed=0,
        target_task_generations=0,navigation_preparations=0,model_forwards=0,backward_calls=0,optimizer_updates=0,S3_task_queries=0,original60_reads=0)
    (OUT/'geometry').mkdir(exist_ok=True)
    def checktime():
        if time.perf_counter()-started>=600:raise TimeoutError('Geometry-only budget reached')
    def build(role,label,seed,mode):
        checktime();counts['layout_attempts']+=1
        if counts['layout_attempts']>111:raise RuntimeError('Geometry construction cap')
        append(journal,dict(event='start',role=role,label=label,map_seed=seed,mode=mode))
        scene=make_layout(10,10,seed,mode);assert not scene['tasks'];key=map_key(scene)
        if key in seen_keys:raise ValueError('Duplicate/cross-group geometry: '+str((role,label,seen_keys[key])))
        seen_keys[key]=(role,label)
        raw=json.dumps(physical_payload(scene),sort_keys=True,separators=(',',':')).encode()
        assert hashlib.sha256(raw).hexdigest()==key
        path=OUT/'geometry'/f'{role}_{label:02}.json';path.write_bytes(raw)
        r=dict(role=role,label=label,map_seed=seed,mode=mode,map_key=key,file=path.relative_to(ROOT).as_posix(),bytes=len(raw),task_records=0)
        records.append(r);counts['layouts_completed']+=1;append(journal,dict(event='complete',**r))
        return r
    try:
        for i in range(3):
            checktime();counts['reference_layout_attempts']+=1
            saved=read(ROOT/f'rc3_global_local/parents/{i:02}_scene.json');mode=saved['complexity']
            rebuilt=make_layout(10,10,saved['seed'],mode)
            assert rebuilt['tasks']==[] and map_key(rebuilt)==map_key(saved)
            assert physical_payload(rebuilt)==physical_payload(saved)
            references.append(dict(case=i,mode=mode,map_seed=saved['seed'],map_key=map_key(saved),matches_existing_saved_S0=True))
            counts['reference_layouts_completed']+=1
        train={}
        for r in registry['S2_train']:train[r['map_seed']]=build('train',r['parent'],r['map_seed'],r['mode'])
        dev={r['map_seed']:r for r in registry['S2_development']}
        for i,(ms,r) in enumerate(sorted(dev.items())):build('development',i,ms,r['mode'])
        unique={(r['group'],r['parent']):r for r in registry['S3']}
        seen=[]
        for (group,label),r in sorted(unique.items()):
            if group=='seen_layout_new_tasks':
                ref=train[r['map_seed']];assert ref['mode']==r['mode']
                seen.append(dict(role='seen',label=label,map_seed=r['map_seed'],mode=r['mode'],map_key=ref['map_key'],source_train_label=ref['label']))
            else:build('unseen',label,r['map_seed'],r['mode'])
        assert len(seen)==6 and len({r['map_key'] for r in seen})==6
        assert counts['layouts_completed']==111
        checktime();status='PASS'
    except Exception as exc:
        status='FAIL_STOP';write(OUT/'GEOMETRY_GATE_FAILURE.json',dict(error=type(exc).__name__,message=str(exc),counts=counts))
        raise
    finally:
        write(OUT/'GEOMETRY_GATE.json',dict(schema='implementation-physical-layout-gate-v1',status=status,counts=counts,
            references=references,records=records,seen=locals().get('seen',[]),elapsed_wall_s=time.perf_counter()-started,
            clusters=dict(train=96,development=3,seen=6,unseen=12) if status=='PASS' else None,
            gate='Unique canonical physical keys; train/dev/unseen mutually disjoint; seen six are declared train keys',
            S3_scope='Only v2-authorized geometry assets, no S3 target seed drawn, no F/nav/route/cost/model; not a performance test',
            original60='sealed'))
    print(json.dumps(dict(status=status,counts=counts,elapsed_wall_s=time.perf_counter()-started)))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--register',action='store_true');args=parser.parse_args()
    register() if args.register else run()
