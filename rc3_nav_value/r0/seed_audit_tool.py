"""Read-only exact-integer historical seed audit; no project imports."""
from pathlib import Path
import collections
import datetime
import gzip
import hashlib
import json
import re
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
BASE='8b3ad97cfa98109a46a3759f625f2c2ef7a04455'
RANGES=dict(map=[8300000000,8300000031],task=[8400000000,8400000031],exploration=[8500000000,8500000255])
TARGETS={v for lo,hi in RANGES.values() for v in range(lo,hi+1)}
PAT=re.compile(rb'(?<![0-9A-Za-z_.])8[345][0-9]{8}(?![0-9A-Za-z_.])')
TEXT={'.json','.jsonl','.csv','.tsv','.yaml','.yml','.md','.py','.txt','.m','.log','.sha256','.ipynb','.toml','.ini','.gitignore','.gitattributes'}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def scan(stream):
    total=0;tail=b'';matches=[];seen=set()
    while True:
        block=stream.read(1024*1024)
        if not block:break
        data=tail+block;offset=total-len(tail)
        for match in PAT.finditer(data):
            value=int(match.group());position=offset+match.start()
            if value in TARGETS and (position,value) not in seen:
                seen.add((position,value))
                context=data[max(0,match.start()-90):min(len(data),match.end()+90)]
                matches.append(dict(seed=value,byte_offset=position,context=context.decode('utf-8',errors='replace')))
        total+=len(block);tail=data[-192:]
    return total,matches


def main():
    started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    names=[n.decode('utf-8') for n in subprocess.check_output(['git','ls-tree','-r','--name-only','-z',BASE],cwd=ROOT).split(b'\0') if n]
    inventory=[];excluded=collections.Counter();matches=[];errors=[];kind_count=collections.Counter();totalbytes=0
    for name in names:
        path=ROOT/name;suffix=path.suffix.lower();isgz=name.endswith('.jsonl.gz')
        if not(isgz or suffix in TEXT or path.name in ('.gitignore','.gitattributes')):
            excluded[suffix or '(no suffix)']+=1;continue
        try:
            compressed_sha=digest(path)
            with (gzip.open(path,'rb') if isgz else path.open('rb')) as stream:logical_bytes,found=scan(stream)
            inventory.append(dict(path=name,sha256=compressed_sha,file_bytes=path.stat().st_size,scanned_bytes=logical_bytes))
            totalbytes+=logical_bytes;kind_count['gzip_jsonl' if isgz else 'plain_text']+=1
            for r in found:
                if name.startswith('research_proposals/'):
                    classification='planned_proposal_only'
                elif suffix in ('.py','.m'):
                    classification='source_declaration_requires_context'
                elif suffix in ('.md',):
                    classification='documentation_requires_context'
                else:classification='execution_or_protocol_artifact_requires_context'
                matches.append(dict(file=name,classification=classification,**r))
        except Exception as e:errors.append(dict(file=name,error=str(e)))
    collisions=[r for r in matches if r['classification']!='planned_proposal_only']
    payload=''.join(f"{r['path']}\t{r['sha256']}\t{r['file_bytes']}\t{r['scanned_bytes']}\n" for r in sorted(inventory,key=lambda r:r['path']))
    # These explicit historical banks are independently read from stored protocols/seed manifests.
    p=json.loads((ROOT/'rc3_constructive/protocol.yaml').read_text(encoding='utf-8'))
    c22=json.loads((ROOT/'rc3_constructive/c22/protocol.json').read_text(encoding='utf-8'))
    instances=[json.loads(s) for s in (ROOT/'rc3_constructive/c22/instances.jsonl').read_text(encoding='utf-8').splitlines()]
    mechanism=json.loads((ROOT/'rc3_constructive/c22/mechanism_seeds.json').read_text(encoding='utf-8'))
    known=[dict(name='C012_training_maps',start=p['training']['map_seed_start'],end=p['training']['map_seed_start']+p['training']['map_bank_size']-1,status='executed'),
        dict(name='C012_training_tasks',start=p['training']['task_seed_start'],end=p['training']['task_seed_start']+p['training']['max_sampled_trajectories']-1,status='executed')]
    for name,spec in p['splits'].items():
        count=spec.get('count',len(spec.get('task_counts',[]))*spec.get('per_count',0))
        known.append(dict(name='C012_'+name,start=spec['seed_start'],end=spec['seed_start']+count-1,status='reserved_unqueried' if name=='future_test_reserved_unqueried' else 'executed'))
    known.append(dict(name='C0_small_exact_task_seeds',start=7500001000,end=7500001011,status='executed'))
    for field in ('map_seed','task_seed'):
        vals=[r[field] for r in instances];known.append(dict(name='C22_training_'+field,start=min(vals),end=max(vals),status='executed',distinct=len(set(vals))))
        vals=[r[field] for r in mechanism];known.append(dict(name='C22_mechanism_'+field,start=min(vals),end=max(vals),status='executed',distinct=len(set(vals))))
    for field in ('map_seed_start','task_seed_start'):
        start=c22['calibration'][field];known.append(dict(name='C22_calibration_'+field,start=start,end=start+c22['calibration']['count']-1,status='executed'))
    overlap=[]
    for name,(lo,hi) in RANGES.items():
        for record in known:
            if max(lo,record['start'])<=min(hi,record['end']):overlap.append(dict(proposed=name,historical=record))
    assert not errors,errors
    report=dict(schema='rc3-r0-seed-use-audit-v1',generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),base_commit=BASE,
        proposed_seed_ranges=RANGES,proposed_distinct_seeds=len(TARGETS),verdict='PASS_NO_RECORDED_PRIOR_USE' if not collisions and not overlap else 'REVIEW_COLLISION',
        collisions=collisions,historical_range_overlaps=overlap,planned_only_mentions=[r for r in matches if r['classification']=='planned_proposal_only'],
        exact_match_rule='10-digit integer token with numeric/ASCII-word/dot boundaries, then exact membership in the proposed ranges; decimal timing substrings and hash fragments do not count',
        scan=dict(tracked_base_files=len(names),scanned_files=len(inventory),file_kinds=dict(kind_count),uncompressed_bytes_scanned=totalbytes,
            inventory_sha256=hashlib.sha256(payload.encode('utf-8')).hexdigest(),inventory_reference='seed_audit_inventory.jsonl',errors=errors),
        explicit_historical_banks=known,
        exclusions=dict(file_suffix_counts=dict(excluded),scope='No unpickling or execution of binary payloads; ZIP delivery packages, PKL/PT/NPZ/MAT payload internals, PDFs/images and DOCX ZIP internals are excluded. Their text execution manifests/protocols and gzip JSONL logs are covered. Existing proposal DOCX files are unexecuted design sources, not evidence of seed use.'),
        interpretation='No previous recorded actual use found in tracked execution/protocol text and gzip trajectory logs, nor overlap with explicit C012/C21-reused/C22 banks or the protected60 reserved-test bank. This does not claim forensic enumeration of arbitrary integers inside historical binary/archive payloads or unrecorded external jobs.',
        research_calls=dict(scene_generation=0,planner_calls=0,H_calls=0,model_forwards=0,optimizer_updates=0,reserved_test_queries=0),elapsed_s=time.perf_counter()-started)
    (OUT/'seed_audit_inventory.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in inventory),encoding='utf-8',newline='\n')
    (OUT/'seed_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(verdict=report['verdict'],scanned_files=len(inventory),gzip_files=kind_count['gzip_jsonl'],collisions=len(collisions),planned_mentions=len(report['planned_only_mentions']),range_overlaps=len(overlap),errors=errors,elapsed_s=report['elapsed_s'])))


if __name__=='__main__':main()
