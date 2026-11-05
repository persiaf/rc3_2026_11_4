"""R1 preflight seed audit of tracked historical records, without project imports.

No scenes, planning, H, models or training are executed. R0 artifacts untouched.
"""
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
BASE='129ba6e916ed5a44ecb92998553c082b4510ecd1'
RANGES=dict(train_map=[8600000000,8600000031],train_task=[8700000000,8700000031],
    development_map=[8800000000,8800000031],development_task=[8900000000,8900000031],
    exploration=[9000000000,9000009999])
SAMPLING=11412026
TARGETS={v for lo,hi in RANGES.values() for v in range(lo,hi+1)}
PAT=re.compile(rb'(?<![0-9A-Za-z_.])(?:(?:8[6-9]|90)[0-9]{8}|11412026)(?![0-9A-Za-z_.])')
PATHPAT=re.compile(r'(?<![0-9.])(?:(?:8[6-9]|90)[0-9]{8}|11412026)(?![0-9.])')
TEXT={'.json','.jsonl','.csv','.tsv','.yaml','.yml','.md','.py','.txt','.m','.log','.sha256','.ipynb','.toml','.ini'}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def scan(stream):
    total=0;tail=b'';found=[];seen=set()
    while True:
        block=stream.read(1024*1024)
        if not block:break
        data=tail+block;offset=total-len(tail)
        for m in PAT.finditer(data):
            value=int(m.group());position=offset+m.start()
            if (value in TARGETS or value==SAMPLING) and (position,value) not in seen:
                seen.add((position,value))
                found.append(dict(seed=value,byte_offset=position,
                    context=data[max(0,m.start()-100):min(len(data),m.end()+100)].decode('utf-8',errors='replace')))
        total+=len(block);tail=data[-256:]
    return total,found


def classify(name,suffix):
    if name.startswith('research_proposals/'):return 'planned_proposal_only'
    if suffix in ('.py','.m'):return 'source_declaration_requires_context'
    if suffix=='.md':return 'documentation_requires_context'
    return 'execution_or_protocol_artifact_requires_context'


def main():
    started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'seed_audit.json').exists():raise FileExistsError('R1 seed audit already frozen; no implicit overwrite')
    names=[n.decode('utf-8') for n in subprocess.check_output(['git','ls-tree','-r','--name-only','-z',BASE],cwd=ROOT).split(b'\0') if n]
    inventory=[];excluded=collections.Counter();mentions=[];errors=[];kind_count=collections.Counter();logical_total=0
    for name in names:
        path=ROOT/name;suffix=path.suffix.lower();isgz=name.endswith('.jsonl.gz')
        if not(isgz or suffix in TEXT or path.name in ('.gitignore','.gitattributes')):
            excluded[suffix or '(no suffix)']+=1;continue
        try:
            raw_sha=digest(path)
            with (gzip.open(path,'rb') if isgz else path.open('rb')) as f:logical,found=scan(f)
            inventory.append(dict(path=name,sha256=raw_sha,file_bytes=path.stat().st_size,scanned_bytes=logical))
            logical_total+=logical;kind_count['gzip_jsonl' if isgz else 'plain_text']+=1
            for r in found:mentions.append(dict(file=name,classification=classify(name,suffix),**r))
        except Exception as e:errors.append(dict(file=name,error=str(e)))
    name_hits=[dict(file=n,seed=int(m.group())) for n in names for m in PATHPAT.finditer(n)
        if int(m.group()) in TARGETS or int(m.group())==SAMPLING]
    collisions=[r for r in mentions if r['seed'] in TARGETS and r['classification']!='planned_proposal_only']
    filename_collisions=[r for r in name_hits if r['seed'] in TARGETS]
    sampling_mentions=[r for r in mentions if r['seed']==SAMPLING]
    sampling_names=[r for r in name_hits if r['seed']==SAMPLING]
    p=json.loads((ROOT/'rc3_constructive/protocol.yaml').read_text(encoding='utf-8'))
    c22=json.loads((ROOT/'rc3_constructive/c22/protocol.json').read_text(encoding='utf-8'))
    instances=[json.loads(s) for s in (ROOT/'rc3_constructive/c22/instances.jsonl').read_text(encoding='utf-8').splitlines()]
    mechanism=json.loads((ROOT/'rc3_constructive/c22/mechanism_seeds.json').read_text(encoding='utf-8'))
    r0=json.loads((ROOT/'rc3_nav_value/r0/seeds.json').read_text(encoding='utf-8'))
    known=[dict(name='C012_training_maps',start=p['training']['map_seed_start'],end=p['training']['map_seed_start']+p['training']['map_bank_size']-1,status='executed'),
        dict(name='C012_training_tasks',start=p['training']['task_seed_start'],end=p['training']['task_seed_start']+p['training']['max_sampled_trajectories']-1,status='executed')]
    for name,spec in p['splits'].items():
        count=spec.get('count',len(spec.get('task_counts',[]))*spec.get('per_count',0))
        known.append(dict(name='C012_'+name,start=spec['seed_start'],end=spec['seed_start']+count-1,
            status='protected_reserved_unqueried' if name=='future_test_reserved_unqueried' else 'executed'))
    known.append(dict(name='C0_small_exact_task_seeds',start=7500001000,end=7500001011,status='executed'))
    for label,records in (('C22_training',instances),('C22_mechanism',mechanism),('R0_primary',r0)):
        for field in ('map_seed','task_seed'):
            values=[r[field] for r in records]
            known.append(dict(name=label+'_'+field,start=min(values),end=max(values),distinct=len(set(values)),status='executed'))
    for field in ('map_seed_start','task_seed_start'):
        start=c22['calibration'][field]
        known.append(dict(name='C22_calibration_'+field,start=start,end=start+c22['calibration']['count']-1,status='executed'))
    known.append(dict(name='R0_exploration_declared_bank',start=8500000000,end=8500000255,status='executed_or_reserved_R0_protocol_bank'))
    overlap=[dict(proposed=name,historical=old) for name,(lo,hi) in RANGES.items() for old in known
        if max(lo,old['start'])<=min(hi,old['end'])]
    assert not errors,errors
    # Inventory file bytes are authoritative. Canonical TSV has a separate name.
    inventory_bytes=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in inventory).encode('utf-8')
    inventory_path=OUT/'seed_audit_inventory.jsonl';inventory_path.write_bytes(inventory_bytes)
    raw_inventory_sha=digest(inventory_path)
    assert raw_inventory_sha==hashlib.sha256(inventory_bytes).hexdigest()
    canonical=''.join(f"{r['path']}\t{r['sha256']}\t{r['file_bytes']}\t{r['scanned_bytes']}\n"
        for r in sorted(inventory,key=lambda r:r['path'])).encode('utf-8')
    report=dict(schema='rc3-r1-seed-use-audit-v1',generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),base_commit=BASE,
        proposed_seed_ranges=RANGES,proposed_scene_and_exploration_seed_count=len(TARGETS),
        policy_sampling_seed=dict(value=SAMPLING,type='policy RNG seed, not a scene/map/task seed',prior_exact_mentions=sampling_mentions,
            prior_filename_mentions=sampling_names,status='NO_RECORDED_PRIOR_MENTION' if not sampling_mentions and not sampling_names else 'MENTIONS_REQUIRING_CONTEXT'),
        verdict='PASS_NO_RECORDED_PRIOR_USE' if not collisions and not filename_collisions and not overlap else 'REVIEW_COLLISION',
        collisions=collisions,filename_collisions=filename_collisions,historical_range_overlaps=overlap,
        planned_only_mentions=[r for r in mentions if r['classification']=='planned_proposal_only'],
        exact_match_rule='Integer token boundaries excluding adjacent digits, ASCII-word characters or decimal dots; exact range membership. Filenames separately use numeric boundaries permitting underscore. Decimal timing fragments and longer integers excluded.',
        scan=dict(tracked_base_files=len(names),scanned_files=len(inventory),file_kinds=dict(kind_count),uncompressed_bytes_scanned=logical_total,
            filenames_checked=len(names),errors=errors),
        inventory_file='seed_audit_inventory.jsonl',inventory_file_sha256=raw_inventory_sha,
        inventory_file_hash_semantics='SHA256 of actual JSONL file bytes as written, including its UTF8 and LF newlines',
        canonical_inventory_sha256=hashlib.sha256(canonical).hexdigest(),
        canonical_inventory_format='Sorted UTF8 TSV rows: path TAB sha256 TAB physical_file_bytes TAB scanned_uncompressed_bytes LF; this hash is NOT the JSONL file hash',
        audited_tool_sha256=digest(Path(__file__)),explicit_historical_banks=known,
        exclusions=dict(file_suffix_counts=dict(excluded),scope='No unpickling or execution of binary payloads. Historical ZIP/PKL/PT/NPZ/MAT/PDF/image/DOCX internals excluded; their text execution manifests/protocols and gzip JSONL logs covered. No claim about unrecorded external runs.'),
        interpretation='No recorded prior actual use in the audited tracked execution/protocol text, gzip routes or filenames and no overlap with explicit historical/protected banks. Proposed-only design mentions are not counted as executed use. Binary/archive scope limitation remains explicit.',
        research_calls=dict(scene_generation=0,planner_calls=0,H_calls=0,model_forwards=0,optimizer_updates=0,reserved_test_queries=0),elapsed_s=time.perf_counter()-started)
    (OUT/'seed_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(verdict=report['verdict'],scanned_files=len(inventory),gzip_files=kind_count['gzip_jsonl'],collisions=len(collisions),filename_collisions=len(filename_collisions),range_overlaps=len(overlap),sampling_mentions=len(sampling_mentions),inventory_file_sha256=raw_inventory_sha,elapsed_s=report['elapsed_s'])))


if __name__=='__main__':main()
