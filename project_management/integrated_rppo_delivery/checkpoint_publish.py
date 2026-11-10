"""Publish only immutable closed evidence while the authorized worker runs."""
import argparse
import json
import subprocess
from pathlib import Path
import time
from rc3_integrated_rppo.common import *


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,encoding='utf8').strip()


def main():
    a=argparse.ArgumentParser();a.add_argument('--label',required=True);args=a.parse_args()
    require(not git('diff','--cached','--name-only'),'Existing staged user changes: publication refuses to mix')
    events,tail=read_chain(OUT/'ledger/events.jsonl');files=set()
    completions=[e for e in events if e['event']=='request_done']
    for e in completions:
        ref=e['object'];require(digest(ROOT/ref['path'])==ref['sha256'],'Immutable result changed');files.add(ref['path'])
        p=OUT/'ledger/request_receipts'/(e['key'].replace(':','_')+'.json')
        if p.exists():files.add(label(p))
    closed=[e for e in events if e['event']=='checkpoint_closed'];last={}
    for e in closed:
        archive=OUT/'checkpoints'/str(e['seed'])/e['stage']/f"closed_{e['round']:04}.pt"
        if archive.exists():
            require(digest(archive)==e['checkpoint']['sha256'],'Closed archive differs from event');files.add(label(archive))
        last[(e['seed'],e['stage'])]=max(last.get((e['seed'],e['stage']),0),e['round'])
    for (seed,stage),round_ in last.items():
        for p in (OUT/'frames'/str(seed)/stage).glob('*.pt'):
            if int(p.stem)<=round_:files.add(label(p))
        for p in (OUT/'frames'/str(seed)/(stage+'_warm')).glob('*.pt'):files.add(label(p))
    for p in (OUT/'actors').rglob('*.pt.json'):
        d=read(p);require(digest(ROOT/d['path'])==d['actor_sha'],'Frozen actor changed');files.add(label(p));files.add(d['path'])
    counts={k:0 for k in CAPS}
    for e in events:
        if e['event']=='request_start':counts[e['query']['kind']]+=1
    snapshot=HOME/f'PARTIAL_PUBLICATION_{args.label}.json'
    write_new(snapshot,dict(status='PARTIAL_RUN_NOT_CLOSED_NO_PERFORMANCE_CLAIM',experiment=EXP,source_binding_sha256=digest(source_path()),
        protocol_sha256=digest(protocol_path()),ledger_prefix_events=len(events),ledger_prefix_tail=tail,
        requests_predebit=counts,completed_requests=len(completions),closed_rounds={str(k):v for k,v in last.items()},
        scope='Only immutable result objects/receipts/closed checkpoint archives and policy frames; live journal/HEAD/CURRENT excluded',utc=now()))
    files.add(label(snapshot));files.add('project_management/integrated_rppo_delivery/checkpoint_publish.py')
    files.add('.gitattributes');files.add('project_management/integrated_rppo_delivery/git_contention.py')
    for p in HOME.glob('GIT_CONTENTION_RECEIPT_*.json'):files.add(label(p))
    v=read(ROOT/'PROJECT_VERSION.json');v['integrated_rppo'].update(phase='RUNNING_SMALL256',partial_publication=label(snapshot))
    text=json.dumps(v,ensure_ascii=False,indent=2)+'\n';(ROOT/'PROJECT_VERSION.json').write_bytes(text.replace('\n','\r\n').encode('utf8'));files.add('PROJECT_VERSION.json')
    manifest=read(ROOT/'project_management/GIT_MANIFEST.json')
    for name in files:
        p=ROOT/name;manifest['files'][name]=dict(sha256=digest(p),bytes=p.stat().st_size)
    manifest['integrated_rppo_maintenance']['phase']='RUNNING_IMMUTABLE_PARTIAL_EVIDENCE'
    text=json.dumps(manifest,ensure_ascii=False,indent=2)+'\n';(ROOT/'project_management/GIT_MANIFEST.json').write_bytes(text.replace('\n','\r\n').encode('utf8'))
    files.add('project_management/GIT_MANIFEST.json');names=sorted(files)
    for i in range(0,len(names),40):subprocess.run(['git','add','-f','--',*names[i:i+40]],cwd=ROOT,check=True)
    journal=label(OUT/'ledger/events.jsonl')
    old_journal=subprocess.check_output(['git','show','HEAD:'+journal],cwd=ROOT)
    if old_journal.startswith(b'version https://git-lfs.github.com/spec/v1'):
        # Change storage representation using the EXACT already frozen P0
        # content, not by staging or touching the growing working journal.
        oid=next(x.split(b':',1)[1].decode() for x in old_journal.splitlines() if x.startswith(b'oid sha256:'))
        source=ROOT/'.git/lfs/objects'/oid[:2]/oid[2:4]/oid;require(digest(source)==oid,'Original frozen LFS prefix missing')
        content=source.read_bytes();blob=subprocess.check_output(['git','hash-object','-w','--stdin'],input=content,cwd=ROOT).decode().strip()
        subprocess.run(['git','update-index','--cacheinfo','100644,'+blob+','+journal],cwd=ROOT,check=True)
        require(subprocess.check_output(['git','show',':'+journal],cwd=ROOT)==content,'P0 prefix storage normalization altered bytes')
    subprocess.run(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-q','-m','Publish immutable Integrated RPPO partial evidence '+args.label],cwd=ROOT,check=True)
    print(json.dumps(dict(commit=git('rev-parse','HEAD'),files=len(files),live_model_or_log_written=False)),flush=True)


if __name__=='__main__':main()
