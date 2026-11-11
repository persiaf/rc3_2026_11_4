import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

HOME=Path(__file__).resolve().parent
ROOT=HOME.parent
BASE='16f5d05952c03c7ddf20623ddcc59b09c0011040'
TABLE='rc3_r04_m2_20261111/attempts/02/FIT_ONLY_COMPONENT_ANCHORS.json'
ROLES=('PG-global','PG-cond','continued_SL_A1')
METHODS=('FrozenSL','continued_SL_A1','PG-global','PG-cond','Fixed3','Detour16')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def put(p,v):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def append(p,v):
    with Path(p).open('a',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def archive(p,v):
    raw=(json.dumps(v,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode()
    data=gzip.compress(raw,compresslevel=1,mtime=0)
    with Path(p).open('ab') as f:
        offset=f.tell();f.write(data);f.flush();os.fsync(f.fileno())
    return dict(offset=offset,compressed_bytes=len(data),sha256=hashlib.sha256(raw).hexdigest())
def csvrow(p,row):
    exists=Path(p).exists()
    with Path(p).open('a',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row));
        if not exists:w.writeheader()
        w.writerow(row);f.flush();os.fsync(f.fileno())

class Ledger:
    def __init__(self,stage,start_m,start_w):
        self.stage=stage;self.path=HOME/stage;self.path.mkdir(exist_ok=False)
        self.start_m=start_m;self.start_w=start_w
        self.prior=read(HOME/'REGISTRATION_RECEIPT.json')['wall_s'] if stage=='engineering' else 0.
        self.cap=read(HOME/'PROTOCOL.json')['caps'][stage]
        self.counts=dict(requests=0,requests_completed=0,requests_failed=0,actor_optimizer_attempts=0,
            actor_optimizer_completed=0,critic_optimizer_attempts=0,teacher_replays=0,new_teacher_requests=0,
            new_scenes=0,navigation_solves=0,original60_reads=0,oldS3_reads=0,
            full_H_calls=0,native_requests=0,feature_states=0,feature_s=0.,input_prepare_s=0.,request_io_s=0.,
            internal_search_counts={},internal_search_inclusive_s={},internal_search_exclusive_s={})
        self.ids=set();self.status='RUNNING'
    def elapsed(self):return self.prior+max(time.monotonic()-self.start_m,time.time()-self.start_w)
    def check(self):
        if self.elapsed()>self.cap['wall_s']-15:raise TimeoutError('Stage cumulative stop boundary')
    def event(self,event,**kw):
        self.check();append(self.path/'EVENTS.jsonl',dict(event=event,elapsed_s=self.elapsed(),**kw))
    def optimizer(self,role,op):
        self.check()
        if self.counts['actor_optimizer_attempts']>=self.cap['actor_optimizer_attempts']:raise RuntimeError('Actor attempt cap')
        self.counts['actor_optimizer_attempts']+=1;self.event('actor_optimizer_debit',role=role,opportunity=op,count=self.counts['actor_optimizer_attempts'])
    def request(self,rid,compute,*,role,op=0,kind='training'):
        self.check()
        if rid in self.ids or self.counts['requests']>=self.cap['requests']:raise RuntimeError('Duplicate request or request cap')
        self.ids.add(rid);self.counts['requests']+=1;self.event('request_debit',request_id=rid,role=role,opportunity=op,kind=kind)
        start=time.perf_counter()
        try:
            result=compute();record=result['record'] if 'record' in result else result
            record.update(request_id=rid,method=role,opportunity=op,request_kind=kind,stage=self.stage)
            self.counts['native_requests']+=int(role in ('Fixed3','Detour16'))
            self.counts['full_H_calls']+=record.get('counts',{}).get('full_H_calls',0)
            for key in ('counts','inclusive_s','exclusive_s'):
                dst=self.counts['internal_search_'+key]
                for k,v in record.get('internal_search',{}).get(key,{}).items():dst[k]=dst.get(k,0)+v
            feature_s=record.get('components_s',{}).get('dynamic_features_s',0.)
            self.counts['feature_s']+=feature_s;self.counts['feature_states']+=len(record.get('actions',[]))
            self.counts['input_prepare_s']+=record.get('timing',{}).get('input_prepare_s',0.)
            tick=time.perf_counter();ref=archive(self.path/'REQUESTS.jsonl.gz',record);io=time.perf_counter()-tick
            self.counts['request_io_s']+=io;self.counts['requests_completed']+=1
            row=dict(request_id=rid,role=role,opportunity=op,kind=kind,case_id=record.get('case_id',''),
                cohort=record.get('cohort',''),physical_key=record.get('physical_key',''),
                width=record['spec']['width'],height=record['spec']['height'],
                **record['metrics'],valid=bool(record['plan']['validation']['valid_model_plan']),
                request_wall_s=time.perf_counter()-start,io_s=io,input_prepare_s=record.get('timing',{}).get('input_prepare_s',0.),
                model_load_s=record.get('timing',{}).get('model_load_s',0.),feature_s=feature_s,
                H_s=record.get('H_s',0.),native_full_H_calls=record.get('counts',{}).get('full_H_calls',0),
                added_feature_s=sum(record.get('new_feature_overhead_s',{}).values()),
                added_feature_calls=sum(record.get('new_feature_calls',{}).values()),**ref)
            csvrow(self.path/'REQUEST_INDEX.csv',row)
            self.event('request_complete',request_id=rid,wall_s=row['request_wall_s'],record=ref)
            full=time.perf_counter()-start
            csvrow(self.path/'REQUEST_TIMING.csv',dict(request_id=rid,full_request_s=full,
                archive_IO_s=io,index_and_journal_s=max(0.,full-row['request_wall_s']),
                timing_receipt_scope='Through complete record, index and completion journal; timing receipt own write separately in stage wall'))
            return result
        except BaseException as e:
            self.counts['requests_failed']+=1;self.event('request_failure',request_id=rid,error=repr(e));raise
    def snapshot(self):return dict(stage=self.stage,status=self.status,wall_s=self.elapsed(),counts=self.counts,caps=self.cap)
