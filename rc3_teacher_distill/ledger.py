"""Separate T-series journal with frozen R1 terminal H/certification code reused."""
import math
import threading
from pathlib import Path
import time
from rc3_nav_value.r1.oracle import Ledger, INTERNAL, CERTIFICATION, _append

LIMITS = dict(train=3072, development=768, audit=256, reference=512,
              T2_routes=3072,T2_teacher=1368)


class TLedger(Ledger):
    def __init__(self,out):
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        self.journal=self.out/'call_journal.jsonl';self.terminal=self.out/'terminal.jsonl'
        self.cache_journal=self.out/'cache_hits.jsonl'
        if self.journal.exists():
            raise FileExistsError('No implicit run replay or overwrite')
        self.deadline=math.inf;self.check_hook=None;self.lock=threading.RLock()
        self.used={k:0 for k in LIMITS};self.started=self.used
        self.completed={k:0 for k in LIMITS};self.failed={k:0 for k in LIMITS}
        self.internal_counts={k:0 for k in INTERNAL}
        self.internal_inclusive_s={k:0. for k in INTERNAL}
        self.internal_exclusive_s={k:0. for k in INTERNAL}
        self.certification_counts={k:0 for k in CERTIFICATION}
        self.cache_hits=0;self.oracle_s=0.

    def _reserve(self,kind,category,metadata):
        self.check()
        if category not in LIMITS or self.used[category]>=LIMITS[category]:
            raise RuntimeError('Frozen T query category exhausted')
        if self.total>=sum(LIMITS.values()):
            raise RuntimeError('Frozen T total query budget exhausted')
        call_id=self.total+1
        _append(self.journal,dict(event='start',call_id=call_id,kind=kind,
            category=category,metadata=metadata,clock_s=time.perf_counter()))
        self.used[category]+=1
        return call_id

    def snapshot(self):
        return dict(schema='rc3-T-independent-ledger-v1',limits=LIMITS,
            total_limit=sum(LIMITS.values()),T1_limit=4608,T2_limit=4440,
            used=self.used,completed=self.completed,failed=self.failed,
            total_started=self.total,total_completed=sum(self.completed.values()),
            total_failed=sum(self.failed.values()),cache_hits=self.cache_hits,
            internal_counts=self.internal_counts,internal_inclusive_s=self.internal_inclusive_s,
            internal_exclusive_s=self.internal_exclusive_s,certification_counts=self.certification_counts,
            oracle_s=self.oracle_s,scope='Only T-series calls; inclusive times overlap, exclusive do not')

    def _cache_hit(self,category,metadata,key_hash,source_call_id):
        self.check()
        if category not in LIMITS:raise ValueError('Invalid T cache category')
        rec=dict(event='cache_hit',category=category,metadata=metadata,
            cache_key_sha256=key_hash,source_call_id=source_call_id,clock_s=time.perf_counter())
        _append(self.cache_journal,rec);_append(self.journal,rec)
        self.cache_hits+=1;self.write_summary()
