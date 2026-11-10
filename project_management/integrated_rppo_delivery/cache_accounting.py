"""Reconstruct process-local cache counts from frozen schedules, no data solve."""
from collections import OrderedDict
from rc3_integrated_rppo.common import *


class LedgerModel:
    def __init__(self,refs,index):self.refs=refs;self.index=index;self.parents=OrderedDict();self.nav=OrderedDict();self.c={}
    def inc(self,k):self.c[k]=self.c.get(k,0)+1
    def touch(self,s,view=True,clear=False):
        if clear:self.parents.clear()
        if view:self.inc('target_view_checks')
        k=f"{s['map_seed']}_{s['task_seed']}_A{s['A_condition']}"
        if k in self.parents:self.inc('parent_memory_hits');self.parents.move_to_end(k);return
        phys=self.refs['geometry'][str(s['map_seed'])]['physical_key']
        if phys in self.nav:self.inc('navigation_only_hits');self.nav.move_to_end(phys)
        else:
            self.inc('saved_navigation_loads');self.nav[phys]=True
            if len(self.nav)>4:self.nav.popitem(last=False)
        if k not in self.index:
            if k in self.refs['bank']:self.inc('original_bank_loads')
            else:self.inc('new_task_draws');self.inc('proxy_FA_generations')
            self.index.add(k)
        self.parents[k]=True
        if len(self.parents)>4:self.parents.popitem(last=False)


def main():
    t=time.perf_counter();refs=read(HOME/'DATA_REFERENCES.json');data=read(HOME/'DATA_SPLITS.json')
    events,tail=read_chain(OUT/'ledger/events.jsonl');teachers=[read(ROOT/e['path'])['spec'] for e in refs['teachers']]
    index=set();r1=LedgerModel(refs,index);specs=[]
    for seed in SEEDS:
        for batch in data['training'][str(seed)]+data['warm'][str(seed)]:specs.extend(batch)
    specs += [c['spec'] for f in ('selection','promotion') for c in data[f]]
    unique={identity(s):s for s in specs};groups={}
    for s in unique.values():groups.setdefault((s['map_seed'],s['task_seed'],s['A_condition']),[]).append(s)
    for _,ss in sorted(groups.items()):r1.touch(ss[0],view=False)
    expected=read(HOME/'INPUT_FEASIBILITY.json')['counts']
    require(all(r1.c.get(k,0)==expected.get(k,0) for k in set(r1.c)|set(expected)),'Historical preflight accounting does not match recorded counters')
    for seed in SEEDS:
        for c in data['engineering']:r1.touch(c['spec'])
    for s in data['engineering_benchmark']:r1.touch(s)
    for i in data['SL_batches'][str(SEEDS[0])][0]:r1.touch(teachers[i])
    r2=LedgerModel(refs,index)
    for s in data['engineering_benchmark']:r2.touch(s)
    for i in data['SL_batches'][str(SEEDS[0])][0]:r2.touch(teachers[i])
    r3=LedgerModel(refs,index)
    for c in data['selection']:
        for _ in range(2):r3.touch(c['spec'],clear=True)
    for batch in data['warm'][str(SEEDS[0])]:
        for s in batch:r3.touch(s)
    for c in data['selection']:r3.touch(c['spec'],clear=True)
    completed={e['round']:e for e in events if e['event']=='round_complete' and e['variant']=='main' and e['seed']==SEEDS[0]}
    for r in range(1,257):
        for s in data['training'][str(SEEDS[0])][r-1]:r3.touch(s)
        if completed[r]['matched_SL']:
            for i in data['SL_batches'][str(SEEDS[0])][r-1]:r3.touch(teachers[i])
        if r in (64,128,256):
            for c in data['selection']:
                for _ in range(2):r3.touch(c['spec'],clear=True)
    observed=read(HOME/'RESOURCE_ACCOUNTING.json')['input_preparation']
    require(all(r3.c.get(k,0)==observed.get(k,0) for k in set(r3.c)|set(observed)),'Main-run accounting differs from recorded process-local counters')
    actual_index=read(OUT/'inputs/INDEX.json');require(index==set(actual_index),'Reconstructed unique input keys differ')
    total={k:sum(m.c.get(k,0) for m in (r1,r2,r3)) for k in set(r1.c)|set(r2.c)|set(r3.c)}
    record=dict(status='PASS_ACCOUNTING_RECONSTRUCTION',source_sha256=digest(Path(__file__)),scientific_source_sha256=digest(source_path()),
        method='Simulate the exact frozen loader4-entry parent/nav LRU and actual observed call schedule; no F/A,D,scene/model generation or solver execution',
        recorded_preflight_counts_matched=True,recorded_R03_counts_matched=True,unique_frozen_input_keys=len(index),
        process_R01_counts=r1.c,process_R02_counts=r2.c,process_R03_observed_and_reconstructed=r3.c,cumulative_source_loading_counts=total,
        historical_cache_totals='Explicit retrospective deterministic reconstruction, not falsely labeled live instrumentation',
        teacher_or_oracle_queries=0,additional_complete_requests=0,new_model_forward=0,new_H=0,new_navigation_solve=0,new_optimizer=0,
        elapsed_s=time.perf_counter()-t,utc=now())
    write_new(HOME/'CACHE_ACCOUNTING_RECONSTRUCTION.json',record);print(json.dumps(record,ensure_ascii=False))


if __name__=='__main__':main()
