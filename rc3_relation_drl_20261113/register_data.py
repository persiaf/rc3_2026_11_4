"""Freeze deterministic metadata schedules without evaluating any task."""
from collections import defaultdict
import random
from .common import ROOT,HOME,read,put,sha,digest

def main():
    joint=read(ROOT/'rc3_d5_revision_20261112/JOINT_DATA_INDEX.json')
    train={**joint['full'],**joint['local']}
    development=read(ROOT/'rc3_d5_revision_20261112/REVISION_PROTOCOL.json')['evaluation']['development']
    cases=['F012','F039','L030','L245','L170','L164','EMPTY_CLOSED','EMPTY_OPEN']
    engineering=dict(schema='relation-engineering-cases-v1',cases=cases,training_cases={k:train[k] for k in cases[:6]},
        boundary_cases={'EMPTY_CLOSED':{'source':'F012','task_ids':[],'endpoints':'closed_depot'},'EMPTY_OPEN':{'source':'L030','task_ids':[],'endpoints':'original_open'}},
        selection='Metadata only: maximum candidates full mixed and blocked; all three local sizes and both endpoints. No cost selection.')
    put(HOME/'ENGINEERING_CASES.json',engineering)
    groups=defaultdict(lambda:defaultdict(list))
    for tid,e in train.items():groups[(e['spec']['width'],e['spec']['height'])][e['physical_key']].append(tid)
    class Stream:
        def __init__(self,size,seed):
            self.rng=random.Random(seed);self.by=groups[size];self.parents=sorted(self.by);self.rng.shuffle(self.parents);self.pos=0;self.cursor={p:0 for p in self.parents}
            self.tasks={p:sorted(self.by[p]) for p in self.parents}
            for ids in self.tasks.values():self.rng.shuffle(ids)
        def next(self):
            p=self.parents[self.pos%len(self.parents)];self.pos+=1
            ids=self.tasks[p];i=self.cursor[p];self.cursor[p]+=1
            return ids[i%len(ids)]
    def schedule(count,full_each,local_each,seed):
        sizes=[(4,3),(6,6),(8,8)];streams={size:Stream(size,seed+i) for i,size in enumerate([(10,10),*sizes])}
        rows=[];local_cursor=0
        for u in range(count):
            ids=[streams[(10,10)].next() for _ in range(full_each)]
            for j in range(local_each):
                size=sizes[local_cursor%3];local_cursor+=1;ids.append(streams[size].next())
            rows.append(dict(update=u+1,tasks=ids))
        return rows
    init=schedule(512,4,4,2026111302)
    paired=schedule(2048,2,2,2026111402)
    put(HOME/'M2_DATA_SCHEDULE.json',dict(schema='relation-M2-data-schedule-v1',seed=2026111302,paired_stream_seed=2026111402,
        rule='Per-size shuffled physical-parent cycle, per-parent shuffled existing-label cycle; half full, local sizes round-robin; no replacement of failed cases',
        init=init,paired=paired,training_metadata=train,
        source={'path':'rc3_d5_revision_20261112/JOINT_DATA_INDEX.json','sha256':sha(ROOT/'rc3_d5_revision_20261112/JOINT_DATA_INDEX.json')}))
    put(HOME/'M2_EVALUATION_CASES.json',dict(schema='relation-development-v1',LOCAL66=development,FULL12=[train[f'F{i:03}'] for i in range(12)],
        FULL12_ids=[f'F{i:03}' for i in range(12)],full_selection='First twelve registered full-task IDs, metadata fixed before new model evaluation',
        interpretation='LOCAL66 exposed development; FULL12 exposed training-parent diagnostic. Separate metrics, never mixed parent means. No new-parent or heldout claim.'))
    print({'engineering_cases':len(cases),'init_steps':len(init),'paired_steps':len(paired),'local_dev':len(development),'full_dev':12})

if __name__=='__main__':main()
