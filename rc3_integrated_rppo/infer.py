"""One explicit user-consumer plan; separate from completed research requests."""
import argparse
import time
from .common import *
from .data import Inputs
from .eval import plan


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=11622026);parser.add_argument('--case',default='selection_000')
    parser.add_argument('--output',required=True);parser.add_argument('--execute-one',action='store_true');a=parser.parse_args()
    require(a.execute_one,'Explicit --execute-one required; this command makes one NEW consumer H request, not a benchmark replay')
    output=Path(a.output).resolve();require(output.is_relative_to(ROOT) and not output.exists(),'Output must be a new project path')
    configure();verify_sources();index=read(HOME/'ACTOR_INDEX.json');entry=index['selected'][str(a.seed)]['chosen']['RL']['entry']
    events,_=read_chain(OUT/'ledger/events.jsonl')
    class ReadLog:
        def __init__(self):self.events=events
        def __call__(self,row):raise ValueError('Consumer cannot alter scientific input registry')
    class Budget:
        def __init__(self):self.start=time.perf_counter()
        def check(self):require(time.perf_counter()-self.start<300,'Consumer300s limit')
        def remaining(self):self.check();return 300-(time.perf_counter()-self.start)
    clock=Budget();inputs=Inputs(ReadLog(),clock);cases=[c for family in ('selection','promotion','final_seen','final_unseen') for c in inputs.declared[family]]
    found=[c for c in cases if c['id']==a.case];require(len(found)==1,'Unknown declared case')
    s=found[0]['spec'];key=f"{s['map_seed']}_{s['task_seed']}_A{s['A_condition']}";require(key in inputs.index,'Consumer does not materialize sealed future inputs')
    receipt=output.with_name(output.name+'.REQUEST.json');write_new(receipt,dict(experiment='explicit_consumer_plan',source_experiment=EXP,
        new_complete_request_predebit=1,cap=1,wall_cap_s=300,seed=a.seed,case=a.case,actor=entry,utc=now()))
    try:
        r=plan(dict(seed=a.seed,role='RL'),found[0],inputs,entry,clock);write_new(output,r)
        write_new(output.with_name(output.name+'.COMPLETE.json'),dict(output_sha256=digest(output),elapsed_s=time.perf_counter()-clock.start,
            actual_H=r['counters']['full_H_calls'],new_optimizer=0,scientific_comparison=False))
    except BaseException as exc:
        write_new(output.with_name(output.name+'.FAILED.json'),dict(error=repr(exc),elapsed_s=time.perf_counter()-clock.start,refund=0));raise


if __name__=='__main__':main()
