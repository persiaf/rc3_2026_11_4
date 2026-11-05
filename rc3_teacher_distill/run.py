"""One preregistered supervised screen, with gate-controlled frozen route tests."""
import datetime
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from rc3_fast.scenes import make_complex_scene
from rc3_constructive.problem import Cache, resample_tasks
from rc3_nav_value.r1.policy import State, complete_gB
from rc3_nav_value.r1.oracle import Oracle, _append
from .candidate_menu import generate, rule_action
from .features import build
from .register import ROOT, OUT, write, read, digest, signature, verify
from .ledger import TLedger
from .rank_model import fit, save
from .evaluate_actions import evaluate


class Experiment:
    def __init__(self,p):
        self.p=p;self.t0=time.perf_counter();self.stage='T1'
        self.stats={k:0 for k in ('scene_generation','preparations','nav_cache_hits','hot_priming',
            'raw_gB_trajectories','raw_gB_decisions','label_uses','menus','feature_rows',
            'training_forward_batches','training_rows','optimizer_updates',
            'inference_forward_batches','inference_action_rows','raw_student_trajectories',
            'student_decisions','old_model_trajectories','old_model_decisions')}
        self.stats.update(T0_new_research_calls=0,retained_test_reads=0,T3_calls=0)
        self.ledger=TLedger(OUT/'run');self.oracle=Oracle(self.ledger)
        assert self.oracle.H_sha256==p['H_sha256']
        self.ledger.deadline=self.t0+p['budgets']['wall_T1_s']
        self.records=[];self.models={};self.seeds=read(OUT/'seeds.json')
        write(OUT/'run_started.json',dict(protocol_sha256=digest(OUT/'protocol.json'),
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage='T1',
            deadline_s=p['budgets']['wall_T1_s']))

    def check(self):self.ledger.check()

    def gb(self,item,B,prefix=(),action=None):
        self.check();self.stats['raw_gB_trajectories']+=1
        raw=complete_gB(item,B,prefix,action)
        self.stats['raw_gB_decisions']+=len(raw)-2-len(prefix)
        return raw

    def scene(self,split,case):
        self.check();s=self.seeds[split][case]
        if split=='T2' and not read(OUT/'T1_GATE.json')['passed']:
            raise RuntimeError('T2 denied by frozen T1 gate')
        scene=resample_tasks(make_complex_scene(10,10,s['map_seed'],s['mode']),s['task_seed'])
        self.stats['scene_generation']+=1
        assert scene['seed']==s['map_seed'] and scene['task_seed']==s['task_seed']
        path=OUT/'scenes'/f'{split}_{case:02}.json';write(path,scene)
        return scene,digest(path)

    def prepare(self,scene,cache):
        self.check();item=cache.prepare(scene)
        self.stats['preparations']+=1;self.stats['nav_cache_hits']+=int(item.cache_hit)
        return item

    def menu(self,item,B,prefix):
        self.check();self.stats['menus']+=1
        return generate(item,B,prefix,8)

    def features(self,item,B,prefix,menu):
        self.stats['feature_rows']+=len(menu)
        return build(item,B,prefix,menu)

    def label(self,item,B,sha,prefix,action,category,meta):
        tick=time.perf_counter();raw=self.gb(item,B,prefix,action)
        completion_s=time.perf_counter()-tick
        key=dict(scene=sha,prefix=list(prefix),action=action,reference=signature(B),H=self.p['H_sha256'])
        row=self.oracle.full(item,raw,category,dict(**meta,scene_sha256=sha,prefix=list(prefix),
            action=action,reference='gB-v1',reference_sha256=signature(B),completion_s=completion_s),cache_key=key)
        self.stats['label_uses']+=1
        return row

    def collect(self,split):
        for spec in self.seeds[split]:
            self.check();tick=time.perf_counter();case=spec['case']
            scene,sha=self.scene(split,case);item=self.prepare(scene,Cache(0))
            native=self.oracle.native(item,'fixed3','reference',dict(split=split,case=case,purpose='frozen_B'))
            if native['fallback']:raise RuntimeError('Native reference fallback disallowed')
            B=native['route'][1:-1]
            raw=self.gb(item,B);s=State(item);states=[];seen=set()
            for threshold in self.p['data']['prefix_thresholds']:
                while not s.complete and 1-s.remaining.mean()+1e-12<threshold:
                    s.step(next(a for a in B if s.legal_mask()[a-1]))
                if s.complete or tuple(s.prefix) in seen:continue
                seen.add(tuple(s.prefix));states.append((threshold,list(s.prefix)))
            for threshold,prefix in states:
                menu=self.menu(item,B,prefix)
                labels=[self.label(item,B,sha,prefix,a,split,dict(split=split,case=case,threshold=threshold)) for a in menu]
                rec=dict(split=split,case=case,partition=spec['partition'],mode=spec['mode'],
                    scene_sha256=sha,B=B,reference_sha256=signature(B),prefix=prefix,threshold=threshold,
                    remaining=int(State(item,prefix).remaining.sum()),menu=menu,rule_action=rule_action(item,menu,prefix),
                    legal_action_count=int(State(item,prefix).legal_mask().sum()),
                    features=self.features(item,B,prefix,menu).tolist(),Y=[r['J'] for r in labels],
                    Yref=labels[0]['J'],call_ids=[r['call_id'] for r in labels],
                    raw_routes=[r['raw_route'] for r in labels],final_routes=[r['route'] for r in labels])
                _append(OUT/'labels.jsonl',rec);self.records.append(rec)
            _append(OUT/'scene_resources.jsonl',dict(split=split,case=case,states=len(states),elapsed_s=time.perf_counter()-tick))
            if case%8==7 or case==len(self.seeds[split])-1:
                self.resources();print(json.dumps(dict(stage='labels_'+split,cases=case+1,calls=self.ledger.total,elapsed_s=time.perf_counter()-self.t0)),flush=True)

    def resources(self):
        write(OUT/'resources.json',dict(stage=self.stage,stats=self.stats,ledger=self.ledger.snapshot(),
            elapsed_wall_s=time.perf_counter()-self.t0,
            deadline_remaining_s=max(0.,self.ledger.deadline-time.perf_counter()),
            scope='Separate T series only; T0 byte/record audit has zero new research calls; labels, exact hits, actual evaluations and training counted separately'))

    def run(self):
        self.collect('train');self.collect('development');verify()
        write(OUT/'before_training.json',dict(protocol_sha256=digest(OUT/'protocol.json'),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),optimizer_updates=0,
            labels=len(self.records),terminal_calls=self.ledger.total,elapsed_s=time.perf_counter()-self.t0))
        (OUT/'weights').mkdir(exist_ok=True)
        for kind in ('linear','neural'):
            self.check();bundle=fit(self.records,kind,self.p['models']['stochastic_seeds'],self.check,self.stats)
            self.models[kind]=bundle
            path=OUT/'weights'/f'{kind}.pt';save(bundle,path,digest(OUT/'protocol.json'))
            write(OUT/f'{kind}_training.json',{k:v for k,v in bundle.items() if k not in ('mean','std','networks')})
            write(OUT/f'{kind}_actions.json',evaluate(self.records,bundle,self.stats))
            self.resources()
        neural=read(OUT/'neural_actions.json');linear=read(OUT/'linear_actions.json')
        self.check()  # Last prediction/statistics/IO still belong to the T1 deadline.
        gate=dict(stage='T1',passed=neural['passed'],checks=neural['checks'],
            primary='neural ensemble',linear_passed=linear['passed'],
            rule='All preregistered criteria; no endpoint/model/threshold selection',
            frozen_model_sha256={k:digest(OUT/'weights'/f'{k}.pt') for k in self.models},
            protocol_sha256=digest(OUT/'protocol.json'))
        write(OUT/'T1_GATE.json',gate)
        if not gate['passed']:
            write(OUT/'STATUS.json',dict(T0='complete',T1='failed_stop',T2='not_started_gate_failed',
                T3='not_authorized',R2='not_started',R3='not_started',retained60='sealed',
                default_methods=['Fixed3','Detour'],new_training_extension=False))
            write(OUT/'T2_NOT_RUN.json',dict(reason='Pre-registered T1 neural primary gate failed',
                failed_checks=[k for k,v in gate['checks'].items() if not v],T2_scene_generation=0,
                T2_planner_calls=0,T2_teacher_queries=0,T3_authorized=False))
        else:
            verify();self.check();self.stats['T1_elapsed_s']=time.perf_counter()-self.t0
            self.stage='T2';self.ledger.deadline=time.perf_counter()+self.p['budgets']['wall_T2_s']
            write(OUT/'STATUS.json',dict(T0='complete',T1='passed',T2='started',T3='not_authorized',retained60='sealed'))
            from .evaluate_routes import routes
            routes(self)
        self.resources();verify();self.check()


def main():
    p=verify()
    if (OUT/'run_started.json').exists():raise FileExistsError('One registered run only; no replay/replacement')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    x=None
    try:
        with threadpool_limits(limits=1):
            x=Experiment(p);x.run()
    except Exception as exc:
        if x is not None:
            x.resources()
        stage=x.stage if x is not None else 'T1'
        write(OUT/'STATUS.json',dict(T0='complete',stage=stage,status='inconclusive_stop',
            T1='passed' if stage=='T2' else 'inconclusive_stop',
            T2='inconclusive_stop' if stage=='T2' else 'not_started',
            error=type(exc).__name__,message=str(exc),T3='not_authorized',retained60='sealed'))
        write(OUT/'failure.json',dict(error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
