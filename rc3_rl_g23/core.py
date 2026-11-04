"""G23 versioned caps, globally charged calls and aligned rule candidate."""
import copy,hashlib,json,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
from rc3_rl_g22.common import jsonable,digest,sha,read,write
ROOT=Path(__file__).resolve().parent;WS=ROOT.parent

class LimitReached(RuntimeError):pass
class Ledger:
    def __init__(self,limits):
        self.limits=limits;self.counts={k:0 for k in limits};self.phase='setup';self.started=time.perf_counter()
    def use(self,key,n=1,**detail):
        if time.perf_counter()-self.started>5400:raise LimitReached('Wall-time cap 90min')
        if key=='nonstop' and self.phase.startswith('seed') and self.counts[key]+n-getattr(self,'seed_base',0)>1536:
            raise LimitReached('Per-seed train repair cap')
        if self.counts[key]+n>self.limits[key]:raise LimitReached('Global cap '+key)
        self.counts[key]+=n
        with (ROOT/'outputs/resource.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(jsonable(dict(kind=key,n=n,total=self.counts[key],phase=self.phase,**detail)),allow_nan=False)+'\n')
        write(ROOT/'outputs/resources.json',dict(counts=self.counts,limits=self.limits,elapsed_s=time.perf_counter()-self.started))

def register():
    path=ROOT/'outputs/protocol.json'
    if path.exists():raise FileExistsError('Immutable new protocol already exists')
    old=read(WS/'rc3_rl_g2/outputs/protocol.json');seen={c['seed'] for group in old['splits'].values() for c in group}
    rng=np.random.default_rng(20261104023);splits={}
    for name,n in [('train',12),('calibration',3),('selection',3),('test',6)]:
        group=[]
        for i in range(n):
            seed=int(rng.integers(1,2**31-1))
            while seed in seen:seed=int(rng.integers(1,2**31-1))
            seen.add(seed);group.append(dict(seed=seed,nx=10,ny=10,mode=('regular','mixed','blocked')[i%3],case_id=i,split=name))
        splits[name]=group
    p=dict(schema='g23-rule-value-iteration-v1',date='2026-11-04',seeds=[2301,2302,2303],splits=splits,
        config=old['config'],complete_soft_s=.8,repetitions=2,wall_limit_s=5400,alignment_cases=list(range(12)),
        limits=dict(nonstop=8192,menus=20000,stops=4000,finalizations=2000,cold_plans=300,optimizer_updates=3840),
        training=dict(rounds=2,states_per_round=8,first_actions=3,updates_per_fit=256,lr=.001,batch=16,threshold=0.,train_calls_per_seed=1536),
        value='Full finite-horizon MC under a separately frozen continuation; gamma1; STOP baseline paid pending menu plus final correction',
        controls=['same_interface_rule','one_step_repeated','frozen_init','matched_static_rule_value','multi_policy_iteration','original_fixed3','original_detour16'],
        main_gate=dict(mean_gain_percent=.3,paired_ci_lower_above_zero=True,comparators=['same_interface_rule','one_step_repeated','frozen_init','matched_static_rule_value','original_detour16']),
        stopping='If aligned capped trial fails the preregistered quality gate, stop treating local window choice+STOP as the DRL core; no extensions',
        initialization='Rule policy for round0; MC0 checkpoint forms pi1; round1 follows pi1 and also evaluates static rule on identical first transitions',
        matched_training='one-step and multi/static use same first transitions/capacity, 256+256 updates; frozen_init remains first256 checkpoint',
        counter_scope='All original reference repairs, paired aligned repairs, branch rollouts, training and final comparisons globally counted',
        engineering='No new preprocessing/proposal cache; common interfaces for all shared methods; no claimed cache speedup',
        uncertainty='Scene-level paired bootstrap after averaging repetitions and training seeds; only six independent new test scenes, exploratory external validity',
        calibration='Threshold0 predetermined; calibration/selection seeds reserved but not queried or used for tuning',
        training_deadline='Training MC finishes under finite h/work, without wall-time cuts; deployment additionally has complete .8s soft cap',
        work_proxy='G2 context+candidate units, plus original rule scan proxy windows*(1+M); original dense DP evaluated entries counted; not elapsed time',
        arm_graph='Separate left/right reachability edge subgraphs encoded by two identical graph encoders plus shared per-action numeric head; all controls same capacity',
        cross_scale='Only after main quality gate passes; otherwise do not expand',request_sha256=sha(ROOT/'inputs/user_request.txt'),
        source_sha256={f.name:sha(f) for f in ROOT.glob('*.py')})
    from rc3_rl_g22.register import frozen_snapshot
    frozen=frozen_snapshot();frozen['g22_manifest']=read(WS/'rc3_rl_g22/MANIFEST.sha256.json')
    frozen['g22_archive_sha']=sha(WS/'rc3_rl_g22_delivery.zip')
    for name,x in frozen['g22_manifest']['files'].items():
        if sha(WS/name)!=x['sha256']:raise ValueError('G22 changed')
    write(ROOT/'outputs/freeze_before.json',frozen);write(path,p);(path.with_suffix('.sha256')).write_text(sha(path))
    return p

from rc3_rl_g2.env_g2 import SearchEnvG2
from rc3_rl_g2.action_memory import build_actions
from rc3_rl_research.action_builder import local_context
from rc3_rl_research.types import Action,ProposalResult,SearchConfig
from rc3_v3.environment import adaptive_windows
from rc3_improve.repair import window_features
from rc3_improve.environment import canonical_route
from .repair_counted import _repair

class Env(SearchEnvG2):
    def __init__(self,adapter,ledger,seed=0,config=None):
        self.ledger=ledger;self.rule_failed=set();self.rule_info=None
        super().__init__(adapter,SearchConfig() if config is None else config,seed,proposer=self.primitive,action_builder=self.builder)
    def primitive(self,problem,route,action,*,config):
        self.ledger.use('nonstop',action=action.to_dict())
        if action.kind=='replace':
            from rc3_rl_research.repair_proposal import propose
            return propose(problem,route,action,config=config)
        counters={};candidate=_repair(problem,route,dict(start=action.start,end=action.end),counters=counters)
        candidate=canonical_route(problem,candidate)
        signature=self.original_signatures.get((action.start,action.end))
        if signature is not None:
            from rc3_fast.solver import route_objective
            if route_objective(problem,candidate)<route_objective(problem,route)-1e-8:self.rule_failed.discard(signature)
            else:self.rule_failed.add(signature)
        return ProposalResult(candidate,True,candidate!=route,int(sum(counters.values())),counters=counters,
            diagnostics=dict(primitive='original-dense-g23'))
    def builder(self,problem,route,**kw):
        self.ledger.use('menus')
        menu=build_actions(problem,route,attempt_memory=self.attempt_memory,**kw)
        windows,signatures=adaptive_windows(problem,route);features=window_features(problem,route,windows)
        self.original_signatures={(w['start'],w['end']):s for w,s in zip(windows,signatures)}
        eligible=[i for i,s in enumerate(signatures) if s not in self.rule_failed]
        chosen=max(eligible,key=lambda i:(float(features[i,8]),-i)) if eligible else None
        self.rule_info=None if chosen is None else dict(window=windows[chosen],signature=signatures[chosen],features=features[chosen].tolist())
        if chosen is not None:
            window=windows[chosen];a=Action('improve',window['start'],window['end'],key=f"improve:{window['start']}:{window['end']}:-")
            existing=next((i for i,x in enumerate(menu.actions) if x.kind==a.kind and x.start==a.start and x.end==a.end),None)
            if existing is None:
                c=local_context(problem,route,a.start,a.end);nodes=np.asarray(route)
                retained=c['retained_counts'];removed=c['removed'];cover=c['cover'];pool=c['pool'];required=c['required']
                travel=float(problem.nav['costs'][nodes[a.start-1:a.end],nodes[a.start:a.end+1]].sum());direct=float(problem.nav['costs'][nodes[a.start-1],nodes[a.end]])
                counts=c['counts'];covered=cover[:,removed-1].any(1);overlap=covered&(retained>0);extra=cover[:,pool-1].any(1)&(retained>0)
                f=[1.,0.,0.,(a.start-1)/max(len(route)-2,1),a.end-a.start,len(required),len(pool),travel/2.2,direct/2.2,(travel-direct)/2.2,(travel-direct)/max(travel,1e-8),
                    np.log1p(len(route)-2),np.log1p(len(cover)),np.log1p(cover.shape[1]),float(counts.mean()),float(counts.max()),int(covered.sum()),float(overlap.sum()/max(covered.sum(),1)),int(extra.sum()),kw['remaining_work']/self.config.work_budget,kw['steps_remaining']/self.config.horizon,len(self.history)]
                detail=dict(required_tasks=required.tolist(),pool_nodes=pool.tolist(),retained_nodes=c['retained'].tolist(),retained_coverage_counts=retained.tolist())
                if len(menu.actions)==1:
                    menu.actions.insert(0,a);menu.features.insert(0,f);menu.mask.insert(0,True);menu.diagnostics['action_details'].insert(0,detail);existing=0
                else:
                    existing=len(menu.actions)-2
                    menu.actions[existing]=a;menu.features[existing]=f;menu.diagnostics['action_details'][existing]=detail
            order=[existing]+[i for i in range(len(menu.actions)) if i!=existing]
            menu.actions=[menu.actions[i] for i in order];menu.features=[menu.features[i] for i in order];menu.mask=[menu.mask[i] for i in order]
            menu.diagnostics['action_details']=[menu.diagnostics['action_details'][i] for i in order]
        menu.work_units+=int(len(windows)*(1+problem.F.shape[1]))
        menu.mask=[bool(x and (kw['remaining_work']>menu.work_units)) if a.kind!='stop' else True for x,a in zip(menu.mask,menu.actions)]
        menu.context_key=digest(dict(legacy=menu.context_key,rule_failed=sorted(map(repr,self.rule_failed)),version='g23'))
        menu.action_context_keys=[digest(dict(context=menu.context_key,action=a.to_dict())) for a in menu.actions]
        menu.diagnostics['rule_available']=chosen is not None
        return menu
    def step(self,action):
        self.observe()
        a=self._menu.actions[action]
        if a.kind=='stop':self.ledger.use('stops')
        return super().step(action)
    def _finish(self,reason):
        if not self.terminated:self.ledger.use('finalizations')
        return super()._finish(reason)
    def fork(self):
        ledger=self.ledger;self.ledger=None
        try:child=super().fork()
        finally:self.ledger=ledger
        child.ledger=ledger;child._raw_action_builder=child.builder;child._raw_proposer=child.primitive
        return child
    def rule(self):
        obs=self.observe()
        return 0 if self.rule_info is not None and obs['mask'][0] else next(i for i,a in enumerate(obs['actions']) if a['kind']=='stop')
