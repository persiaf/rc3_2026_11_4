"""Registered, single-seed R1 screen. No automatic continuation or seed replay."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

# Must precede importing Torch/CUDA.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from rc3_fast.scenes import make_complex_scene
from rc3_constructive.problem import Cache, resample_tasks
from rc3_constructive import baselines
from rc3_constructive.evaluate import load_policy, greedy_routes
from .value_model import Model
from .policy import State, NeuralPolicy, prepare_data, complete_gB
from .oracle import Ledger, Oracle

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
BASE='129ba6e916ed5a44ecb92998553c082b4510ecd1'
SEED=11412026
MODES=('regular','mixed','blocked')
GROUPS={'A':'base','B':'nav','C':'base','D':'nav'}


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def signature(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def append(path,value):
    with Path(path).open('a',encoding='utf-8') as f:f.write(json.dumps(value,allow_nan=False)+'\n');f.flush()
def tensor_hash(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def register():
    if (OUT/'protocol.json').exists():raise FileExistsError('No protocol replacement or implicit rerun')
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==BASE
    audit=read(OUT/'seed_audit.json')
    assert audit['verdict']=='PASS_NO_RECORDED_PRIOR_USE'
    seeds={split:[dict(case=i,map_seed=ms+i,task_seed=ts+i,mode=MODES[i%3]) for i in range(32)]
           for split,ms,ts in [('train',8600000000,8700000000),('dev',8800000000,8900000000)]}
    write(OUT/'seeds.json',seeds)
    source_names=['rc3_nav_value/r1/'+n for n in ['run.py','value_model.py','policy.py','oracle.py','analyze.py']]
    source_names+=['rc3_constructive/problem.py','rc3_constructive/env.py','rc3_constructive/baselines.py',
        'rc3_constructive/model.py','rc3_constructive/evaluate.py','rc3_fast/solver.py','rc3_fast/planner.py',
        'rc3_fast/scenes.py','rc3_baseline/baseline.py','rc3_baseline/validator.py','rc3_v3/environment.py','rc3_v4/runtime.py']
    H_source={n:digest(ROOT/n) for n in ['rc3_constructive/baselines.py','rc3_fast/solver.py']}
    p=dict(schema='rc3-nav-value-r1-v1',date='2026-11-05',base_commit=BASE,training_seed=SEED,
        scope='One matched development screen; no R2/R3, no retained60 reads, no historical evidence changes',
        seeds_sha256=digest(OUT/'seeds.json'),seed_audit_sha256=digest(OUT/'seed_audit.json'),
        seed_inventory_sha256=digest(OUT/'seed_audit_inventory.jsonl'),seed_boundary_sha256=digest(OUT/'seed_audit_boundary_sanity.json'),
        source_sha256={n:digest(ROOT/n) for n in source_names},H_source_sha256=H_source,H_sha256=signature(H_source),
        old128_sha256=digest(ROOT/'rc3_constructive/outputs/checkpoints/update0128.pt'),
        H='Unchanged baselines.postprocess: twice normalize+two_opt60 then normalize; joint_search2; normalize; export and independent scalar validation',
        objective='J=L+0.75K, closed route; exact positive coverage/current-to-candidate/home reachability action mask',
        target='Delta=(Yref-Ya)/max(Yref,1e-8), Y=J(H(prefix+action+Complete_reference)); positive preferred; model argmax all legal actions',
        supersedes='R1 revised design J-only Huber is superseded BEFORE DATA by user-requested normalized advantage regression+ranking. All four groups receive complete ordered prefix GRU.',
        architecture=dict(hidden=64,layers=2,neighbor_count=20,near=16,long_range=4,distance_scale=10.,score_bound=.2,
            identical_parameters=True,all_groups_full_prefix=True,base_relation='Euclidean candidate distance',nav_relation='true candidate navigation D',
            common='true current/home D, exact masks, prefix edge costs, coverage, geometry',
            topology='geometric nearest16 excluding self plus4 quantile ranks floor((2q+1)*R/8) in remaining Euclidean-distance order; distance ties originalID; topology shared and never selected by D'),
        groups={'A':'base fixed g0 with matched new states','B':'nav fixed g0 with matched new states',
                'C':'base approximate policy iteration','D':'nav approximate policy iteration'},
        limits=dict(init=4096,round=8192,audit=1024,validation=3072,total=16384),
        init='32 nativeFixed3 training reference calls+508 checkpoint menus*up to8; first28 scenes16 thresholds, last4 scenes15. Duplicated crossed threshold prefixes skipped, never replace seeds/states.',
        query='reference first (include both group references in paired states); gain/(distance+.75) descending fill4, current distance fill6, independent permutation fill8. Numeric ties original ID. Only actually queried legal actions enter loss.',
        exploration='9000000000+stable state menu index; below9000010000',
        warm=dict(updates=1024,batch_size=8,shared_labels=True,matched_initial_tensors=True,shared_within_representation=True),
        rounds=dict(count=2,updates_each=512,states_each=128,menu_each=8,
            first='4 coverage thresholds0,.25,.5,.75 from actual g0 trajectories; same pair states and labels',
            second='per scene 2 thresholds0,.5 from fixed g0 plus2 thresholds.25,.75 from current API reference actual trajectories; same pool and menu for fixed/API pair; no duplicate state replacement',
            optimizer='Adam1e-4 at warm; fresh identical Adam1e-4 at round1; retain corresponding optimizer state into round2; clip norm1; no dropout',
            promotion='after round1 only: API candidate vs its current g0 on ALL new dev32: mean relative gain>=.5%, paired bootstrap95 lower>0, all feasible. Otherwise keep g0. No best-checkpoint selection after endpoints.',
            invariant='before promotion A1=C1, B1=D1 exact parameter tensors and dev greedy raw/final routes/J; if reference not promoted also require corresponding round2 identity',
            actual_optimizer_steps=6144,logical_steps_per_configuration=2048,logical_sum_with_inherited_warm=8192),
        loss=dict(regression='SmoothL1 beta.02 on Delta queried actions only',ranking='all queried pairs weighted by absolute true Delta gap; softplus(-signGap*predGap/.02); skip ties<=1e-8',rank_weight=.1,
            unqueried='unknown; no zero fill, no loss and no pseudo label',replay='warm labels used only for warm1024; each round uses only its frozen-reference current labels, eight same-scene states; fixed/API matched scene/minibatch indices'),
        audit='after each round on actual training state pool: evaluate learner all-menu argmax under corresponding frozen reference; if in queried8 reuse label; otherwise1 billed fullH. Never fit/promote/select using these audit outcomes. This tests action generalization on seen states, not unseen-state generalization.',
        validation=dict(initial_methods=['warm_base','warm_nav','old128','prior','gBH','fixed3','detour16'],
            round_methods=['A','B','C','D'],timing_methods=['A','B','C','D','warm_nav','old128','prior','fixed3','detour16'],
            cases=32,repetitions=3,cache_modes=['cold','hot'],reserved_other_calibration=128,
            independence='new development maps independent of train/R0, but SAME dev32 used for promotion and final screen => exploratory, not untouched test confirmation',
            baselines='Fixed3/Detour16 retain full native pipeline; gBH teacher uses commonH; prior/old/neural use commonH. Do not claim all native baselines have an identical pipeline.',
            timing='random deterministic method/cache order; scene generation/JSON load excluded; deploy total=timed C/F/D preparation+feature/static encode/decision+oracle compute (H/native, export, independent scalar checks). Journal/fsync/summary IO excluded by oracle compute stopwatch; observed full request wall and bookkeeping residual separately saved. No inter-request neural embedding reuse; model load outside and reported; GPU synchronize boundaries.',
            top_calls_upper=224+256+1728+128,unknown_audit_upper=1024),
        gate=dict(main='ALL32 valid; D2vsB2 mean>=.5% and paired95lower>0; D2vswarm_nav95lower>0; D2vsold128+H95lower>0',
            strong_deployment='also Detour16 relative degradation one-sided95upper<=.5%; complete time median ratio<=.7 and p95ratio<=1 in both navcache modes; not quality/time dominated byFixed3',
            inference='no online Complete lookahead; one greedy full-menu model construction+H; no rulefallback',
            fail='close this configuration; no further updates/seeds/rounds; retain R0 opportunity result; R2 never automatically starts'),
        bootstrap=dict(resamples=20000,seed=SEED,unit='scene',relative_gain='100*(Jbaseline-Jcandidate)/Jbaseline',CI='paired percentile95; exploratory; no multiple-comparison confirmatory claim'),
        wall=dict(calibration='initial labels;3 untrained raw greedy NN probes per representation;16 forward/backward probes per representation, NO optimizer updates; all tracked',
            freeze_before_optimizer=True,formula='min(7200,max(1800,ceil(4*(6144*max_probe_step_s+12288*max_initial_H_s+1728*max_prepare_s+4096*max_raw_completion_s)+120)))',
            scope='starts immediately before first optimizer step; includes all following training, raw trajectories, oracle calls, audits, validation and timing. Initial collection/calibration separately recorded. Deadline checked between calls/steps, cannot extend.',
            incomplete='inconclusive_stop; preserve failed calls/checkpoints; no replacement run or early-checkpoint substitution'),
        counters='actual terminal native/H calls vs label uses/cachehits; internal solver calls, NN staticencodes/score batches/rows, raw trajectories/decisions, backward probes and actual/logical optimizer steps separately; topcalls are not equal FLOPs')
    write(OUT/'protocol.json',p)
    write(OUT/'registration_receipt.json',dict(protocol_sha256=digest(OUT/'protocol.json'),base_commit=BASE,created_unix=time.time(),research_calls_before_registration=0,optimizer_updates=0))
    print(json.dumps(dict(registered=True,protocol_sha256=digest(OUT/'protocol.json'))),flush=True)


def verify_registration():
    p=read(OUT/'protocol.json');r=read(OUT/'registration_receipt.json')
    assert digest(OUT/'protocol.json')==r['protocol_sha256']
    for n,h in p['source_sha256'].items():assert digest(ROOT/n)==h,('source_changed',n)
    for n,k in [('seeds.json','seeds_sha256'),('seed_audit.json','seed_audit_sha256'),('seed_audit_inventory.jsonl','seed_inventory_sha256'),('seed_audit_boundary_sanity.json','seed_boundary_sha256')]:assert digest(OUT/n)==p[k]
    assert digest(ROOT/'rc3_constructive/outputs/checkpoints/update0128.pt')==p['old128_sha256']
    return p


def scene_load(split,case,cache):
    spec=read(OUT/'seeds.json')[split][case];path=OUT/'scenes'/f'{split}_{case:02}.json'
    if path.exists():scene=read(path)
    else:
        scene=resample_tasks(make_complex_scene(10,10,spec['map_seed'],spec['mode']),spec['task_seed']);write(path,scene)
    assert scene['seed']==spec['map_seed'] and scene['task_seed']==spec['task_seed']
    tick=time.perf_counter();item=cache.prepare(scene)
    return scene,item,time.perf_counter()-tick,digest(path)


def checkpoints(item,raw,thresholds):
    """Unique actual prefixes; crossing several thresholds consumes them all."""
    answer=[];state=State(item);pending=list(thresholds)
    for node in raw[1:-1]:
        fraction=1-state.remaining.sum()/max(1,len(item.C))
        crossed=[t for t in pending if fraction+1e-12>=t]
        if crossed and not state.complete:
            answer.append(list(state.prefix));pending=[t for t in pending if t not in crossed]
        state.step(int(node))
    assert state.complete
    return answer


def menu(item,prefix,refs,index):
    s=State(item,prefix);legal=np.flatnonzero(s.legal_mask())+1
    assert len(legal)>0
    selected=[]
    def add(seq,limit):
        for a in seq:
            if a in legal and int(a) not in selected:selected.append(int(a))
            if len(selected)>=limit:break
    add(refs,8)
    gains=item.C[s.remaining].sum(0)
    rule=gains/(item.D[s.current,1:]+item.base.penalty)
    add(sorted(legal,key=lambda a:(-rule[a-1],a)),4)
    add(sorted(legal,key=lambda a:(item.D[s.current,a],a)),6)
    add(np.random.default_rng(9000000000+index).permutation(legal),8)
    assert len(selected)==min(8,len(legal))
    return selected


class Experiment:
    def __init__(self,p):
        self.p=p;self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.stats={'optimizer_updates':0,'backward_probe_steps':0,'raw_gB_trajectories':0,'label_uses':0,'scenes':0,'prepare_calls':0}
        self.ledger=Ledger(OUT);self.oracle=Oracle(self.ledger)
        self.train=[];self.dev=[];self.prepare_times=[];self.Htimes=[]
        self.models={};self.warm={};self.refs={};self.opts={};self.initial=[];self.round_data={};self.promoted={}
        self.tensor_data={};self.menu_index=0;self.t0=time.perf_counter()

    def check(self):self.ledger.check()
    def sync(self):
        if self.device=='cuda':torch.cuda.synchronize()

    def raw(self,policy,prefix=(),actions=None):
        self.check()
        if actions is None:return policy.trajectory(prefix)
        return policy.complete_many(prefix,actions)

    def gbraw(self,item,B,prefix=(),action=None):
        self.check();raw=complete_gB(item,B,prefix,action)
        self.stats['raw_gB_trajectories']+=1
        self.stats['raw_gB_decisions']=self.stats.get('raw_gB_decisions',0)+len(raw)-2-len(prefix)
        return raw

    def label(self,item,sha,prefix,actions,ref,category,tag,case):
        tick=time.perf_counter()
        if len(ref)==2 and isinstance(ref[0],str) and ref[0]=='gB':
            B=ref[1];refhash=signature(B);refname='gB';raws=[self.gbraw(item,B,prefix,a) for a in actions]
        else:
            policy,refhash,refname=ref;raws=self.raw(policy,prefix,actions)
        self.sync();completion_batch_s=time.perf_counter()-tick;records=[]
        for a,raw in zip(actions,raws):
            key=signature(dict(scene=sha,prefix=prefix,action=a,reference=refhash,H=self.p['H_sha256']))
            meta=dict(case=case,tag=tag,scene_sha256=sha,prefix=prefix,action=a,reference=refname,reference_tensor_sha256=refhash,
                      completion_batch_s=completion_batch_s,completion_batch_size=len(actions),completion_time_semantics='shared batch wall time, not per-action time; do not sum once per result')
            row=self.oracle.full(item,raw,category,meta,cache_key=key)
            records.append(row);self.stats['label_uses']+=1
        ys=[float(r['J']) for r in records];yr=ys[0]
        return dict(case=case,prefix=prefix,actions=actions,Y=ys,Yref=yr,
                    target=[(yr-y)/max(yr,1e-8) for y in ys],reference=refname,reference_hash=refhash,scene_sha256=sha)

    def collect_initial(self):
        cache=Cache(32)
        for i in range(32):
            scene,item,prep,sha=scene_load('train',i,cache);self.train.append((scene,item,sha));self.prepare_times.append(prep)
            self.stats['scenes']+=1;self.stats['prepare_calls']+=1
            native=self.oracle.native(item,'fixed3','init',dict(case=i,tag='train_referenceB'))
            B=native.get('finalroute',native.get('route'));assert B and B[0]==B[-1]==0
            raw=self.gbraw(item,B)
            count=16 if i<28 else 15
            prefixes=checkpoints(item,raw,np.arange(count)/count)
            for prefix in prefixes:
                refraw=self.gbraw(item,B,prefix)
                refaction=refraw[len(prefix)+1]
                actions=menu(item,prefix,[refaction],self.menu_index);self.menu_index+=1
                rec=self.label(item,sha,prefix,actions,('gB',B),'init','warm',i);self.initial.append(rec)
                append(OUT/'labels_initial.jsonl',rec)
            print(json.dumps(dict(stage='initial_labels',case=i,states=len(prefixes),calls=self.ledger.snapshot())),flush=True)
        terminal=OUT/'terminal.jsonl'
        self.Htimes=[json.loads(s)['oracle_s'] for s in terminal.read_text().splitlines() if json.loads(s).get('kind')=='full_H']
        assert self.Htimes,'Calibration requires measured initial full-H times'
        write(OUT/'initial_collection.json',dict(states=len(self.initial),menus_upper=508,label_uses=sum(len(r['actions']) for r in self.initial),elapsed_s=time.perf_counter()-self.t0))

    def data(self,case,rep):
        key=(case,rep)
        if key not in self.tensor_data:self.tensor_data[key]=prepare_data(self.train[case][1],rep,self.device)
        return self.tensor_data[key]

    def batch(self,records,rep):
        case=records[0]['case'];assert all(r['case']==case for r in records)
        item=self.train[case][1];data={k:v.expand((len(records),)+v.shape[1:]) for k,v in self.data(case,rep).items()}
        states=[State(item,r['prefix']) for r in records];T=max(len(r['prefix']) for r in records)
        nodes=np.zeros((len(records),T),np.int64);actions=np.zeros((len(records),8),np.int64)
        target=np.zeros((len(records),8),np.float32);mask=np.zeros((len(records),8),bool)
        for b,r in enumerate(records):
            nodes[b,:len(r['prefix'])]=r['prefix'];k=len(r['actions']);actions[b,:k]=np.array(r['actions'])-1
            target[b,:k]=r['target'];mask[b,:k]=True
        values=dict(remaining=np.stack([s.remaining for s in states]),current=np.array([s.current for s in states]),
            prefix_nodes=nodes,prefix_lengths=np.array([len(r['prefix']) for r in records]),prefix_rawL=np.array([s.rawL for s in states],np.float32),
            actions=actions,target=target,mask=mask)
        return data,{k:torch.as_tensor(v,device=self.device) for k,v in values.items()}

    def loss(self,model,data,b):
        scores=model.forward_state(data,b['remaining'],b['current'],b['prefix_nodes'],b['prefix_lengths'],b['prefix_rawL'])
        self.stats['training_or_probe_forward_batches']=self.stats.get('training_or_probe_forward_batches',0)+1
        self.stats['training_or_probe_forward_rows']=self.stats.get('training_or_probe_forward_rows',0)+len(b['remaining'])
        predicted=scores.gather(1,b['actions']);mask=b['mask'];target=b['target']
        regression=torch.nn.functional.smooth_l1_loss(predicted[mask],target[mask],beta=.02)
        gap=target[:,:,None]-target[:,None,:];pgap=predicted[:,:,None]-predicted[:,None,:]
        pair=mask[:,:,None]&mask[:,None,:]&(gap.abs()>1e-8)
        weighted=gap.abs()*pair
        ranking=(torch.nn.functional.softplus(-gap.sign()*pgap/.02)*weighted).sum()/weighted.sum().clamp_min(1e-8)
        loss=regression+.1*ranking
        return loss,dict(loss=float(loss.detach()),regression=float(regression.detach()),ranking=float(ranking.detach()),mse=float(((predicted[mask]-target[mask])**2).mean().detach()))

    def sample_schedule(self,warm,current,steps,seed):
        rng=np.random.default_rng(seed);wa={i:[r for r in warm if r['case']==i] for i in range(32)}
        cu={i:[r for r in current if r['case']==i] for i in range(32)}
        valid=[i for i in range(32) if wa[i] and (not current or cu[i])]
        assert len(valid)==32, 'Do not drop training scenes silently'
        for _ in range(steps):
            i=int(rng.choice(valid))
            if current:yield [cu[i][int(j)] for j in rng.integers(len(cu[i]),size=8)]
            else:yield [wa[i][int(j)] for j in rng.integers(len(wa[i]),size=8)]

    def save_model(self,name,model,updates):
        path=OUT/'checkpoints'/f'{name}.pt';path.parent.mkdir(exist_ok=True)
        torch.save(dict(state=model.state_dict(),model_config=model.architecture_config(),representation='base' if name.startswith(('A','C','warm_base')) else 'nav',
            tensor_sha256=tensor_hash(model),optimizer_steps_including_inherited=updates,protocol_sha256=digest(OUT/'protocol.json'),seed=SEED),path)
        return tensor_hash(model)

    def calibrate(self):
        torch.manual_seed(SEED);model0=Model().to(self.device)
        step_times=[];raw_times=[];checks=[]
        for rep in ('base','nav'):
            model=copy.deepcopy(model0)
            sample=next(self.sample_schedule(self.initial,[],1,SEED))
            data,b=self.batch(sample,rep)
            for j in range(16):
                self.sync();t=time.perf_counter();model.zero_grad(set_to_none=True);loss,_=self.loss(model,data,b);loss.backward();self.sync()
                step_times.append(time.perf_counter()-t);self.stats['backward_probe_steps']+=1
            model.eval()
            for case in (0,1,2):
                item=self.train[case][1];policy=NeuralPolicy(model,item,rep,self.device,self.stats)
                self.sync();t=time.perf_counter();raw=self.raw(policy);self.sync();raw_times.append(time.perf_counter()-t)
                state=State(item,raw[1:-1]);assert state.complete
                # Full/incremental prefix equivalence; no new trajectory/H.
                with torch.inference_mode():
                    enc=policy.encoded
                    nodes=torch.tensor([raw[1:-1]],device=self.device)
                    lengths=torch.tensor([len(raw)-2],device=self.device)
                    full=model.encode_prefix(enc,nodes,lengths);h=torch.zeros_like(full);cur=torch.zeros(1,dtype=torch.long,device=self.device)
                    for node in raw[1:-1]:
                        nxt=torch.tensor([node],device=self.device);h=model.update_prefix(enc,h,nxt,cur);cur=nxt
                    error=float((full-h).abs().max());assert error<1e-5
                    checks.append(dict(rep=rep,case=case,prefix_equivalence_maxabs=error))
            del model
        # No optimizer exists yet. Use conservative measured upper estimates.
        maxH=max(self.Htimes);maxprep=max(self.prepare_times);maxraw=max(raw_times);maxstep=max(step_times)
        wall=min(7200,max(1800,math.ceil(4*(6144*maxstep+12288*maxH+1728*maxprep+4096*maxraw)+120)))
        write(OUT/'calibration.json',dict(device=self.device,torch=torch.__version__,python=platform.python_version(),gpu=torch.cuda.get_device_name(0) if self.device=='cuda' else None,
            probe_steps=len(step_times),optimizer_updates=0,step_s=step_times,raw_completion_s=raw_times,max_initial_H_s=maxH,max_prepare_s=maxprep,
            prefix_checks=checks,wall_limit_s=wall,formula=self.p['wall']['formula'],initial_and_calibration_s=time.perf_counter()-self.t0,
            parameter_count=model0.parameter_count(),architecture=model0.architecture_config(),stats_before_optimizer=dict(self.stats)))
        self.ledger.deadline=time.perf_counter()+wall;self.optimizer_t0=time.perf_counter();self.wall=wall
        return model0

    def fit(self,model,opt,warm,current,steps,rep,tag,seed):
        model.train();t=time.perf_counter()
        for step,records in enumerate(self.sample_schedule(warm,current,steps,seed),1):
            self.check();data,b=self.batch(records,rep);opt.zero_grad(set_to_none=True)
            loss,metrics=self.loss(model,data,b)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();self.stats['optimizer_updates']+=1
            append(OUT/'training.jsonl',dict(tag=tag,step=step,case=records[0]['case'],**metrics))
            if step%128==0:print(json.dumps(dict(stage=tag,step=step,loss=metrics['loss'],elapsed_s=time.perf_counter()-t)),flush=True)
        model.eval();return time.perf_counter()-t

    def policy(self,model,rep,item):return NeuralPolicy(model,item,rep,self.device,self.stats)

    def prepare_dev(self):
        for i in range(32):
            scene,item,prep,sha=scene_load('dev',i,Cache(0));self.dev.append((scene,item,sha));self.stats['scenes']+=1;self.stats['prepare_calls']+=1

    def evaluate(self,phase,methods,case,item,scene,sha,cachemode=None,rep=None,preparation_s=0.,start=None):
        # methods is a one-element mapping for a single complete request.
        method,model=next(iter(methods.items()));self.check();self.sync()
        tick=time.perf_counter() if start is None else start
        metadata=dict(case=case,phase=phase,method=method,cachemode=cachemode,rep=rep,scene_sha256=sha)
        if method in ('fixed3','detour16'):row=self.oracle.native(item,method,'validation',metadata)
        else:
            dt=time.perf_counter()
            if method=='prior':
                raw=baselines.prior(item)
                self.stats['prior_raw_trajectories']=self.stats.get('prior_raw_trajectories',0)+1
                self.stats['prior_decisions']=self.stats.get('prior_decisions',0)+len(raw)-2
            elif method=='old128':
                raw=greedy_routes(model,[item],self.device)[0][0]
                self.stats['old128_raw_trajectories']=self.stats.get('old128_raw_trajectories',0)+1
                self.stats['old128_decisions']=self.stats.get('old128_decisions',0)+len(raw)-2
            elif method=='gBH':
                B=self.dev_B[case];raw=self.gbraw(item,B)
            else:
                representation=GROUPS.get(method,'base' if method=='warm_base' else 'nav')
                raw=self.raw(self.policy(model,representation,item))
            self.sync();decision=time.perf_counter()-dt
            row=self.oracle.full(item,raw,'validation',metadata);row['decision_s']=decision
        self.sync();row['measured_request_wall_s']=time.perf_counter()-tick;row['preparation_s']=preparation_s
        row['total_s']=preparation_s+row.get('decision_s',0.)+row['oracle_s']
        row['bookkeeping_and_timer_residual_s']=row['measured_request_wall_s']-row['total_s']
        plan=row.pop('plan',None);row={**row,**metadata}
        # Compact journal plus complete deployment certificates, no redundant JSON dump per label.
        append(OUT/'quality.jsonl',row)
        if plan is not None and (phase!='timing' or (cachemode=='cold' and rep==0)):
            write(OUT/'plans'/f'{phase}_{case:02}_{method}.json',plan)
        return row

    def initial_quality(self):
        self.prepare_dev();self.old,loadtime=load_policy(ROOT/'rc3_constructive/outputs/checkpoints/update0128.pt',self.device)
        write(OUT/'model_load.json',dict(old128_s=loadtime,scope='model checkpoint loading outside deployment request; imports excluded'))
        self.dev_B={}
        for i,(scene,item,sha) in enumerate(self.dev):
            fixed=self.evaluate('initial',{'fixed3':None},i,item,scene,sha);self.dev_B[i]=fixed.get('finalroute',fixed.get('route'))
            for method,model in [('warm_base',self.warm['base']),('warm_nav',self.warm['nav']),('old128',self.old),('prior',None),('gBH',None),('detour16',None)]:
                self.evaluate('initial',{method:model},i,item,scene,sha)
            print(json.dumps(dict(stage='initial_quality',case=i,calls=self.ledger.snapshot())),flush=True)

    def round_collect(self,round_index):
        self.round_data[round_index]={}
        for representation,fixed,api in [('base','A','C'),('nav','B','D')]:
            records={fixed:[],api:[]}
            for case,(_,item,sha) in enumerate(self.train):
                g0=self.policy(self.warm[representation],representation,item)
                current=self.policy(self.refs[representation],representation,item)
                if round_index==1:prefixes=checkpoints(item,self.raw(g0),[0,.25,.5,.75])
                else:
                    prefixes=checkpoints(item,self.raw(g0),[0,.5])+checkpoints(item,self.raw(current),[.25,.75])
                    prefixes=list({tuple(p):p for p in prefixes}.values())
                for prefix in prefixes:
                    score0,mask=g0.score_state(prefix);score1,_=current.score_state(prefix)
                    a0=int(np.where(mask,score0,-np.inf).argmax())+1;a1=int(np.where(mask,score1,-np.inf).argmax())+1
                    actions=menu(item,prefix,[a0,a1],self.menu_index);self.menu_index+=1
                    for group,policy,model in [(fixed,g0,self.warm[representation]),(api,current,self.refs[representation])]:
                        # Each reference has its own action as target-zero, while the queried set is exactly matched.
                        refaction=a0 if group==fixed else a1;ordered=[refaction]+[a for a in actions if a!=refaction]
                        ref=(policy,tensor_hash(model),f'{representation}_'+('g0' if model is self.warm[representation] else 'g1'))
                        rec=self.label(item,sha,prefix,ordered,ref,'round',f'round{round_index}_{group}',case)
                        rec.update(group=group,round=round_index);records[group].append(rec);append(OUT/'labels_round.jsonl',rec)
                print(json.dumps(dict(stage=f'round{round_index}_labels',representation=representation,case=case,states=len(prefixes),calls=self.ledger.snapshot())),flush=True)
            self.round_data[round_index].update(records)

    def audit(self,round_index):
        for group,representation in GROUPS.items():
            model=self.models[group]
            refmodel=self.warm[representation] if group in ('A','B') else self.refs[representation]
            for rec in self.round_data[round_index][group]:
                self.check();case=rec['case'];_,item,sha=self.train[case]
                learner=self.policy(model,representation,item);scores,mask=learner.score_state(rec['prefix'])
                action=int(np.where(mask,scores,-np.inf).argmax())+1
                if action in rec['actions']:Y=rec['Y'][rec['actions'].index(action)];billed=False
                else:
                    reference=self.policy(refmodel,representation,item)
                    raw=self.raw(reference,rec['prefix'],[action])[0]
                    key=signature(dict(scene=sha,prefix=rec['prefix'],action=action,reference=tensor_hash(refmodel),H=self.p['H_sha256']))
                    out=self.oracle.full(item,raw,'audit',dict(case=case,group=group,round=round_index,tag='actual_unknown_argmax'),cache_key=key)
                    Y=out['J'];billed=True
                pred=[float(scores[a-1]) for a in rec['actions']]
                append(OUT/'audit.jsonl',dict(group=group,round=round_index,case=case,prefix=rec['prefix'],queried=rec['actions'],selected=action,
                    Yref=rec['Yref'],Yselected=Y,predicted=float(scores[action-1]),known_best=min(rec['Y']),queried_prediction=pred,
                    queried_target=rec['target'],unknown=action not in rec['actions'],requested_new_audit=billed,reference_hash=tensor_hash(refmodel)))
            print(json.dumps(dict(stage=f'audit_round{round_index}',group=group,calls=self.ledger.snapshot())),flush=True)

    def round_quality(self,round_index):
        result={g:[] for g in GROUPS}
        for case,(scene,item,sha) in enumerate(self.dev):
            for group,model in self.models.items():result[group].append(self.evaluate(f'round{round_index}',{group:model},case,item,scene,sha))
        for fixed,api in [('A','C'),('B','D')]:
            if round_index==1 or not self.promoted.get(GROUPS[fixed],False):
                assert tensor_hash(self.models[fixed])==tensor_hash(self.models[api]),'matched parameter invariant failed'
                for a,b in zip(result[fixed],result[api]):
                    assert a['J']==b['J'] and a.get('rawroute',a.get('raw_route'))==b.get('rawroute',b.get('raw_route')) and a.get('finalroute',a.get('route'))==b.get('finalroute',b.get('route')),'matched deployment invariant failed'
        write(OUT/f'round{round_index}_invariants.json',dict(passed=True,pairs={f'{a}={b}':tensor_hash(self.models[a])==tensor_hash(self.models[b]) for a,b in [('A','C'),('B','D')]}))
        return result

    def promote(self,results):
        initial=[json.loads(s) for s in (OUT/'quality.jsonl').read_text().splitlines() if json.loads(s)['phase']=='initial']
        decision={}
        for representation,api in [('base','C'),('nav','D')]:
            baseline=[next(r['J'] for r in initial if r['case']==i and r['method']=='warm_'+representation) for i in range(32)]
            candidate=[r['J'] for r in results[api]]
            # Keep local criterion independent of analyzer output field names.
            values=100*(np.array(baseline)-candidate)/np.array(baseline)
            boot=np.random.default_rng(SEED).choice(values,size=(20000,32),replace=True).mean(1)
            low,high=np.quantile(boot,[.025,.975]);ok=values.mean()>=.5 and low>0 and all(r['valid'] for r in results[api]) and all(r['valid'] for r in initial if r['method']=='warm_'+representation)
            decision[representation]=dict(promoted=bool(ok),mean_gain_percent=float(values.mean()),CI95=[float(low),float(high)],candidate_tensor_hash=tensor_hash(self.models[api]),reference_tensor_hash=tensor_hash(self.warm[representation]))
            self.promoted[representation]=bool(ok)
            self.refs[representation]=copy.deepcopy(self.models[api]) if ok else self.warm[representation]
        write(OUT/'promotion.json',decision);print(json.dumps(dict(stage='reference_promotion',decisions=decision)),flush=True)

    def timing(self):
        methods={**self.models,'warm_nav':self.warm['nav'],'old128':self.old,'prior':None,'fixed3':None,'detour16':None}
        rng=np.random.default_rng(SEED)
        for case,(scene,_,sha) in enumerate(self.dev):
            # Navigation cache only is warm. Every method gets fresh problem/features/encoding.
            hotcache=Cache(1);hotcache.prepare(scene);self.stats['prepare_calls']+=1
            for rep in range(3):
                modes=list(rng.permutation(['cold','hot']))
                for cachemode in modes:
                    for method in rng.permutation(list(methods)):
                        self.check();self.sync();tick=time.perf_counter();cache=Cache(0) if cachemode=='cold' else hotcache
                        pt=time.perf_counter();item=cache.prepare(scene);prep=time.perf_counter()-pt;self.stats['prepare_calls']+=1
                        self.evaluate('timing',{method:methods[method]},case,item,scene,sha,cachemode,rep,prep,tick)
            print(json.dumps(dict(stage='timing',case=case,calls=self.ledger.snapshot(),elapsed_s=time.perf_counter()-self.optimizer_t0)),flush=True)

    def run(self):
        if (OUT/'execution_started.json').exists():raise FileExistsError('No implicit restart/retry')
        write(OUT/'execution_started.json',dict(protocol_sha256=digest(OUT/'protocol.json'),unix=time.time()))
        try:
            self.collect_initial();model0=self.calibrate()
            for rep in ('base','nav'):
                model=copy.deepcopy(model0);opt=torch.optim.Adam(model.parameters(),lr=1e-4)
                self.fit(model,opt,self.initial,[],1024,rep,'warm_'+rep,SEED)
                self.warm[rep]=model;self.save_model('warm_'+rep,model,1024)
                self.refs[rep]=model
            for group,rep in GROUPS.items():
                self.models[group]=copy.deepcopy(self.warm[rep]);self.opts[group]=torch.optim.Adam(self.models[group].parameters(),lr=1e-4)
            self.initial_quality()
            for r in (1,2):
                self.round_collect(r)
                for group,rep in GROUPS.items():
                    self.fit(self.models[group],self.opts[group],self.initial,self.round_data[r][group],512,rep,f'round{r}_{group}',SEED+r)
                    self.save_model(f'{group}_round{r}',self.models[group],1024+r*512)
                self.audit(r);results=self.round_quality(r)
                if r==1:self.promote(results)
            self.timing()
            assert self.stats['optimizer_updates']==6144
            status='completed'
        except Exception as exc:
            write(OUT/'failure.json',dict(type=type(exc).__name__,message=str(exc),stage='registered_R1',stats=self.stats,ledger=self.ledger.snapshot(),elapsed_s=time.perf_counter()-self.t0))
            raise
        finally:
            self.ledger.write_summary()
            write(OUT/'resources.json',dict(status=locals().get('status','inconclusive_stop'),stats=self.stats,ledger=self.ledger.snapshot(),device=self.device,
                elapsed_s=time.perf_counter()-self.t0,post_calibration_elapsed_s=time.perf_counter()-getattr(self,'optimizer_t0',self.t0),wall_limit_s=getattr(self,'wall',None),
                logical_updates_per_configuration=2048 if self.stats['optimizer_updates']==6144 else None,retained_test_queries=0,R2_started=False,R3_started=False))
        print(json.dumps(dict(stage='complete',ledger=self.ledger.snapshot(),stats=self.stats)),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['register','execute']);args=parser.parse_args()
    if args.phase=='register':register();return
    p=verify_registration();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED);np.random.seed(SEED)
    with threadpool_limits(limits=1):Experiment(p).run()


if __name__=='__main__':main()
