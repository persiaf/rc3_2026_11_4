"""Registered S0 only. No optimizer, supervised fitting, S2 or S3 execution."""
import copy
import datetime
import json
import os
import time
import traceback
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from rc3_fast.scenes import make_complex_scene
from rc3_constructive.problem import resample_tasks
from rc3_nav_value.r1.oracle import _instrument,_empty_internal
from .problem import Parent,extract,array_hash,signature
from .environment import Environment,rule_action
from .terminal import TerminalCache
from .policy import GlobalLocalPolicy,tensor_hash
from .register import ROOT,OUT,write,read,digest,verify

COUNTERS=('parent_generation_attempts','parent_scene_generation','navigation_preparation_attempts','navigation_preparations','cropextractions',
    'ordinary_episodes','gradient_episodes','episode_steps','replay_episodes','replay_steps',
    'terminal_requests','terminal_cache_hits','terminal_attempts','terminal_completed','terminal_failed',
    'full_H_calls','export_attempts','exports','validation_attempts','validation_completed',
    'neural_encode_attempts','neural_encodes','neural_score_attempts','neural_score_calls',
    'backward_calls','backward_completed','optimizer_updates','teacher_queries','S1_calls','S2_calls','S3_calls','original60_reads')


def append(path,value):
    with path.open('a',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())


def make_allow(F,condition):
    if condition==0:return np.ones_like(F,dtype=bool)
    i,j,a=np.indices(F.shape)
    A=((i+2*j+a)%(3 if condition==1 else 4))>=(1 if condition==1 else 2)
    for t in range(len(F)):
        ids=np.argwhere(F[t])
        if len(ids):A[t,ids[0,0],ids[0,1]]=True
    return A


class Study:
    def __init__(self,p):
        self.p=p;self.started=time.perf_counter();self.deadline=self.started+p['limits']['wall_s']
        self.stats={k:0 for k in COUNTERS}
        self.parents={};self.views={};self.episodes=[];self.gradient_rows=[];self.seed_data=read(OUT/'seeds.json')
        self.cache=TerminalCache(self.stats,lambda r:append(OUT/'terminal_journal.jsonl',r),p['H_sha256'],
            p['limits']['full_terminal_attempts'],self.deadline)
        self.internal=_empty_internal()
        torch.manual_seed(self.seed_data['S0_network_seed']);self.model=GlobalLocalPolicy()
        self.initial_hash=tensor_hash(self.model)
        torch.save(dict(state=self.model.state_dict(),hidden=64,seed=self.seed_data['S0_network_seed'],
            tensor_sha256=self.initial_hash,S0_protocol_sha256=digest(OUT/'S0_protocol.json'),optimizer_updates=0),OUT/'initial_policy.pt')

    def check(self):
        if time.perf_counter()>=self.deadline:raise TimeoutError('S0 deadline; no successful finish after overrun')

    def resources(self):
        write(OUT/'resources.json',dict(stage='S0',stats=self.stats,internal_search=self.internal,
            elapsed_wall_s=time.perf_counter()-self.started,deadline_remaining_s=max(0.,self.deadline-time.perf_counter()),
            synthetic_pre_registration_units=dict(environment=read(OUT/'unit_receipt.json'),policy=read(OUT/'policy_unit_receipt.json')),
            model_hash_before=self.initial_hash,model_hash_now=tensor_hash(self.model),
            scope='S0 engineering consistency/gradient probes only; zero optimizer/teacher/S2/S3/test60. Synthetic unit probes are separately counted, not new real scenes.'))

    def parent(self,spec):
        self.check();tick=time.perf_counter()
        self.stats['parent_generation_attempts']+=1
        scene=resample_tasks(make_complex_scene(10,10,spec['map_seed'],spec['mode']),spec['task_seed'])
        self.stats['parent_scene_generation']+=1
        # First preparation computes both frozen proxy F and unchanged full-map navigation exactly once.
        self.stats['navigation_preparation_attempts']+=1
        parent=Parent.from_scene(scene);self.stats['navigation_preparations']+=1
        A=make_allow(parent.F,spec['A_condition'])
        parent=Parent(parent.base,parent.F,A)
        preparation_s=time.perf_counter()-tick
        case=spec['case'];write(OUT/'parents'/f'{case:02}_scene.json',parent.scene)
        path=OUT/'parents'/f'{case:02}_arrays.npz'
        np.savez_compressed(path,F=parent.F,A=parent.A,D=parent.D,
            nav_coordinates=np.asarray(parent.base.nav['coordinates']),service_nodes=np.asarray(parent.base.nav['service_nodes']),
            predecessors=parent.base.nav['predecessors'])
        description=dict(case=case,identity=parent.identity,scene_sha256=digest(OUT/'parents'/f'{case:02}_scene.json'),
            arrays_sha256=digest(path),F_sha256=array_hash(parent.F),A_sha256=array_hash(parent.A),D_sha256=array_hash(parent.D),
            metadata=spec,elapsed_preparation_and_save_s=time.perf_counter()-tick,
            kernel_preparation_s=parent.base.preprocessing_s,parent_generation_and_preparation_s=preparation_s,
            navigation_nodes=parent.base.nav['node_count'],navigation_edges=parent.base.nav['edge_count'])
        write(OUT/'parents'/f'{case:02}_receipt.json',description);self.parents[case]=(parent,description)
        return parent

    def view(self,parent,parent_case,width,height,position):
        self.check()
        if position=='corner':x,y,entry,exit=0,0,0,0
        else:
            x,y=(10-width)//2,(10-height)//2
            connected=np.flatnonzero(np.isfinite(parent.D[0,1:])&np.isfinite(parent.D[1:,0]))+1
            if len(connected)<2:raise ValueError('Not enough connected endpoints')
            entry,exit=int(connected[0]),int(connected[-1])
        p=extract(parent,x,y,width,height,entry,exit);self.stats['cropextractions']+=1
        expected_tables={t['id'] for t in parent.scene['tables'] if x<=t['ix']<x+width and y<=t['iy']<y+height}
        expected_tasks=[t['id'] for t in parent.scene['tasks'] if t['table_id'] in expected_tables]
        expected_candidates=[j for j in range(len(parent.scene['candidates'])) if
            any(parent.F[i,j,a] and parent.A[i,j,a] for i in expected_tasks for a in (0,1))
            and np.isfinite(parent.D[entry,j+1]) and np.isfinite(parent.D[j+1,exit])]
        assert p.task_ids.tolist()==expected_tasks and p.candidate_ids.tolist()==expected_candidates
        assert np.array_equal(p.F,(parent.F&parent.A)[np.ix_(p.task_ids,p.candidate_ids,np.arange(2))])
        nodes=p.candidate_ids+1
        assert np.array_equal(p.D[1:,1:],parent.D[np.ix_(nodes,nodes)])
        assert np.array_equal(p.D[0,1:],parent.D[entry,nodes]) and np.array_equal(p.D[1:,0],parent.D[nodes,exit])
        case=len(self.views);self.views[case]=p
        description=dict(case=case,parent_case=parent_case,position=position,**p.description(),
            independent_extraction_checks=True,task_count=len(p.task_ids),candidate_count=len(p.candidate_ids))
        write(OUT/'views'/f'{case:02}.json',description)
        return case,p

    def rollout(self,case,p,method,seed,forced=None,gradient=False):
        self.check();tick=time.perf_counter();env=Environment(p,self.cache)
        logsum=torch.zeros((),dtype=torch.float32);score_rows=[]
        rng=np.random.default_rng(seed);torch.manual_seed(seed)
        with torch.set_grad_enabled(gradient):
            encoded=None
            if method=='neural' and not env.done:
                self.stats['neural_encode_attempts']+=1
                encoded=self.model.encode(p);self.stats['neural_encodes']+=1
            while not env.done:
                self.check();extra=None
                if forced is not None and not env.prefix:a=int(forced)
                elif method=='random':a=int(rng.choice(np.flatnonzero(env.mask())))+1
                elif method=='rule':a=rule_action(env)
                else:
                    self.stats['neural_score_attempts']+=1
                    scores=self.model.score(env,encoded);self.stats['neural_score_calls']+=1
                    mask=torch.tensor(env.mask());prob=torch.softmax(scores,0)
                    assert torch.isfinite(scores[mask]).all() and bool((prob[~mask]==0).all())
                    distribution=torch.distributions.Categorical(logits=scores)
                    node=distribution.sample() if gradient else scores.argmax()
                    a=int(node)+1
                    if gradient:logsum=logsum+distribution.log_prob(node)
                    extra=dict(chosen_logprob=float(distribution.log_prob(node).detach()),
                        entropy=float(distribution.entropy().detach()),legal_actions=int(mask.sum()),
                        illegal_probability_max=0.,logits_sha256=array_hash(scores.detach().numpy()))
                event=env.step(a);self.stats['episode_steps']+=1
                if extra is not None:event.update(policy=extra)
        self.stats['gradient_episodes' if gradient else 'ordinary_episodes']+=1
        raw=[0,*env.prefix,0];result=env.result
        replay=Environment(p,self.cache);replay.replay(env.prefix)
        self.stats['replay_episodes']+=1;self.stats['replay_steps']+=len(env.prefix)
        assert replay.result['route']==result['route'] and abs(sum(replay.rewards)-sum(env.rewards))<1e-7
        assert replay.trace==[{k:v for k,v in event.items() if k!='policy'} for event in env.trace]
        assert replay.rewards==env.rewards and replay.prefix==env.prefix and np.array_equal(replay.remaining,env.remaining)
        assert abs(sum(env.rewards)+result['J'])<1e-7
        row=dict(episode=len(self.episodes),case=case,method=method,seed=seed,gradient_probe=gradient,
            forced_first_local_node=forced,raw_route=raw,final_route=result['route'],
            raw_J=result['raw_J'],J=result['J'],L=result['L'],K=result['K'],
            rewards=env.rewards,return_=sum(env.rewards),trace=env.trace,
            logprob_sum_excluding_forced=float(logsum.detach()),valid=result['plan']['validation']['valid_model_plan'],
            elapsed_s=time.perf_counter()-tick,terminal_cache_hit=result['cache_hit'])
        self.episodes.append(row);append(OUT/'episodes.jsonl',row)
        write(OUT/'plans'/f"{row['episode']:03}_{method}.json",result['plan'])
        return row,logsum

    def run(self):
        terms=[]
        with _instrument(self.internal):
            for spec in self.seed_data['S0']:
                parent=self.parent(spec)
                for width,height in ((4,3),(6,6),(8,8)):
                    for position in ('corner','center'):
                        case,p=self.view(parent,spec['case'],width,height,position)
                        for method in ('random','rule','neural'):
                            self.rollout(case,p,method,9900000000+case)
                        if (width,height,position)==(8,8,'center'):
                            ids=np.flatnonzero(p.C.any(0))+1
                            starts=sorted(set(int(ids[k]) for k in (0,(len(ids)-1)//2,len(ids)-1)))
                            rows=[];logs=[]
                            for k,first in enumerate(starts):
                                row,logp=self.rollout(case,p,'neural',11612026+100+spec['case']*3+k,first,True)
                                rows.append(row);logs.append(logp)
                            returns=np.asarray([r['return_'] for r in rows]);advantages=returns-returns.mean()
                            terms.extend(-float(a)*lp for a,lp in zip(advantages,logs))
                            self.gradient_rows.append(dict(parent_case=spec['case'],case=case,
                                forced_starts=starts,returns=returns.tolist(),baseline=float(returns.mean()),advantages=advantages.tolist()))
                _,old=self.parents[spec['case']]
                assert signature(dict(scene=parent.scene,F=array_hash(parent.F),A=array_hash(parent.A),mode=parent.mode))==parent.identity
                assert array_hash(parent.D)==old['D_sha256']
                self.resources();print(json.dumps(dict(parent_done=spec['case']+1,episodes=len(self.episodes),stats=self.stats)),flush=True)
        self.check();loss=torch.stack(terms).mean()
        if not loss.requires_grad or not torch.isfinite(loss):raise ValueError('No finite policy-gradient graph')
        self.model.zero_grad(set_to_none=True);self.stats['backward_calls']+=1;loss.backward();self.stats['backward_completed']+=1
        norms={name:float(param.grad.norm()) if param.grad is not None else 0. for name,param in self.model.named_parameters()}
        if not all(np.isfinite(list(norms.values()))) or sum(norms.values())<=1e-12:raise ValueError('Empirical gradient is zero/nonfinite; no formal training admission')
        assert tensor_hash(self.model)==self.initial_hash
        assert self.stats['parent_scene_generation']==3 and self.stats['navigation_preparations']==3
        assert self.stats['cropextractions']==18 and self.stats['ordinary_episodes']==54 and self.stats['gradient_episodes']==9
        assert self.stats['terminal_failed']==0 and all(r['valid'] for r in self.episodes)
        write(OUT/'gradient_probe.json',dict(phase='S0 backward-only consistency probe',loss=float(loss.detach()),
            groups=self.gradient_rows,parameter_gradient_norms=norms,all_finite=True,nonzero=True,
            tensor_hash_before=self.initial_hash,tensor_hash_after=tensor_hash(self.model),optimizer_updates=0,
            forced_first_logprob_excluded=True,claim='Gradient/MDP engineering check only; no learning improvement evidence'))
        self.check();verify();self.resources();self.check()
        write(OUT/'STATUS.json',dict(S0='complete_pass',S1='not_started',S2='not_started',S3='not_started',
            DRL_core_fixed=True,optimizer_updates=0,original60='sealed',physical_joint_trajectory='not_certified',
            old_studies='frozen',default_methods=['Fixed3','Detour'],pushed=False))
        self.check()


def main():
    p=verify()
    if (OUT/'run_started.json').exists():raise FileExistsError('S0 already attempted; no implicit replay')
    write(OUT/'run_started.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),S0_protocol_sha256=digest(OUT/'S0_protocol.json'),optimizer_updates=0))
    study=None;initialization_started=time.perf_counter()
    try:
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        with threadpool_limits(limits=1):study=Study(p);study.run()
    except Exception as e:
        if study is not None:study.resources()
        else:
            write(OUT/'resources.json',dict(stage='S0',stats={k:0 for k in COUNTERS},internal_search=_empty_internal(),
                elapsed_wall_s=time.perf_counter()-initialization_started,deadline_remaining_s=None,
                scope='Initialization failed before Study returned; no parent/route work starts in constructor; constructor failure preserved, zero real calls'))
        write(OUT/'STATUS.json',dict(S0='incomplete_stop',S1='not_started',S2='not_started',S3='not_started',
            error=type(e).__name__,message=str(e),optimizer_updates=0,original60='sealed',pushed=False))
        write(OUT/'failure.json',dict(error=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
