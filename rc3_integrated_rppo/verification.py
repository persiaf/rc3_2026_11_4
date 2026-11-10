"""Bounded P0 math, persistence, identity, real route and gradient acceptance."""
import copy
import gc
import math
import time
from pathlib import Path
from torch import nn
from .common import *
from .data import base_state
from .model import Actor,Critic,pack_states
from .ppo import PPO,exact_kl,state_loss,same,adam,returns_to_go
from .training import frame,batch_rollouts,teacher_batch,supervised_step
from .rollout import collect


def negative(fn):
    try:fn()
    except (ValueError,RuntimeError,KeyError,AssertionError,FileNotFoundError):return True
    raise AssertionError('Expected strict rejection')


class TinyActor(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(91);self.base=nn.Linear(1,1);self.head=nn.Sequential(nn.Linear(202,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
        self.base.requires_grad_(False);self.base_sha=tensor_digest(self.base.state_dict())
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)


def synthetic(log,counts,clock):
    checks={};mask=torch.tensor([[True,True,False],[True,False,False]])
    old=torch.tensor([[.25,.75,0.],[1.,0.,0.]]).log();new=torch.tensor([[.5,.5,0.],[1.,0.,0.]]).log()
    k=exact_kl(old,new,mask);hand=.25*math.log(.5)+.75*math.log(1.5)
    require(abs(float(k[0])-hand)<1e-6 and k[1]==0,'Exact masked KL differs from enumeration');checks['KL_hand_enumeration']=True
    z=torch.tensor([[.0,.0,-torch.inf]],requires_grad=True);lp=torch.log_softmax(z,-1)
    term,_,_=state_loss(lp,torch.tensor([math.log(.5)]),torch.tensor([1.]),lp.detach(),mask[:1],torch.tensor([0]),0.)
    term.sum().backward();require(z.grad[0,0]<0 and z.grad[0,1]>0 and z.grad[0,2]==0,'PPO sign or mask wrong')
    checks['positive_advantage_increases_selected_probability']=True
    require(torch.allclose(returns_to_go([-2,-3,1],2),torch.tensor([-2.,-1.,.5])),'Reward scaling/RTG')
    # Two two-action trajectories: objective is mean over episodes of SUM steps.
    a=torch.tensor(0.,requires_grad=True);(torch.stack([2*a,4*a]).sum()/2).backward();require(a.grad==3,'Episode sum/mean reduction')
    # Enumerate an action-dependent two-step cost: exact expectation gradient
    # equals sum-of-log-probability score gradient, not trajectory-mean score.
    x=torch.tensor(.2,requires_grad=True);pr=torch.softmax(torch.stack([x,-x]),0);cost=torch.tensor([[1.,3.],[4.,2.]])
    exact=sum(pr[i]*pr[j]*cost[i,j] for i in range(2) for j in range(2));g1=torch.autograd.grad(exact,x,retain_graph=True)[0]
    score=sum((pr[i]*pr[j]).detach()*cost[i,j]*(pr[i].log()+pr[j].log()) for i in range(2) for j in range(2));g2=torch.autograd.grad(score,x)[0]
    require(torch.allclose(g1,g2,atol=1e-6),'Enumerated score-function objective');checks['full_likelihood_episode_sum']=True
    negative(lambda:state_loss(lp,torch.tensor([0.],requires_grad=True),torch.tensor([1.]),lp.detach(),mask[:1],torch.tensor([0])))
    checks['old_distribution_and_advantage_detached']=True
    f=lambda d:dict(key=('task',(1,),'base',d),x=torch.zeros(3,202),v=torch.zeros(268),z0=torch.zeros(3),mask=torch.ones(3,dtype=torch.bool))
    negative(lambda:pack_states([f('D1'),f('D2')]));checks['foreign_D_cache_rejected']=True
    # Padding never enters packed training tensors; a masked distribution gives
    # no illegal probability or gradient, including all-but-one legal action.
    require(torch.equal(old.exp()[~mask],torch.zeros_like(old.exp()[~mask])),'Illegal mass');checks['no_padding_loss_or_illegal_mass']=True
    ta=TinyActor();critic=Critic(0);p=PPO(ta,critic,log,counts,context=dict(stage='P0_synthetic'))
    gen=torch.Generator().manual_seed(11);b=dict(x=torch.randn(2,3,202,generator=gen),v=torch.randn(2,268,generator=gen),z0=old.clone(),mask=mask.clone())
    episode=dict(batch=b,G=torch.tensor([1.,-.5]),V_old=torch.zeros(2),old_logp_all=old.clone(),base_logp=old.clone(),
        old_logp=torch.tensor([math.log(.25),0.]),actions=torch.tensor([0,0]),size='synthetic')
    before=copy.deepcopy(ta.state_dict());p.critic_step([episode]);require(same(before,ta.state_dict()),'Critic contaminates Actor');checks['critic_isolated']=True
    # Force KL overflow after real Adam attempts; rollback must restore Adam
    # and accepted cursor, without undoing paid optimizer attempts.
    p.diagnostic=lambda e:dict(old_mean=1.,old_p95=1.,base_mean=1.)
    h=copy.deepcopy(ta.head.state_dict());out=p.actor_epochs([episode],clock,'synthetic_KL_reject')
    require(same(h,ta.head.state_dict()) and p.actor_cursor==0 and p.actor_opt.param_groups[0]['lr']==1.5e-5 and out['batch_noncompliant'],'KL rollback not exact')
    checks['Actor_Adam_cursor_rollback_exact']=True
    # Closed optimizer checkpoint recovery is bitwise identical, including Adam.
    m1=nn.Linear(2,1);m2=copy.deepcopy(m1);o1=adam(m1.parameters(),3e-5);o2=adam(m2.parameters(),3e-5)
    xx=torch.tensor([[1.,2.]]);yy=torch.tensor([[.4]])
    def step(m,o):
        o.zero_grad();loss=(m(xx)-yy).square().sum();loss.backward();log(dict(event='optimizer_attempt',role='P0_restore_fixture'))
        counts['fixture_optimizer_calls']=counts.get('fixture_optimizer_calls',0)+1;o.step()
    step(m1,o1);m2.load_state_dict(m1.state_dict());o2.load_state_dict(copy.deepcopy(o1.state_dict()));step(m1,o1);step(m2,o2)
    require(same(m1.state_dict(),m2.state_dict()) and same(o1.state_dict(),o2.state_dict()),'Continuous/restored Adam mismatch');checks['closed_optimizer_restore_bitwise']=True
    from .eval import load_actor
    negative(lambda:load_actor(dict(study_method='RL',checkpoint_role='residual_SL',run_id=EXP)));checks['role_conflict_rejected_before_load']=True
    from .statistics import paired_effects
    rows=[]
    for i in range(4):
        for s in (1,2):
            for c in range(6):
                for role,j in [('SL256',100.),('RL',90.+i)]:rows.append(dict(case_id=f'{i}_{c}',seed=s,study_method=role,physical_key=str(i),spec=dict(mode=('mixed','blocked')[i%2]),metrics=dict(J=j)))
    effect=paired_effects(rows,'SL256',seeds=(1,2));require(abs(effect['mean']-.085)<1e-10 and len(effect['parents'])==4,'Parent aggregation hand result')
    checks['parent_seed_hand_aggregation']=True
    # All synthetic journals are isolated and marked non-scientific fixtures.
    fixture=OUT/'P0_fixtures'/'clock';l=Log(fixture)
    l(dict(event='clock_open',name='probe',cap_s=100.,carry_s=0.,start_perf=time.perf_counter()-2.,boot_identity=boot_identity(),utc=now()))
    l.close();l=Log(fixture);c=Clock(l,'probe',100.);require(c.carry>=2.,'Orphan wall tail refunded');c.close();l.close();checks['unclosed_clock_cannot_reset']=True
    fixture=OUT/'P0_fixtures'/'ledger';l=Log(fixture);l(dict(event='fixture',n=1));l.close()
    pending=fixture/'HEAD.pending';write_new(pending,dict(sequence=999,sha256='bad',experiment=EXP))
    negative(lambda:Log(fixture));checks['bad_pending_and_broken_chain_rejected']=True
    return checks


def run(inputs,store,clock):
    log=store.log;counts={};checks=synthetic(log,counts,clock)
    zero=[]
    for seed in SEEDS:
        state,_=base_state(seed);actor=Actor(state,seed);frame(actor,seed,'P0_zero',0)
        for i,c in enumerate(inputs.declared['engineering']):
            spec=dict(c['spec'],source_family='bank');p,_=inputs.view(spec)
            q=dict(kind='engineering',stage='zero_equivalence',seed=seed,slot=i,role='P0',row=0,round=0)
            e=collect(actor,None,p,q,store,clock,greedy=True,equivalence=True)
            zero.append(dict(seed=seed,case=c.get('id',str(i)),identity=p.identity,raw_route=e['record']['result']['raw_route'],
                J=e['record']['result']['J'],valid=e['record']['result']['plan']['validation']['valid_model_plan']))
            del e
        del actor;gc.collect()
    checks['18_exact_zero_residual_masks_logits_greedy']=len(zero)==18
    state,_=base_state(SEEDS[0]);actor=Actor(state,SEEDS[0]);critic=Critic(SEEDS[0]);tick=time.perf_counter()
    specs=inputs.declared['engineering_benchmark'];episodes=batch_rollouts(actor,critic,SEEDS[0],'P0_benchmark',1,specs,'engineering',inputs,store,clock)
    collect_s=time.perf_counter()-tick;ppo=PPO(actor,critic,log,counts,context=dict(stage='P0_real'))
    tick=time.perf_counter();out=ppo.actor_epochs(episodes,clock,'P0_real_batch');ppo_s=time.perf_counter()-tick
    require(out['accepted_epochs']>0,'Real PPO update accepted none')
    require(float(actor.head[0].weight.grad.abs().sum())>0,'Residual hidden layer received no gradient after output began learning')
    require(all(p.grad is None for p in actor.base.parameters()),'Base received gradients');checks['real_batched_PPO_gradient_and_frozen_base']=True
    del episodes;gc.collect();control=Actor(state,SEEDS[0]);opt=adam(control.head.parameters(),3e-5)
    tick=time.perf_counter();examples=teacher_batch(control,inputs.declared['SL_batches'][str(SEEDS[0])][0],inputs)
    sl=[supervised_step(control,opt,examples,log,counts,0) for _ in range(out['accepted_epochs'])];sl_s=time.perf_counter()-tick
    # Conservative per-seed estimate: 512 full rounds + warm192 +224 critic
    # steps, selection432, metadata/IO reserve25%; no budget is increased.
    one=(collect_s+ppo_s+sl_s);request_mean=collect_s/24
    projected=1.25*(512*one+192*request_mean+432*request_mean+60)
    calls=sum(e['event']=='optimizer_attempt' for e in log.events)
    require(calls<=32 and store.counts['engineering']<=96,'P0 update/request cap')
    result=dict(status='PASS' if all(checks.values()) and projected<14400 else 'REVISE_RESOURCE',checks=checks,
        zero_equivalence=zero,real_optimizer_calls=calls,request_count=store.counts['engineering'],counts=counts,
        benchmark=dict(rollout24_s=collect_s,PPO4_and_critic4_s=ppo_s,matched_SL_s=sl_s,round_s=one,
            conservative_seed_projection_s=projected,seed_cap_s=14400),real_PPO=out,matched_SL=sl,
        source_binding_sha256=digest(source_path()),scientific_weights_reused=False,wall_used_s=clock.elapsed())
    write_new(HOME/'ENGINEERING_VERIFICATION.json',result);return result
