"""Remaining preregistered P0 budget: CUDA head FP32, CPU frozen base/critic."""
import gc
import time
from .common import *
from .data import Inputs,base_state
from .model import Actor,Critic,batch_logits
from .ppo import PPO,adam
from .training import batch_rollouts,teacher_batch,supervised_step


def main():
    configure();verify_sources();reg=read(HOME/'GPU_ENGINEERING_REGISTRATION.json');log=Log();clock=Clock(log,'P0',1800);inputs=Inputs(log,clock);store=Store(log);counts={}
    try:
        if (HOME/'GPU_ENGINEERING_VERIFICATION.json').exists():raise ValueError('Do not repeat completed GPU probe')
        require(torch.cuda.is_available(),'Registered CUDA device unavailable')
        existing=sum(e['event']=='optimizer_attempt' for e in log.events)
        require(existing==22,'GPU probe requires original closed22-engineering-update boundary')
        def bounded(row):
            if row['event']=='optimizer_attempt':require(sum(e['event']=='optimizer_attempt' for e in log.events)<32,'P0 optimizer hard cap')
            log(row)
        state,_=base_state(SEEDS[0]);actor=Actor(state,SEEDS[0],TRAIN_DEVICE);critic=Critic(SEEDS[0]);tick=time.perf_counter()
        episodes=batch_rollouts(actor,critic,SEEDS[0],'P0_GPU',1,inputs.declared['engineering_benchmark'],'engineering',inputs,store,clock)
        torch.cuda.synchronize();rollout_s=time.perf_counter()-tick
        # Exact zero output: every recorded taken LP must equal frozen pi0 and
        # all legal logits, not just a sampled action, were checked on CPU.
        equal=True
        for e in episodes:
            if e['batch'] is None:continue
            with torch.no_grad():z=batch_logits(actor,e['batch']).cpu()
            equal=equal and torch.equal(z,e['batch']['z0'])
        require(equal,'Zero CUDA residual changed frozen base logits/mask')
        # Independently compare FP32 batch objective gradients, no Adam calls.
        example=next(e for e in episodes if e['batch'] is not None);cpu=Actor(state,SEEDS[0]);b=example['batch'];b={k:(v[:2] if isinstance(v,torch.Tensor) else v) for k,v in b.items()}
        def grad(a):
            lp=torch.log_softmax(batch_logits(a,b),-1);actions=example['actions'][:2].to(lp.device)
            loss=-lp.gather(1,actions[:,None]).sum();return torch.autograd.grad(loss,tuple(a.head.parameters()))
        g1=grad(cpu);g2=grad(actor);errors=[]
        for x,y in zip(g1,g2):
            errors.append(float((x-y.cpu()).abs().max()));require(torch.allclose(x,y.cpu(),atol=1e-5,rtol=2e-4),'CPU/CUDA FP32 gradient mismatch')
        del cpu;gc.collect();ppo=PPO(actor,critic,bounded,counts,context=dict(stage='P0_GPU'))
        tick=time.perf_counter();out=ppo.actor_epochs(episodes,clock,'P0_GPU');torch.cuda.synchronize();ppo_s=time.perf_counter()-tick
        del episodes;gc.collect();control=Actor(state,SEEDS[0],TRAIN_DEVICE);opt=adam(control.head.parameters(),3e-5)
        tick=time.perf_counter();examples=teacher_batch(control,inputs.declared['SL_batches'][str(SEEDS[0])][0],inputs);prepare_s=time.perf_counter()-tick
        steps=min(2,32-sum(e['event']=='optimizer_attempt' for e in log.events));require(steps>0,'No engineering Adam budget remains')
        tick=time.perf_counter();sl=[supervised_step(control,opt,examples,bounded,counts,0) for _ in range(steps)];torch.cuda.synchronize();sl_step_s=(time.perf_counter()-tick)/steps
        round_s=rollout_s+ppo_s+prepare_s+4*sl_step_s
        projected=1.25*(512*round_s+192*rollout_s/24+432*rollout_s/24+60)
        result=dict(status='PASS' if projected<14400 else 'REVISE_RESOURCE',old_CPU_report_sha256=digest(HOME/'ENGINEERING_VERIFICATION.json'),
            exact_zero_residual_logits_masks=equal,FP32_gradient_abs_errors=errors,gradient_tolerance=reg['gradient_tolerance'],
            GPU=torch.cuda.get_device_name(0),training_actor_device='cuda_fp32_noTF32',base_critic_and_deployment='CPU_FP32',
            benchmark=dict(rollout24_s=rollout_s,PPO4_and_critic4_s=ppo_s,teacher_prepare_s=prepare_s,SL_step_s=sl_step_s,round_s=round_s,
                conservative_seed_projection_s=projected,seed_cap_s=14400),
            optimizer_total=sum(e['event']=='optimizer_attempt' for e in log.events),engineer_complete_requests=store.counts['engineering'],counts=counts,
            actual_PPO=out,actual_SL=sl,scientific_optimizer_updates=0,source_binding_sha256=digest(source_path()),wall_used_s=clock.elapsed())
        require(result['optimizer_total']<=32 and result['engineer_complete_requests']<=96,'P0 hard cap')
        write_new(HOME/'GPU_ENGINEERING_VERIFICATION.json',result);log(dict(event='GPU_probe_complete',receipt_sha256=digest(HOME/'GPU_ENGINEERING_VERIFICATION.json')))
        print(json.dumps({k:result[k] for k in ('status','optimizer_total','engineer_complete_requests','benchmark')}),flush=True)
    finally:clock.close();log.close()


if __name__=='__main__':main()
