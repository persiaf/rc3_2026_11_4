"""Batch-64 training update timing on real raw local graphs, no learned cache."""
import json
from pathlib import Path
import time
import torch
from rc3_rl_g2.actor_critic import ActorCritic
from rc3_rl_g2.graph_observation import build_policy_observation
from rc3_rl_g2.adapter_g2 import ResearchAdapterG2
from rc3_rl_g2.env_g2 import SearchEnvG2


def rss():
    try:
        import psutil
        return dict(process_rss=psutil.Process().memory_info().rss,available_system=psutil.virtual_memory().available)
    except ImportError:return {}


def main():
    torch.set_num_threads(1);torch.manual_seed(20261103)
    observations=[];queries=0
    for case_id in (1,14):
        path=Path(f'rc3_rl_research/outputs/g0/case_{case_id:02d}/scene.json')
        env=SearchEnvG2(ResearchAdapterG2.from_scene(json.loads(path.read_text(encoding='utf-8'))))
        for step in range(4):
            observations.append(build_policy_observation(env))
            obs=env.observe()
            legal=[i for i,valid in enumerate(obs['mask']) if valid and obs['actions'][i]['kind']!='stop']
            if not legal:break
            env.step(legal[step%len(legal)]);queries+=1
        if not env.done:env.terminate('engineering_device_probe')
    batch=[observations[i%len(observations)] for i in range(64)]
    results=[];initial_state=None
    for device in ('cpu','cuda') if torch.cuda.is_available() else ('cpu',):
        torch.manual_seed(20261103)
        model=ActorCritic(kind='graph',hidden=64,message_steps=2).to(device)
        if initial_state is None:initial_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:model.load_state_dict(initial_state)
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
        actions,old_logp,old_value=model.act(batch)
        advantages=torch.linspace(-1,1,64,device=device)
        target_value=old_value+advantages
        def update():
            logp,entropy,value=model.evaluate_actions(batch,actions)
            ratio=(logp-old_logp).exp()
            loss=torch.maximum(-advantages*ratio,-advantages*ratio.clamp(.8,1.2)).mean()
            loss=loss+.5*(value-target_value).square().mean()-.01*entropy.mean()
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.5)
            if not bool(torch.isfinite(norm)):raise AssertionError('nonfinite gradient norm')
            optimizer.step()
            return float(loss.detach().cpu())
        for _ in range(3):update()
        if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        before=rss();tick=time.perf_counter()
        losses=[update() for _ in range(10)]
        if device=='cuda':torch.cuda.synchronize()
        elapsed=time.perf_counter()-tick
        result=dict(device=device,batch_size=64,message_steps=2,parameters=sum(p.numel() for p in model.parameters()),
            warmup_updates=3,timed_updates=10,mean_complete_update_s=elapsed/10,losses=losses,
            memory_before=before,memory_after=rss())
        if device=='cuda':
            result.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                cuda_allocated_bytes=torch.cuda.memory_allocated(),cuda_reserved_bytes=torch.cuda.memory_reserved())
        results.append(result)
        del optimizer,model
        if device=='cuda':torch.cuda.empty_cache()
    report=dict(scope='Training-device engineering probe only, not a planning or learning performance comparison. Reuses G01 development cases; all optimizer updates are disposable.',
        raw_observations=len(observations),distinct_state_keys=len(set(o['state_key'] for o in observations)),
        preparation_environment_queries=queries,torch_threads=torch.get_num_threads(),results=results)
    output=Path('rc3_rl_g2/outputs/model_device_probe.json')
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
