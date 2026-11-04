"""One real hundred-table input and CPU/CUDA model-contract smoke checks."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from rc3_rl_g2.actor_critic import ActorCritic
from rc3_rl_g2.graph_observation import build_policy_observation
from rc3_rl_g2.adapter_g2 import ResearchAdapterG2
from rc3_rl_g2.env_g2 import SearchEnvG2


def main():
    torch.set_num_threads(1);torch.manual_seed(20261103)
    source=Path('rc3_rl_research/outputs/g0/case_01/scene.json')
    scene=json.loads(source.read_text(encoding='utf-8'))
    env=SearchEnvG2(ResearchAdapterG2.from_scene(scene))
    tick=time.perf_counter();obs=build_policy_observation(env);build_s=time.perf_counter()-tick
    results=[]
    for device in ('cpu','cuda') if torch.cuda.is_available() else ('cpu',):
        for kind in ('mlp','graph'):
            model=ActorCritic(kind=kind).to(device)
            action,old,value=model.act([obs]*8)
            new,entropy,again=model.evaluate_actions([obs]*8,action)
            max_error=float((old-new).abs().max().detach().cpu())
            if max_error>1e-6:raise AssertionError('unchanged policy probability mismatch')
            before=model.actor.weight.detach().clone()
            # value from act is intentionally no_grad; use evaluated critic for
            # a real differentiable optimization check.
            objective=-(new*torch.linspace(-1.,1.,8,device=device)).mean()+.5*again.square().mean()
            optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
            optimizer.zero_grad();objective.backward()
            finite=all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
            optimizer.step()
            changed=not torch.equal(before,model.actor.weight)
            if not finite or not changed:raise AssertionError('gradient/update failed')
            for _ in range(2):model.act([obs]*8)
            if device=='cuda':torch.cuda.synchronize()
            tick=time.perf_counter()
            for _ in range(10):model.act([obs]*8)
            if device=='cuda':torch.cuda.synchronize()
            batch_s=(time.perf_counter()-tick)/10
            results.append(dict(device=device,kind=kind,parameters=sum(p.numel() for p in model.parameters()),
                unchanged_logp_max_error=max_error,finite_gradient=finite,actor_updated=changed,
                warm_eight_input_batch_mean_s=batch_s))
    result=dict(scope='Engineering input/update/throughput probe only. Reuses old development case_01; not G2 validation or model performance.',
        source=source.as_posix(),task_count=len(scene['tasks']),candidate_count=len(scene['candidates']),
        actions=len(obs['mask']),graph_tasks=len(obs['task_features']),graph_candidates=len(obs['candidate_features']),
        coverage_edges=len(obs['edge_task']),navigation_edges=len(obs['nav_source']),
        graph_build_s=build_s,torch_version=torch.__version__,results=results)
    target=Path('rc3_rl_g2/outputs/model_contract_probe.json');target.parent.mkdir(exist_ok=True,parents=True)
    target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
