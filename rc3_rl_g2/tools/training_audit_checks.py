"""Independent training-contract probes; no formal PPO/BC updates or samples."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from rc3_rl_g2.collector import VectorCollector
from rc3_rl_g2.ppo_math import compute_gae
from rc3_rl_g2.actor_critic import ActorCritic
from rc3_rl_g2.checkpoints import save_model,load_model


class ValueProbe:
    def act(self,observations):
        n=len(observations)
        return torch.zeros(n,dtype=torch.long),torch.zeros(n),torch.tensor([o['tag'] for o in observations],dtype=torch.float32)
    def __call__(self,observations):
        return torch.zeros((len(observations),1)),torch.tensor([o['tag'] for o in observations],dtype=torch.float32)


def main():
    torch.set_num_threads(1)
    checks=[]
    collector=object.__new__(VectorCollector)
    collector.workers=2
    collector.observations=[dict(tag=10),dict(tag=100)]
    sequence=[([20,110],[1,2],[True,False]),([30,120],[3,4],[False,True]),([40,130],[5,6],[False,False])]
    index=[0]
    def exchange(command,actions):
        assert command=='step' and actions==[0,0]
        tags,rewards,terminal=sequence[index[0]];index[0]+=1
        return [dict(observation=dict(tag=t),reward=r,terminated=d,summary=None,
                     action_kind='improve',route_changed=True) for t,r,d in zip(tags,rewards,terminal)]
    collector.command_all=exchange
    batch=collector.collect(ValueProbe(),3)
    assert [o['tag'] for o in batch['observations']]==[10,100,20,110,30,120]
    np.testing.assert_array_equal(batch['values'],[[10,100],[20,110],[30,120]])
    np.testing.assert_array_equal(batch['next_values'],[[0,110],[30,0],[40,130]])
    advantages,_=compute_gae(batch['rewards'],batch['values'],batch['next_values'],batch['terminated'])
    np.testing.assert_allclose(advantages.numpy(),[[-9,-88.7],[27.25,-106],[15,16]],atol=1e-5)
    assert collector.observations==[dict(tag=40),dict(tag=130)]
    checks.append(dict(name='T-major batch, true terminal reset masking, GAE and collector continuation',passed=True))

    logits=torch.tensor([2.,1.,1000.],requires_grad=True)
    evidence=torch.tensor([True,True,False]);ties=torch.tensor([True,False,False])
    loss=-logits.masked_fill(~evidence,-torch.inf).log_softmax(0)[ties].mean();loss.backward()
    assert logits.grad[2].item()==0.
    expected=torch.log1p(torch.exp(torch.tensor(-1.)))
    torch.testing.assert_close(loss,expected)
    checks.append(dict(name='Unknown teacher action absent from normalization and negative gradient',passed=True,
                       unknown_logit=1000.,unknown_gradient=float(logits.grad[2])))

    directory=Path('rc3_rl_g2/outputs/training_review');directory.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(7);model=ActorCritic(message_steps=2)
    checkpoint=directory/'synthetic_init.pt'
    save_model(checkpoint,model,protocol_sha256='synthetic_contract_probe_not_training')
    children=[load_model(checkpoint)[0] for _ in range(3)]
    assert all(all(torch.equal(t,child.state_dict()[k]) for k,t in model.state_dict().items()) for child in children)
    checks.append(dict(name='Three branches loaded from one checkpoint have exactly identical initial parameters',passed=True,
                       synthetic_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
    root=Path(__file__).resolve().parents[1]
    report=dict(scope='Independent synthetic contract probes; no formal training, dataset use, or superiority measurement.',
        checks=checks,all_passed=all(c['passed'] for c in checks),
        inspected_source_hashes={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in
                                ('train_ppo.py','collector.py','worker.py','checkpoints.py')})
    (directory/'numerical_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
