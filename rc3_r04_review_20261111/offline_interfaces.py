"""Bounded saved-logit and synthetic typed-interface checks, never a rollout."""
import csv
import gzip
import itertools
from pathlib import Path
import torch
from torch import nn

from rc3_conservative_refinement.adapter import LookaheadResidualActor
from rc3_conservative_refinement.model import remaining_cost_loss
from rc3_conservative_refinement.trainer import masked_log_softmax,ppo_state_loss
from .relative_actor import RelativeCorrection,rank_response,actor_only_package,load_actor_only_package
from .critic_interfaces import C1Critic,normalized_component_loss

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'rc3_critic_diagnostics_20261111/ACTOR_CANDIDATES.csv.gz'


class FrozenFixtureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fixture=nn.Linear(2,2)

    def forward(self,*args,**kwargs):
        raise AssertionError('No real or synthetic encoder inference is needed')


def check_saved_logits(charge):
    """Use only the first eight immutable state groups, never select by cost.

    The 206 features here are explicitly synthetic zero tensors. This verifies
    the algebraic zero-head contract on saved SL logits, not real-feature
    reconstruction, a deployed route, or a new planning-performance test.
    """
    charge('source_reads',path=SOURCE.relative_to(ROOT).as_posix())
    fixtures=[]
    with gzip.open(SOURCE,'rt',encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream)
        groups=itertools.groupby(reader,key=lambda r:(r['trajectory_id'],r['state_index']))
        for key,group in itertools.islice(groups,8):
            rows=list(group);M=max(int(r['candidate_local_index']) for r in rows)+2
            z=torch.full((1,M),-torch.inf,dtype=torch.float32)
            mask=torch.zeros((1,M),dtype=torch.bool)
            for row in rows:
                j=int(row['candidate_local_index']);z[0,j]=float(row['base_logit']);mask[0,j]=True
            x=torch.zeros((1,M,206),dtype=torch.float32)
            x[~mask]=torch.nan
            wrapper=LookaheadResidualActor(FrozenFixtureEncoder(),distances_enabled=True)
            wrapper.correction=RelativeCorrection();wrapper.train()
            assert not wrapper.encoder.training and all(not p.requires_grad for p in wrapper.encoder.parameters())
            result=wrapper.correction(z,x,mask)
            assert torch.equal(result,z) and torch.equal(torch.isfinite(result),mask)
            identity={'path':'saved-logit-contract/no-base-loading','sha256':'1'*64}
            package=actor_only_package(wrapper.correction,identity)
            restored=load_actor_only_package(package,expected_base_identity=identity)
            restored_logits=restored(z,x,mask)
            assert torch.equal(result,restored_logits)
            action=int(z.argmax(-1));report=rank_response(z,z,restored_logits,mask,
                actions=torch.tensor([action]),collection_logits=z)
            fixtures.append(dict(trajectory_id=key[0],state_index=int(key[1]),
                legal_candidates=int(mask.sum()),zero_SL_exact=True,mask_exact=True,
                package_reload_logits_exact=True,synthetic_features=True,
                encoder_calls=0,complete_routes_generated=0,rank_report=report))
    assert len(fixtures)==8
    return dict(status='PASS',selection='first8 saved state groups in byte order; no cost selection',
                real_SL_weight_loads=0,real_feature_reconstructions=0,fixtures=fixtures)


def check_loss_interfaces():
    """Exercise stateless old PPO loss and C1 adapters without a trainer/Adam."""
    torch.manual_seed(11622026)
    actor=RelativeCorrection();critic=C1Critic()
    features=torch.arange(2*3*206,dtype=torch.float32).reshape(2,3,206)/1000
    z=torch.tensor([[.1,.2,.0],[.2,.0,-.2]])
    mask=torch.tensor([[True,True,False],[True,True,True]])
    before=actor(z,features,mask).detach();actions=torch.tensor([0,2])
    oldlp=masked_log_softmax(before,mask).gather(1,actions[:,None]).squeeze(1)
    advantage=torch.tensor([.7,-.4]).detach()
    objective,ratio,_=ppo_state_loss(masked_log_softmax(actor(z,features,mask),mask),
        oldlp,advantage,masked_log_softmax(z,mask).detach(),mask,actions)
    assert torch.equal(ratio,torch.ones_like(ratio))
    objective.sum().backward()
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in actor.parameters())
    assert all(p.grad is None for p in critic.parameters())
    for p in actor.parameters():p.grad=None
    prediction=critic(torch.zeros(2,268),5)
    targets=dict(scale=5,task_count=5,raw_remaining_L=[4.,2.],raw_remaining_K=[2.,1.],
                 terminal_H_credit=[1.,1.],returns=[-.9,-.35])
    loss=normalized_component_loss(prediction,targets)
    compatibility=remaining_cost_loss(prediction.as_training_dict(),targets,5)
    assert torch.allclose(loss,compatibility,atol=1e-6,rtol=1e-6)
    loss.mean().backward()
    assert all(p.grad is None for p in actor.parameters())
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in critic.parameters())
    return dict(status='PASS',Actor_optimizer_calls=0,Critic_optimizer_calls=0,
        PPO_loss_finite=True,initial_ratio_one=True,C1_raw_adapter_loss_equal=True,
        gradients_finite=True,Actor_Critic_parameter_gradients_disjoint=True,
        batch_features_and_targets='synthetic only',
        old_trainer_instantiated=False,production_loader_invoked=False)
