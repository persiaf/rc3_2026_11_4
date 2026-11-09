"""Natural sampling reused verbatim from GL-v4 B except the frozen baseline."""
import math
import numpy as np
import torch
from rc3_global_local.environment import Environment
from rc3_gl_v4_20261109.mechanism import AuditTrainer,RecordingTerminal as OldRecordingTerminal,save_gradient_and_weights
from .common import require
from .objective import fixed_reference_objective
from .reference import rng_guard


class RecordingTerminal(OldRecordingTerminal):
    def state_dict(self):
        return dict(super().state_dict(),reference_state=self.reference.state_dict())


class GreedyBaselineTrainer(AuditTrainer):
    def _backward(self,loss):
        previous={n:None if p.grad is None else p.grad.detach().clone() for n,p in self.model.named_parameters()}
        super()._backward(loss)
        delta={n:float((p.grad.detach().double()-(0 if previous[n] is None else previous[n].double())).norm())
            for n,p in self.model.named_parameters() if p.grad is not None}
        self.instance_gradient_norms.append(dict(batch_scaled_l2=math.sqrt(sum(x*x for x in delta.values())),per_parameter_l2=delta))

    def _finish_update(self,diagnostics):
        previous={n:p.detach().clone() for n,p in self.model.named_parameters()}
        result=super()._finish_update(diagnostics)
        steps={n:float((p.detach().double()-previous[n].double()).norm()) for n,p in self.model.named_parameters()}
        total=math.sqrt(sum(x*x for x in steps.values()));norm=math.sqrt(sum(float(x.double().square().sum()) for x in previous.values()))
        result.update(instance_gradient_norms=self.instance_gradient_norms,Adam_step_l2=total,Adam_relative_step_l2=total/max(norm,1e-30),
            Adam_per_parameter_step_l2=steps,Adam_internal_steps=[int(s['step']) for s in self.optimizer.state.values()])
        return result

    def policy_instance_loss(self,view,update,slot):
        reference=rng_guard(self.streams,lambda:self.reference(view))
        rows=[];log_sums=[];device=next(self.model.parameters()).device
        for role in ('primary','forced_low','forced_high'):
            env=Environment(view,self._terminal);generator=self.streams.torch(self.phase,update,slot,role,device=str(device))
            actions=[];logps=[];entropies=[];masks=[];counts=[]
            unique=0
            if not env.done:
                unique=int(env.mask().sum());require(unique>0,'No legal first action')
                encoded=self._encode(view)
                while not env.done:
                    mask_np=env.mask();masks.append((np.flatnonzero(mask_np)+1).tolist());counts.append(int(mask_np.sum()))
                    logits,legal=self._score(env,encoded);logs=torch.log_softmax(logits,0);probs=torch.softmax(logits,0)
                    chosen=torch.multinomial(probs,1,generator=generator).squeeze(0)
                    node=int(chosen)+1;lp=logs[chosen];entropy=-(probs[legal]*logs[legal]).sum()
                    env.step(node);actions.append(node);logps.append(lp);entropies.append(entropy);self._inc('environment_actions')
            self._inc('policy_trajectories')
            logsum=sum(logps,torch.zeros((),device=device));cost=float(env.result['J']);ret=float(sum(env.rewards))
            require(math.isfinite(cost) and math.isfinite(ret) and abs(ret+cost)<1e-7,'Reward/cost inconsistency')
            require(not actions or logps[0] is not None,'Policy first action logprob omitted')
            rows.append(dict(role=role,sampling_seed=self.streams.seed(self.phase,update,slot,role),forced_first=None,
                unique_legal_first=unique,actions=actions,legal_counts=counts,masks_local_nodes=masks,
                logprobs=[float(v.detach()) for v in logps],logprob_sum=float(logsum.detach()),
                entropies=[float(v.detach()) for v in entropies],rewards=[float(v) for v in env.rewards],
                return_=ret,J=cost,raw_J=float(env.result.get('raw_J',cost)),raw_route=[0,*actions,0],
                final_route=list(env.result.get('route',[0,*actions,0])),terminal_cache_hit=bool(env.result.get('cache_hit',False)),
                H_s=float(env.result.get('H_s',0.)),export_s=float(env.result.get('export_s',0.)),validation_s=float(env.result.get('validation_s',0.))))
            log_sums.append(logsum)
        returns=torch.tensor([x['return_'] for x in rows],dtype=torch.float64,device=device)
        baseline=torch.full_like(returns,-float(reference['J'])).detach()
        loss,advantages=fixed_reference_objective(log_sums,[x['J'] for x in rows],reference['J'])
        require(bool(torch.isfinite(loss)),'Nonfinite IID policy-gradient loss')
        info=dict(problem=view.identity,parent=view.parent.identity,update=int(update),slot=int(slot),
            empty=not len(view.C),weight_version=self.weight_version,rows=rows,baseline=baseline.cpu().tolist(),
            advantages=advantages.cpu().tolist(),mixture_weights=[1.,.25,.25],loss=float(loss.detach()),gamma=1.,
            reference=reference,objective='full natural-start weighted REINFORCE, detached frozen SL256 greedy+H return baseline')
        return loss,self.attach_terminal_metrics(info)
