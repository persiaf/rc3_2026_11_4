"""Natural starts with ORIGINAL1/.25/.25 detached-LOO; original A math reused."""
from collections import deque
import math
import numpy as np
import torch
from rc3_global_local.environment import Environment
from rc3_global_local.policy import tensor_hash
from rc3_global_local_trainer.engine import Trainer,InconclusiveBatch
from rc3_global_local_trainer.runner import BudgetedTerminal
from .common import require,torch_new,digest,label


class RecordingTerminal:
    def __init__(self,ledger,folder,h_identity,clock,log):
        self.log=log;self.clock=clock;self.folder=folder
        self.inner=BudgetedTerminal(ledger,folder/'terminal_plans',h_identity,{'RL':1536},clock,log,cache_size=128)
        self.recent=deque(maxlen=12);self.request_count=0

    def __call__(self,p,raw):
        self.clock.check();require(self.request_count<1536,'Training terminal request cap')
        self.request_count+=1;ordinal=self.request_count
        self.log(dict(event='terminal_request_predebit',request=ordinal,problem=p.identity,raw_route=list(raw)))
        result=self.inner(p,raw)
        row=dict(request=ordinal,problem=p.identity,raw_route=list(raw),
            **{k:result[k] for k in ('J','L','K','raw_J','raw_L','raw_K')},cache_hit=result['cache_hit'])
        self.recent.append(row);self.log(dict(event='terminal_request_complete',**row))
        return result

    def state_dict(self):return dict(request_count=self.request_count,original_terminal=self.inner.state_dict())


class AuditTrainer(Trainer):
    def _finish_update(self,diagnostics):
        self.audit_pre_grad={n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in self.model.named_parameters()}
        result=super()._finish_update(diagnostics)
        self.audit_post_grad={n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in self.model.named_parameters()}
        return result

    def policy_instance_loss(self,view,update,slot):
        loss,info=super().policy_instance_loss(view,update,slot)
        return loss,self.attach_terminal_metrics(info)

    def attach_terminal_metrics(self,info):
        if isinstance(self.terminal,RecordingTerminal):
            recent=list(self.terminal.recent)[-3:]
            require(len(recent)==3,'Wrong terminal trace count')
            for row,result in zip(info['rows'],recent):
                require(result['problem']==info['problem'] and result['raw_route']==row['raw_route'] and result['J']==row['J'],'Terminal trace mismatch')
                row.update(terminal_request=result['request'],L=result['L'],K=result['K'],raw_L=result['raw_L'],raw_K=result['raw_K'])
        return info


class NaturalTrainer(AuditTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # Explicit new recipe, outside and without altering the frozen Trainer.
        assert self.config['mixture_weights']==[1.,.25,.25]

    def policy_instance_loss(self,view,update,slot):
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
        baseline=((returns.sum()-returns)/2).detach();advantages=(returns-baseline).detach()
        loss=sum(-w*a*logp for w,a,logp in zip(self.config['mixture_weights'],advantages,log_sums))/sum(self.config['mixture_weights'])
        require(bool(torch.isfinite(loss)),'Nonfinite IID policy-gradient loss')
        info=dict(problem=view.identity,parent=view.parent.identity,update=int(update),slot=int(slot),
            empty=not len(view.C),weight_version=self.weight_version,rows=rows,baseline=baseline.cpu().tolist(),
            advantages=advantages.cpu().tolist(),mixture_weights=[1.,.25,.25],loss=float(loss.detach()),gamma=1.,
            objective='three independent natural starts, full-logprob, original weighted detached-LOO REINFORCE')
        return loss,self.attach_terminal_metrics(info)


def save_gradient_and_weights(folder,trainer,update):
    state={k:v.detach().cpu().clone() for k,v in trainer.model.state_dict().items()}
    receipt=torch_new(folder/f'update_{update:04}.pt',dict(schema='rc3-pg-full-gradient-weight-audit-v1',
        update=update,model=state,gradient_before_clip=trainer.audit_pre_grad,gradient_after_clip=trainer.audit_post_grad,
        model_tensor_sha256=tensor_hash(trainer.model)))
    return receipt
