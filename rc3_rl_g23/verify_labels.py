"""Reexecute only registered training branches, charge all calls, save full ledger evidence."""
import gzip,pickle,time,copy
import torch
from .core import ROOT,Ledger,read,write,digest,sha
from .trial import make,choose,follow,stop
from .learning import Value,observation

def compact(env,prefix_steps,prefix_reward,continuation):
    transitions=[dict(action=t['action'],reward=t['reward'],quality=t['info']['quality_reward'],work=t['info']['work_units'],
        current_J=t['info']['current_cost'],best_J=t['info']['best_cost']) for t in env.transitions[prefix_steps:]]
    events=copy.deepcopy(env.events)
    actual=env.total_reward-prefix_reward
    if abs(sum(t['reward'] for t in transitions)+sum(t['reward'] for t in events)-actual)>1e-7:raise ValueError('Suffix ledger mismatch')
    return dict(continuation_policy=continuation,remaining_return=actual,transitions=transitions,events=events,total_return=env.total_reward,work=env.work_used,
        returned_J=env.returned_certificate.cost,route=env.returned_certificate.route,validation=env.returned_certificate.plan['validation'],
        finalization=env.finalization,anchor_J=env.anchor_cost)

def main():
    from rc3_rl_g2.train_ppo import configure
    configure(2300);p=read(ROOT/'outputs/protocol.json');old=read(ROOT/'outputs/resources.json')
    if not read(ROOT/'outputs/run_receipt.json')['completed']:raise ValueError('Trial incomplete')
    if (ROOT/'outputs/verified_labels.json.gz').exists():raise FileExistsError('Verification already saved')
    authority=read(ROOT/'outputs/verification_protocol.json');authority['script_sha256']=sha(__file__)
    write(ROOT/'outputs/verification_execution.json',authority)
    ledger=Ledger(p['limits']);ledger.counts=old['counts'].copy();ledger.started=time.perf_counter()-old['elapsed_s'];ledger.phase='verification'
    before=ledger.counts.copy();proof=[]
    for i,seed in enumerate(p['seeds']):
        for phase in (0,1):
            with gzip.open(ROOT/f'outputs/seed{seed}_round{phase}.pkl.gz','rb') as f:data=pickle.load(f)
            policy=None
            if phase:
                payload=torch.load(ROOT/f'outputs/model_{seed}_init.pt',map_location='cpu',weights_only=False)
                policy=Value();policy.load_state_dict(payload['state']);policy.eval()
                if digest({k:v.detach().numpy() for k,v in policy.state_dict().items()})!=data[0]['policy_id']:raise ValueError('Continuation snapshot mismatch')
            for case in p['splits']['train'][i*4:(i+1)*4]:
                parent=make(case,ledger,p)
                for index in range(2):
                    if parent.done:break
                    raw=observation(parent);key=digest(raw)
                    rows=[s for s in data if s['case']['seed']==case['seed'] and digest(s['observation'])==key]
                    if not rows:raise ValueError('Saved full state not reproduced')
                    baseline=parent.fork();prefix=parent.total_reward;steps=parent.steps;qstop=stop(baseline)-prefix
                    for item in rows:
                        child=parent.fork();_,r,done,_,info=child.step(item['action']);one=child.fork()
                        one_return=stop(one)-prefix
                        if phase:
                            static=child.fork();control_return=follow(static,None)-prefix
                        else:static=None;control_return=None
                        multi_return=follow(child,policy)-prefix
                        observed=dict(one=one_return-qstop,multi=multi_return-qstop,static=multi_return-qstop if control_return is None else control_return-qstop,qstop=qstop,step_reward=r)
                        if any(abs(observed[k]-item[k])>1e-7 for k in observed):raise ValueError('Training value label did not reproduce')
                        proof.append(dict(seed=seed,round=phase,case=case['case_id'],action=item['action'],state_key=key,
                            labels=observed,stop=compact(baseline,steps,prefix,'STOP'),one=compact(one,steps,prefix,'STOP after first action'),
                            multi=compact(child,steps,prefix,item['policy_id']),static=None if static is None else compact(static,steps,prefix,'rule'),
                            evaluated_policy=item['policy_id'],h=item['remaining_h'],remaining_work=item['remaining_work']))
                    if index==0:
                        for _ in range(2):
                            if parent.done:break
                            parent.step(choose(parent,policy))
                stop(parent)
    with gzip.open(ROOT/'outputs/verified_labels.json.gz','wt',encoding='utf-8') as f:
        import json
        json.dump(proof,f,ensure_ascii=False,allow_nan=False)
    result=dict(all_passed=True,training_rows=len(proof),before_counts=before,after_counts=ledger.counts,
        verification_calls={k:ledger.counts[k]-before[k] for k in before},optimizer_updates=0,
        scope='Actual additional branch checks, counted inside original caps; all value targets reproduce and retain transition/STOP/finalization ledgers')
    write(ROOT/'outputs/label_verification.json',result);print(result)

if __name__=='__main__':main()
