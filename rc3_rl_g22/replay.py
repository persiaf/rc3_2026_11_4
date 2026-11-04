"""Zero-update inference replay on the actual saved policy inputs, no search."""
from __future__ import annotations
import numpy as np
from .common import ROOT,WORKSPACE,read,write,sha
from .evaluate import model_identity,greedy
from .audit import load_evidence

def arrays(raw):
    raw=dict(raw)
    raw['mask']=np.asarray(raw['mask'],dtype=bool)
    for key,width in (('features',30),('task_features',9),('candidate_features',14)):
        raw[key]=np.asarray(raw[key],dtype=np.float32).reshape(-1,width)
    raw['nav_distance']=np.asarray(raw['nav_distance'],dtype=np.float32)
    for key in ('task_action','candidate_action','task_ids','candidate_nodes','edge_task','edge_candidate','nav_source','nav_target'):
        raw[key]=np.asarray(raw[key],dtype=np.int64)
    return raw

def main():
    from rc3_rl_g2.checkpoints import load_model
    from rc3_rl_g2.train_ppo import configure
    configure(2026110401)
    p=read(ROOT/'outputs/protocol.json');rows=read(ROOT/'outputs/evaluation/results.json');records=[]
    for method,identity in p['models'].items():
        if sha(WORKSPACE/identity['path'])!=identity['sha256']:raise ValueError('Checkpoint changed')
        model,_=load_model(WORKSPACE/identity['path'],'cpu');model.eval();initial=model_identity(model)
        count=0;max_score_error=0.;max_probability_error=0.;same_actions=True
        for row in rows:
            if row['method']!=method:continue
            path=ROOT/f"outputs/evaluation/case_{row['case_id']:02d}_{method}/evidence.json.gz"
            evidence=load_evidence(path)
            for item in evidence['policy_observations']:
                action,scores=greedy(model,arrays(item['observation']));count+=1
                same_actions=same_actions and action==item['action_id']
                saved=item['scores']
                for before,after in zip(saved['logits'],scores['logits']):
                    if (before is None)!=(after is None):raise ValueError('Replay mask differs')
                    if before is not None:max_score_error=max(max_score_error,abs(after-before))
                max_probability_error=max(max_probability_error,float(np.max(np.abs(np.asarray(saved['probabilities'])-scores['probabilities']))))
                if not np.isclose(saved['value'],scores['value'],rtol=1e-6,atol=1e-6):raise ValueError('Value output differs')
        unchanged=model_identity(model)==initial
        passed=same_actions and unchanged and max_score_error<1e-5 and max_probability_error<1e-6
        records.append(dict(method=method,state_evaluations=count,actions_equal=same_actions,max_logit_error=max_score_error,
            max_probability_error=max_probability_error,model_unchanged=unchanged,passed=passed))
        if not passed:raise ValueError('Recorded greedy inference did not replay')
    result=dict(all_passed=True,models=records,state_evaluations=sum(r['state_evaluations'] for r in records),
        optimizer_updates=0,environment_actions=0,search_proposals=0,
        scope='Repeated model/input evaluations from 24 saved episodes; not additional plans, scenes, or reward queries')
    write(ROOT/'outputs/inference_replay.json',result);print(result)

if __name__=='__main__':main()
