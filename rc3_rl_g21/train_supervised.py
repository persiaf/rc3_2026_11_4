"""Bounded evidence-only supervised diagnostics; never steps an environment."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from rc3_rl_g2.teacher_dataset import load_teacher
from rc3_rl_g2.checkpoints import load_model,save_model,sha256
from rc3_rl_g2.tools.teacher_fit_diagnostic import sample_metrics
from rc3_rl_g2.train_ppo import configure

ROOT=Path(__file__).resolve().parent
G2=ROOT.parent/'rc3_rl_g2'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def checked_samples(split='initial'):
    path=G2/f'outputs/teacher_{split}/teacher.pkl'
    if sha256(path)!=read(path.parent/'summary.json')['teacher_sha256']:raise ValueError('Teacher changed')
    return load_teacher(path)['samples']

def root_half_weights(samples):
    groups=defaultdict(list)
    for i,s in enumerate(samples):groups[s['scene_id']].append(i)
    weights=np.zeros(len(samples),dtype=np.float64)
    for indices in groups.values():
        roots=[i for i in indices if not samples[i]['path']]
        descendants=[i for i in indices if samples[i]['path']]
        if len(roots)!=1 or not descendants:raise ValueError('One root and successors required per scene')
        weights[roots[0]]=.5/len(groups)
        routes=defaultdict(list)
        for i in descendants:routes[tuple(samples[i]['current_route'])].append(i)
        for own in routes.values():
            for i in own:weights[i]=.5/(len(groups)*len(routes)*len(own))
    if not np.isclose(weights.sum(),1):raise AssertionError('Weight mass mismatch')
    return weights

def supervision_loss(logits,samples,auxiliary=False):
    losses=[];aux=[]
    for j,s in enumerate(samples):
        n=len(s['evidence_mask']);e=torch.as_tensor(s['evidence_mask'],dtype=torch.bool,device=logits.device)
        ties=torch.as_tensor(s['ties'],dtype=torch.bool,device=logits.device)
        if (ties&~e).any() or not ties.any():raise ValueError('Invalid teacher evidence')
        lp=logits[j,:n].masked_fill(~e,-torch.inf).log_softmax(0)
        losses.append(-lp[ties].mean())
        if auxiliary and not s['path']:
            if s['teacher_depth']!=2:raise ValueError('Gain calibration is root-depth2 only')
            known=e.clone();known[int(s['stop_id'])]=False
            target=torch.as_tensor(s['advantages_vs_stop'],dtype=logits.dtype,device=logits.device)
            predicted=logits[j,:n]-logits[j,int(s['stop_id'])]
            aux.append(torch.nn.functional.smooth_l1_loss(predicted[known],target[known]))
    ce=torch.stack(losses).mean()
    gain=torch.stack(aux).mean() if aux else ce.new_zeros(())
    return ce+gain,ce,gain

@torch.no_grad()
def diagnose(model,samples):
    rows=[]
    for start in range(0,len(samples),16):
        batch=samples[start:start+16];logits,_=model([s['observation'] for s in batch])
        for j,s in enumerate(batch):
            row=sample_metrics(logits[j].cpu().numpy(),s)
            evidence=np.asarray(s['evidence_mask']);q=np.asarray(s['q_values']);stop=s['stop_id']
            mask=evidence.copy();mask[stop]=False;delta=float(q[mask].max()-q[stop])
            row.update(sample_index=start+j,scene_id=s['scene_id'],is_root=not s['path'],
                teacher_depth=s['teacher_depth'],delta=delta,known_positive=delta>.001,
                clear_positive=delta>=.1,clear_stop=delta<=-.02)
            for prefix in ['full','evidence']:
                a=row[prefix+'_greedy_action'];known=bool(evidence[a])
                row[prefix+'_known_positive_continue']=bool(known and a!=stop and q[a]-q[stop]>.001)
                row[prefix+'_selected_gain']=float(q[a]-q[stop]) if known else None
            rows.append(row)
    groups={}
    for name,predicate in [('all',lambda r:True),('roots',lambda r:r['is_root']),
        ('successors',lambda r:not r['is_root']),('positive_roots',lambda r:r['is_root'] and r['known_positive']),
        ('clear_stop_roots',lambda r:r['is_root'] and r['clear_stop'])]:
        own=[r for r in rows if predicate(r)]
        if not own:continue
        metrics={}
        for key in ['evidence_greedy_near_best_hit','full_legal_greedy_observed_near_best_hit',
            'full_greedy_unknown','full_greedy_stop','stop_probability','unknown_probability_mass',
            'evidence_known_positive_continue','full_known_positive_continue',
            'evidence_greedy_Q_gap','observed_full_greedy_Q_gap']:
            vals=[r[key] for r in own if r[key] is not None]
            metrics[key]=dict(mean=float(np.mean(vals)) if vals else None,n=len(vals),
                count=sum(bool(v) for v in vals) if vals and isinstance(vals[0],bool) else None)
        groups[name]=dict(n=len(own),metrics=metrics)
    return dict(groups=groups,rows=rows,new_environment_queries=0)

def toy_gate(diagnostic):
    rows=diagnostic['rows'];positive=[r for r in rows if r['clear_positive']]
    clear_stop=[r for r in rows if r['clear_stop']]
    composition=(len(rows)==16 and len(positive)==8 and len(clear_stop)==8
        and all(r['is_root'] and r['teacher_depth']==2 for r in rows)
        and len({r['scene_id'] for r in rows})==16
        and all(bool(r['clear_positive']) != bool(r['clear_stop']) for r in rows))
    near=sum(r['evidence_greedy_near_best_hit'] for r in rows)
    correct=sum(r['evidence_known_positive_continue'] for r in positive)
    return dict(passed=composition and near>=15 and correct>=7,composition_valid=composition,
        evidence_near_best=near,positive_continue=correct,n=16,positive_n=8,
        scope='Memorization on queried finite-depth labels; not full-action deployment or generalization')

def run(stage,variant=None):
    p=read(ROOT/'outputs/protocol.json');digest=sha256(ROOT/'outputs/protocol.json')
    if digest!=(ROOT/'outputs/protocol.sha256').read_text().strip():raise ValueError('Protocol changed')
    configure(p['seed']);all_samples=checked_samples();selection=read(ROOT/'outputs/selection.json')
    if sha256(ROOT/'outputs/selection.json')!=p['selection_sha256']:raise ValueError('Selection changed')
    if stage=='toy':
        name='toy16';samples=[all_samples[i] for i in selection['sample_indices']]
        settings=p['toy'];init=G2/'outputs/training/bc_initial.pt';weights=None;auxiliary=False
    elif stage=='sampling':
        if not read(ROOT/'outputs/toy16/receipt.json')['gate']['passed']:raise ValueError('Toy learnability gate failed; no downstream training')
        if variant not in ('original','root_half'):raise ValueError('Sampling variant required')
        name='sampling_'+variant;samples=all_samples;settings=p['sampling'];init=G2/'outputs/training/bc_initial.pt';auxiliary=False
        weights=np.array([s['weight'] for s in samples]) if variant=='original' else root_half_weights(samples)
    elif stage=='gain':
        if not read(ROOT/'outputs/toy16/receipt.json')['gate']['passed']:raise ValueError('Toy gate failed')
        if variant not in ('control','gain'):raise ValueError('Gain-control variant required')
        if not (ROOT/'outputs/gain_authorization.json').exists():raise ValueError('Conditional gate decision must be recorded first')
        name='gain_'+variant;samples=all_samples;settings=p['gain'];init=ROOT/'outputs/sampling_root_half/model.pt'
        weights=root_half_weights(samples);auxiliary=variant=='gain'
    else:raise ValueError('Unknown stage')
    out=ROOT/'outputs'/name
    if out.exists():raise FileExistsError('Preserve existing run: '+str(out))
    out.mkdir(parents=True)
    if weights is not None:weights=weights/weights.sum()
    model,parent=load_model(init,'cpu');model.train()
    if sha256(init)!=(p['initial_bc_sha256'] if stage!='gain' else read(ROOT/'outputs/sampling_root_half/receipt.json')['checkpoint_sha256']):raise ValueError('Initialization changed')
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.Adam(model.parameters(),lr=settings['lr']);rng=np.random.default_rng(p['seed']+91)
    registration=dict(stage=stage,variant=variant,settings=settings,initial_sha256=sha256(init),protocol_sha256=digest,
        trainer_sha256=sha256(__file__),initial_teacher_sha256=sha256(G2/'outputs/teacher_initial/teacher.pkl'),
        sampling_weights=weights.tolist() if weights is not None else None,full_batch=weights is None,new_environment_queries=0)
    write(out/'request.json',registration)
    history=[];root_exposures=0;total_exposures=0
    initial=diagnose(model,samples);write(out/'diagnostic_0000.json',initial)
    started=time.perf_counter();last_diagnostic=None
    with (out/'updates.jsonl').open('w',encoding='utf-8') as log:
        for update in range(1,settings['updates']+1):
            batch=samples if weights is None else [samples[int(i)] for i in rng.choice(len(samples),settings['batch_size'],p=weights)]
            root_exposures+=sum(not s['path'] for s in batch);total_exposures+=len(batch)
            logits,_=model([s['observation'] for s in batch]);loss,ce,gain=supervision_loss(logits,batch,auxiliary)
            if not torch.isfinite(loss):raise AssertionError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.5)
            if not torch.isfinite(norm):raise AssertionError('Nonfinite gradients')
            optimizer.step();row=dict(update=update,loss=float(loss.detach()),ce=float(ce.detach()),gain=float(gain.detach()),grad_norm=float(norm),root_exposures=root_exposures,total_exposures=total_exposures)
            log.write(json.dumps(row)+'\n');log.flush()
            if update%settings['check_every']==0 or update==settings['updates']:
                last_diagnostic=diagnose(model,samples);write(out/f'diagnostic_{update:04d}.json',last_diagnostic)
                row['root_metrics']=last_diagnostic['groups'].get('roots');history.append(row)
                print(json.dumps(dict(stage=name,update=update,loss=row['loss'],
                    evidence_hits=sum(r['evidence_greedy_near_best_hit'] for r in last_diagnostic['rows']),
                    full_positive=sum(r['full_known_positive_continue'] for r in last_diagnostic['rows'] if r['is_root'] and r['known_positive']),
                    gate=toy_gate(last_diagnostic) if stage=='toy' else None)),flush=True)
    elapsed=time.perf_counter()-started
    if not all(torch.equal(v,model.state_dict()[k]) for k,v in before.items() if k.startswith('critic.')):raise AssertionError('Actor-only training changed critic head')
    sha=save_model(out/'model.pt',model,protocol_sha256=digest,parent_bc_sha256=sha256(init),
        g21_trainer_sha256=sha256(__file__),updates=settings['updates'],stage=stage,variant=variant,optimizer=optimizer.state_dict())
    extra=diagnose(model,checked_samples('extra'));write(out/'extra_existing_teacher_diagnostic.json',extra)
    receipt=dict(completed=True,stage=stage,variant=variant,updates=settings['updates'],elapsed_s=elapsed,
        checkpoint_sha256=sha,request_sha256=sha256(out/'request.json'),protocol_sha256=digest,
        final_diagnostic=last_diagnostic['groups'],gate=toy_gate(last_diagnostic) if stage=='toy' else None,
        parameter_delta_l2=float(torch.sqrt(sum((model.state_dict()[k]-v).float().square().sum() for k,v in before.items()))),
        critic_head_unchanged=True,new_environment_queries=0,
        root_exposures=root_exposures,total_exposures=total_exposures,
        interpretation='Finite-label supervised fitting only. Extra32 is previously used G2 development teacher data, not a new unseen confirmation set.')
    write(out/'receipt.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='final_diagnostic'}),flush=True)
    return receipt

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['toy','sampling','gain'],required=True)
    parser.add_argument('--variant');args=parser.parse_args();run(args.stage,args.variant)
