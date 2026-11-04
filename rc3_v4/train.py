"""Same architecture/labels/budget; only the graph information is ablated."""
import argparse
import copy
import hashlib
import json
import os
import pickle
import shutil
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch.nn import functional as F
from rc3_baseline.baseline import write_json,write_csv
from rc3_fast.solver import prepare,route_objective
from rc3_v3.environment import Env
from rc3_v3.model import select_action
from .features import observation,collate,local_graphs
from .model import make_policy,batch_torch,save_policy,export_numpy


def loss_for(model,rows,scale,pos_weight,device):
    batch=batch_torch(collate(rows),device)
    mask=batch['mask'];gain=torch.zeros(mask.shape,device=device)
    for i,r in enumerate(rows):gain[i,:len(r['gains'])]=torch.as_tensor(r['gains'],device=device)
    gain=gain.clamp(min=0)*mask;positive=(gain>1e-7).float()
    weights=gain.sqrt();sums=weights.sum(1);target=weights/sums.clamp(min=1e-8)[:,None]
    logits,pred,binary=model(batch)
    logp=torch.log_softmax(logits[:,:-1],-1)
    rank=-(target*torch.where(mask,logp,0.)).sum()/ (sums>1e-8).sum().clamp(min=1)
    stop=F.binary_cross_entropy_with_logits(logits[:,-1],(sums<=1e-8).float())
    reg=(F.smooth_l1_loss(pred[mask],(gain/scale)[mask],reduction='none')*(1+4*positive[mask])).mean()
    binary_loss=F.binary_cross_entropy_with_logits(binary[mask],positive[mask],pos_weight=torch.tensor(pos_weight,device=device))
    return rank+stop+reg+.2*binary_loss


def validate(model,problems,routes,device):
    envs=[Env(p,8,initial_route=r) for p,r in zip(problems,routes)]
    stopped=[False]*len(envs);cache={};stop_info=[]
    model.eval()
    while True:
        active=[];rows=[]
        for i,e in enumerate(envs):
            base,mask=e.observe()
            if stopped[i] or e.done or not mask.any():continue
            key=(i,tuple(e.route))
            if key not in cache:cache[key]=local_graphs(e.problem,e.route,e.windows)
            active.append(i);rows.append(dict(features=base,mask=mask,graphs=cache[key]))
        if not active:break
        for start in range(0,len(active),4):
            sub=rows[start:start+4];b=batch_torch(collate(sub),device)
            with torch.no_grad():logits=model(b)[0].cpu().numpy()
            for j,i in enumerate(active[start:start+4]):
                n=len(sub[j]['mask']);action=select_action(np.r_[logits[j,:n],logits[j,-1]],sub[j]['mask'])
                if action is None:stopped[i]=True;stop_info.append(dict(scene=i,step=envs[i].steps))
                else:envs[i].step(action)
    gains=[100*(e.reference_cost-route_objective(e.problem,e.route))/max(e.reference_cost,1e-8) for e in envs]
    return dict(mean_gain_pct=float(np.mean(gains)),mean_calls=float(np.mean([e.steps for e in envs])),
        gains_pct=gains,stopped=sum(stopped),stop_events=stop_info)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('rc3_v4/data'))
    p.add_argument('--output',type=Path,default=Path('rc3_v4/training'))
    p.add_argument('--epochs',type=int,default=80);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--seeds',type=int,nargs='+',default=[17,29,43]);p.add_argument('--variants',nargs='+',default=['control','structured'])
    p.add_argument('--device',default='cuda');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    with (a.data/'data.pkl').open('rb') as f:data=pickle.load(f)
    if data['graph_schema']!='rc3-v4-local-graph-v1':raise ValueError('schema mismatch')
    rows=data['states'];flat=np.concatenate([r['features'][r['mask']] for r in rows])
    mean,std=flat.mean(0),flat.std(0).clip(.05)
    gains=np.concatenate([r['gains'][r['mask']] for r in rows]);positive=gains>1e-7
    scale=max(float(np.quantile(gains[positive],.9)),.05)
    pos_weight=min(float((~positive).sum()/positive.sum()),8.)
    started=time.perf_counter();problems=[prepare(s) for s in data['scenes']['validation']]
    histories=[];runs=[];updates=0
    for variant in a.variants:
        for seed in a.seeds:
            torch.manual_seed(seed);rng=np.random.default_rng(seed)
            model=make_policy(mean,std,variant=variant).to(a.device)
            opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
            best=-np.inf;best_state=None;best_stats=None;best_epoch=None
            for epoch in range(a.epochs+1):
                losses=[]
                if epoch:
                    model.train()
                    for ids in np.array_split(rng.permutation(len(rows)),int(np.ceil(len(rows)/a.batch_size))):
                        loss=loss_for(model,[rows[i] for i in ids],scale,pos_weight,a.device)
                        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                        losses.append(float(loss.detach()));updates+=1
                if epoch%10==0 or epoch==a.epochs:
                    stats=validate(model,problems,data['initial_routes']['validation'],a.device)
                    if stats['mean_gain_pct']>best+1e-10 or (abs(stats['mean_gain_pct']-best)<=1e-10 and stats['mean_calls']<best_stats['mean_calls']):
                        best=stats['mean_gain_pct'];best_state=copy.deepcopy(model.state_dict());best_epoch=epoch;best_stats=stats
                    row=dict(variant=variant,seed=seed,epoch=epoch,loss=float(np.mean(losses)) if losses else None,
                        val_gain_pct=stats['mean_gain_pct'],val_calls=stats['mean_calls'],val_stops=stats['stopped'],updates=updates,elapsed_s=time.perf_counter()-started)
                    histories.append(row);write_csv(a.output/'history.csv',histories)
                    print(f'{variant} seed={seed} epoch={epoch} gain={stats["mean_gain_pct"]:.5f}% best={best:.5f}% seconds={time.perf_counter()-started:.0f}',flush=True)
            model.load_state_dict(best_state);name=f'{variant}_seed{seed}.pt'
            save_policy(a.output/name,model,seed=seed,epoch=best_epoch,validation=best_stats)
            export_numpy(a.output/name,(a.output/name).with_suffix('.npz'))
            runs.append(dict(variant=variant,seed=seed,epoch=best_epoch,checkpoint=name,validation=best_stats,
                parameter_count=sum(v.numel() for v in model.parameters())))
            write_json(a.output/'progress.json',runs)
    selected={}
    for variant in a.variants:
        chosen=max([r for r in runs if r['variant']==variant],key=lambda r:(r['validation']['mean_gain_pct'],-r['validation']['mean_calls']))
        selected[variant]=chosen
        for ext in ('.pt','.npz'):shutil.copy2((a.output/chosen['checkpoint']).with_suffix(ext),a.output/(variant+ext))
    summary=dict(status='complete',runs=runs,selected=selected,states=len(rows),updates=updates,
        training_s=time.perf_counter()-started,data_sha256=hashlib.sha256((a.data/'data.pkl').read_bytes()).hexdigest(),
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path(__file__).with_name('model.py'),Path(__file__).with_name('features.py')]},
        config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        torch=torch.__version__,device=torch.cuda.get_device_name() if a.device=='cuda' else a.device)
    write_json(a.output/'summary.json',summary);print(json.dumps(dict(status='complete',updates=updates,training_s=summary['training_s'])),flush=True)


if __name__=='__main__':main()
