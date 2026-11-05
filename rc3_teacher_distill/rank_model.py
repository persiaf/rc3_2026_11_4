"""Matched linear / small MLP cost-advantage rankers. Supervised learning only."""
import time
import numpy as np
import torch
from torch import nn
from .features import NAMES


class Ranker(nn.Module):
    def __init__(self,kind):
        super().__init__()
        d=len(NAMES)
        self.net=(nn.Linear(d,1,bias=False) if kind=='linear' else
                  nn.Sequential(nn.Linear(d,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1)))

    def forward(self,x):
        z=self.net(x).squeeze(-1)
        return z-z[...,0:1]  # reference is always first; its true advantage is zero


def fit(records,kind,seeds,check,stats):
    rows=[r for r in records if r['partition']=='fit']
    flat=np.concatenate([np.asarray(r['features']) for r in rows])
    mean=flat.mean(0).astype(np.float32)
    std=np.maximum(flat.std(0),.05).astype(np.float32)
    X=[];Y=[];V=[]
    for r in rows:
        x=np.zeros((8,len(NAMES)),np.float32);y=np.zeros(8,np.float32);v=np.zeros(8,bool)
        k=len(r['menu']);x[:k]=(np.asarray(r['features'])-mean)/std
        y[:k]=100*(r['Yref']-np.asarray(r['Y']))/r['Yref'];v[:k]=True
        X.append(x);Y.append(y);V.append(v)
    X=torch.tensor(np.asarray(X));Y=torch.tensor(np.asarray(Y));V=torch.tensor(np.asarray(V))
    networks=[];history=[];started=time.perf_counter()
    for seed in seeds:
        check();torch.manual_seed(seed);model=Ranker(kind)
        opt=torch.optim.Adam(model.parameters(),lr=1e-3)
        rng=np.random.default_rng(seed)
        for step in range(1024):
            check();ind=rng.integers(len(rows),size=32)
            pred=model(X[ind]);target=Y[ind];valid=V[ind]
            regression=nn.functional.smooth_l1_loss(pred[valid],target[valid],beta=.5)
            gap=target[:,:,None]-target[:,None,:]
            pgap=pred[:,:,None]-pred[:,None,:]
            pair=valid[:,:,None]&valid[:,None,:]&(gap.abs()>1e-7)
            weights=gap.abs()*pair
            rank=(nn.functional.softplus(-gap.sign()*pgap/.25)*weights).sum()/weights.sum().clamp_min(1e-8)
            loss=regression+.25*rank
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite supervised loss')
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            stats['optimizer_updates']+=1
            stats['training_forward_batches']+=1
            stats['training_rows']+=32
            if step in (0,255,511,767,1023):
                history.append(dict(kind=kind,seed=seed,step=step+1,loss=float(loss.detach()),
                    value=float(regression.detach()),ranking=float(rank.detach())))
        networks.append(model.eval())
    return dict(kind=kind,seeds=seeds,mean=mean,std=std,networks=networks,
        history=history,training_s=time.perf_counter()-started,
        parameters=sum(p.numel() for p in networks[0].parameters()))


def predict(bundle,features,stats=None):
    x=torch.as_tensor((np.asarray(features,dtype=np.float32)-bundle['mean'])/bundle['std'])
    with torch.inference_mode():
        z=np.stack([m(x).numpy() for m in bundle['networks']])
    if stats is not None:
        stats['inference_forward_batches']+=len(bundle['networks'])
        stats['inference_action_rows']+=len(features)*len(bundle['networks'])
    if not np.isfinite(z).all():raise FloatingPointError('Nonfinite rank predictions')
    return z.mean(0),z.std(0)


def save(bundle,path,protocol_sha):
    torch.save(dict(kind=bundle['kind'],seeds=bundle['seeds'],mean=bundle['mean'],std=bundle['std'],
        states=[n.state_dict() for n in bundle['networks']],protocol_sha256=protocol_sha,
        features=NAMES,parameters=bundle['parameters'],optimizer_updates_each=1024),path)


def load(path,protocol_sha):
    started=time.perf_counter();d=torch.load(path,map_location='cpu',weights_only=False)
    assert d['protocol_sha256']==protocol_sha and d['features']==NAMES
    networks=[]
    for state in d['states']:
        model=Ranker(d['kind']);model.load_state_dict(state);networks.append(model.eval())
    return dict(kind=d['kind'],seeds=d['seeds'],mean=d['mean'],std=d['std'],networks=networks),time.perf_counter()-started
