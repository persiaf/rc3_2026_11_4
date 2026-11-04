"""Small set-attention actor/critic; actions score local repair windows."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch import nn
from .repair import FEATURE_NAMES

def batch_states(states,device):
    length=max(1,max(len(s[0]) for s in states))
    x=np.zeros((len(states),length,len(FEATURE_NAMES)),np.float32)
    mask=np.zeros((len(states),length),bool)
    for i,(features,valid) in enumerate(states):
        x[i,:len(features)]=features;mask[i,:len(valid)]=valid
    return torch.as_tensor(x,device=device),torch.as_tensor(mask,device=device)

class Policy(nn.Module):
    def __init__(self,mean,std,dim=64):
        super().__init__();self.dim=dim
        self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32))
        self.register_buffer('std',torch.tensor(std,dtype=torch.float32))
        self.embed=nn.Linear(len(mean),dim)
        self.encoder=nn.ModuleList([nn.TransformerEncoderLayer(dim,4,dim*2,dropout=0.,batch_first=True) for _ in range(2)])
        self.actor=nn.Sequential(nn.Linear(dim,dim),nn.ReLU(),nn.Linear(dim,1))
        self.value=nn.Sequential(nn.Linear(dim,dim),nn.ReLU(),nn.Linear(dim,1))
    def forward(self,x,mask):
        if not mask.any(1).all():raise ValueError('no legal action in policy batch')
        h=self.embed((x-self.mean)/self.std)
        for layer in self.encoder:h=layer(h,src_key_padding_mask=~mask)
        logits=self.actor(h).squeeze(-1).masked_fill(~mask,-torch.inf)
        pooled=(h*mask[...,None]).sum(1)/mask.sum(1)[:,None]
        return logits,self.value(pooled).squeeze(-1)
    def scores(self,features,mask):
        self.eval()
        with torch.no_grad():
            x,m=batch_states([(features,mask)],self.mean.device)
            return self(x,m)[0][0,:len(features)].cpu().numpy()

def save_policy(path,model,**meta):
    torch.save(dict(state_dict=model.state_dict(),mean=model.mean.cpu().tolist(),std=model.std.cpu().tolist(),dim=model.dim,feature_names=FEATURE_NAMES,**meta),path)

def load_policy(path,device='cpu'):
    torch.set_num_threads(1)
    checkpoint=torch.load(path,map_location=device,weights_only=True)
    if checkpoint['feature_names']!=FEATURE_NAMES:raise ValueError('feature schema mismatch')
    model=Policy(checkpoint['mean'],checkpoint['std'],checkpoint['dim']).to(device)
    model.load_state_dict(checkpoint['state_dict']);model.eval();return model
