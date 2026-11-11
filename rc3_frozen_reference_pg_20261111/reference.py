"""Frozen action-independent reference; components are ALREADY per initial n."""
import copy
import torch
from .common import read,ROOT,TABLE,sha

class FrozenReference:
    def __init__(self,expected_sha):
        if sha(ROOT/TABLE)!=expected_sha:raise ValueError('Frozen M2 reference table SHA mismatch')
        self.table=read(ROOT/TABLE);self.sha256=expected_sha
        if len(self.table['parent_ids'])!=33 or self.table['unknown_condition']!='fit-global':raise ValueError('Reference identity')
    def values(self,features,window):
        if features.ndim!=2 or features.shape[1]!=268:raise ValueError('Observable saved state shape')
        f=features[:,256].detach()
        if not bool(torch.isfinite(f).all()) or not bool(((f>=-1e-6)&(f<=1+1e-6)).all()):raise ValueError('Remaining fraction')
        q=torch.floor(4*(1-f.clamp(0,1))).long().clamp(0,3).tolist()
        global_components=torch.tensor(self.table['global'],dtype=features.dtype,device=features.device).expand(len(f),3)
        keys=[f'{window}|q{x}' for x in q]
        conditional=torch.tensor([self.table['conditional'].get(k,self.table['global']) for k in keys],dtype=features.dtype,device=features.device)
        compose=lambda a:(-a[:,0]-.75*a[:,1]+a[:,2]).detach()
        return dict(global_value=compose(global_components),conditional_value=compose(conditional),
            global_components=global_components.detach(),conditional_components=conditional.detach(),
            condition_keys=keys,fallback=[k not in self.table['conditional'] for k in keys])
