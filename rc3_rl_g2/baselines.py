"""Specified shared-menu controls; no free all-action repair queries."""
from __future__ import annotations
import copy
from rc3_rl_research.policies import RandomPolicy, DetourPolicy


class StopPolicy(RandomPolicy):
    name='stop'
    def select(self,observation,rng):
        return next(i for i,(a,m) in enumerate(zip(observation['actions'],observation['mask']))
                    if m and a['kind']=='stop')


class DetourMemoryPolicy(DetourPolicy):
    """Highest cheap detour among the shared, pre-cap memory-filtered menu."""
    name='detour_memory'


class ILSPolicy(RandomPolicy):
    """Improve until stagnation, one perturbation, then another improvement pass.

    Default: 3 consecutive actions without a best improvement trigger replace;
    at most 2 perturbations, then another stagnation streak ends with STOP.
    Lack of an improve action also triggers a perturbation. No-op/failed results
    never receive an improvement score. All acceptance belongs to the shared env.
    """
    name='ils'
    def __init__(self,stagnation_limit=3,max_perturbations=2):
        if type(stagnation_limit) is not int or stagnation_limit<1:raise ValueError('invalid stagnation limit')
        if type(max_perturbations) is not int or max_perturbations<0:raise ValueError('invalid perturbation cap')
        self.stagnation_limit=stagnation_limit;self.max_perturbations=max_perturbations
        self.stagnation=0;self.perturbations=0;self.selections=0

    def select(self,obs,rng):
        legal=[i for i,m in enumerate(obs['mask']) if m]
        stop=next(i for i in legal if obs['actions'][i]['kind']=='stop')
        improve=[i for i in legal if obs['actions'][i]['kind']=='improve']
        replace=[i for i in legal if obs['actions'][i]['kind']=='replace']
        detour=obs['feature_names'].index('detour_excess')
        best=lambda rows:max(rows,key=lambda i:(obs['features'][i][detour],-i))
        self.selections+=1
        stalled=self.stagnation>=self.stagnation_limit or not improve
        if stalled:
            if self.perturbations>=self.max_perturbations or not replace:return stop
            return best(replace)
        return best(improve) if improve else stop

    def update(self,info):
        kind=info['action']['kind']
        if kind=='replace':self.perturbations+=1;self.stagnation=0
        elif info.get('best_improved') and info.get('route_changed',True):self.stagnation=0
        elif kind!='stop':self.stagnation+=1

    def state_dict(self):
        return dict(name=self.name,stagnation_limit=self.stagnation_limit,max_perturbations=self.max_perturbations,
                    stagnation=self.stagnation,perturbations=self.perturbations,selections=self.selections)

    def load_state_dict(self,state):
        if state.get('name')!=self.name:raise ValueError('wrong policy identity')
        other=ILSPolicy(state['stagnation_limit'],state['max_perturbations'])
        for key in ('stagnation','perturbations','selections'):
            if type(state[key]) is not int or state[key]<0:raise ValueError('invalid ILS counter')
            setattr(other,key,state[key])
        self.__dict__.update(other.__dict__)


def make_policy(name,**kwargs):
    factories={'stop':StopPolicy,'random':RandomPolicy,'detour_memory':DetourMemoryPolicy,'ils':ILSPolicy}
    if name not in factories:raise ValueError('unknown G2 control '+name)
    return factories[name](**kwargs)
