"""G2 shared-memory search and transactional training-state restoration."""
from __future__ import annotations
import copy
from dataclasses import asdict
import math
import numbers
import numpy as np
from rc3_rl_research.rl_env import SearchEnv, _menu_from_dict, _menu_dict
from rc3_rl_research.types import SearchConfig, Certificate
from rc3_rl_research.repair_proposal import propose
from .action_memory import build_actions, proposal_key, is_blocked

SCHEMA='rc3-rl-g2-v1'


def _number(value,name,nonnegative=False):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,numbers.Real) or not math.isfinite(float(value)):
        raise ValueError(name+' must be a finite number, not bool')
    if nonnegative and value<0:raise ValueError(name+' must be nonnegative')


def _integer(value,name):
    if type(value) is not int or value<0:raise ValueError(name+' must be a nonnegative integer')


def _all_finite(value,path='state'):
    if isinstance(value,dict):
        for key,item in value.items():_all_finite(item,path+'.'+str(key))
    elif isinstance(value,(list,tuple)):
        for i,item in enumerate(value):_all_finite(item,path+f'[{i}]')
    elif isinstance(value,numbers.Real) and not isinstance(value,(bool,np.bool_)):
        if not math.isfinite(float(value)):raise ValueError('nonfinite '+path)


def _close(actual,expected,name):
    if not math.isclose(actual,expected,rel_tol=1e-8,abs_tol=1e-8):
        raise ValueError(name+' mismatch')


class SearchEnvG2(SearchEnv):
    def __init__(self,adapter,config=SearchConfig(),seed=0,*,record_transitions=True,
                 proposer=None,action_builder=None,certifier=None):
        self.attempt_memory={}
        self.record_transitions=bool(record_transitions)
        self._raw_proposer=propose if proposer is None else proposer
        self._raw_action_builder=action_builder
        self._raw_certifier=adapter.certify if certifier is None else certifier
        self._pending_proposal_key=None
        super().__init__(adapter,config,seed,proposer=self._remember_propose,
                         action_builder=self._build_actions,certifier=self._record_certificate)

    def _build_actions(self,problem,route,**kwargs):
        if self._raw_action_builder is not None:
            return self._raw_action_builder(problem,route,**kwargs)
        return build_actions(problem,route,attempt_memory=self.attempt_memory,**kwargs)

    def _remember_propose(self,problem,route,action,*,config):
        key=proposal_key(self.adapter.problem_id,route,action,config)
        self._pending_proposal_key=key
        result=self._raw_proposer(problem,route,action,config=config)
        old=self.attempt_memory.get(key,dict(attempts=0,no_change=False,failed=False))
        self.attempt_memory[key]=dict(attempts=old['attempts']+1,
            no_change=bool(result.valid and result.route==list(route)),
            failed=bool(not result.valid),last_failure=result.failure_reason,
            problem_id=self.adapter.problem_id,route=list(route),action=action.to_dict())
        return result

    def _record_certificate(self,route,method='rl_search'):
        """Record acceptance failure only after the mandatory certification.

        Given the same current route and fixed solver inputs, best never rises;
        a previously certified over-bound proposal can never become admissible.
        This information is shared by every policy before the next menu cap.
        """
        active=method=='rl_search_accept' and self._pending_proposal_key in self.attempt_memory
        try:
            certificate=self._raw_certifier(route,method=method)
        except Exception:
            if active:
                record=self.attempt_memory[self._pending_proposal_key]
                record.update(failed=True,last_failure='certification_failed',certificate_failed=True)
            raise
        if active:
            bound=self.best_cost+self.config.epsilon*self.anchor_cost
            if certificate.cost>bound+1e-8:
                record=self.attempt_memory[self._pending_proposal_key]
                record.update(failed=True,bound_rejected=True,last_failure='outside_best_plus_epsilon_anchor',
                              certified_proposal_cost=float(certificate.cost),acceptance_bound=float(bound))
        return certificate

    def observe(self):
        obs=super().observe();obs['schema']=SCHEMA
        actions=[] if self._menu is None else self._menu.actions
        keys=[proposal_key(self.adapter.problem_id,self.current,a,self.config) for a in actions]
        obs['proposal_keys']=keys
        obs['action_attempts']=[copy.deepcopy(self.attempt_memory.get(k,
            dict(attempts=0,no_change=False,failed=False))) for k in keys]
        obs['attempt_memory_size']=len(self.attempt_memory)
        return obs

    def step(self,action_id):
        output=super().step(action_id)
        info=output[-1]
        info['route_changed']=bool(info['accepted'] and info['proposal'] is not None
                                   and info['proposal']['route']!=info['sampled_observation']['current'])
        if self.transitions:self.transitions[-1]['info']['route_changed']=info['route_changed']
        if not self.record_transitions:self.transitions.clear()
        return output

    def terminate(self,reason='external_deadline'):
        was_done=self.done
        reward,event=super().terminate(reason)
        if not was_done:
            event['event_id']=f'{self.adapter.problem_id}:{self.seed}:{self.steps}:{len(self.events)-1}'
            self.events[-1]['event_id']=event['event_id']
        return reward,event

    def state_dict(self):
        state=super().state_dict();state['schema']=SCHEMA
        state.update(attempt_memory=copy.deepcopy(self.attempt_memory),record_transitions=self.record_transitions)
        return state

    def fork(self):
        """Trusted in-memory oracle branch; persisted states use validated load."""
        child=copy.copy(self)
        shared={'adapter','proposer','action_builder','certifier','_raw_proposer','_raw_action_builder','_raw_certifier'}
        for key,value in vars(self).items():
            if key not in shared:setattr(child,key,copy.deepcopy(value))
        child.proposer=child._remember_propose
        child.action_builder=child._build_actions
        child.certifier=child._record_certificate
        return child

    def load_state_dict(self,state):
        """Validate a detached candidate completely before committing fields."""
        state=copy.deepcopy(state);_all_finite(state)
        if state.get('schema')!=SCHEMA or state.get('problem_id')!=self.adapter.problem_id:
            raise ValueError('state schema/problem differs')
        if state.get('config')!=asdict(self.config):raise ValueError('state config differs')
        if type(state.get('terminated')) is not bool or type(state.get('record_transitions')) is not bool:
            raise ValueError('terminal and recording flags must be bool')
        for key in ('anchor_cost','current_cost','best_cost'):_number(state[key],key,True)
        for key in ('total_reward','total_quality_reward','total_work_penalty','total_shaping_reward'):
            _number(state[key],key,key=='total_work_penalty')
        for key in ('steps','work_used','seed'):_integer(state[key],key)
        if state['steps']>self.config.horizon:raise ValueError('steps exceed configured horizon')
        for key,value in state['timings'].items():_number(value,'timing '+key,True)
        if set(state['timings'])!=set(self.timings):raise ValueError('timing fields changed')
        if set(state['counters'])!=set(self.counters):raise ValueError('counter fields changed')
        for key,value in state['counters'].items():
            if key=='work_count_incomplete':
                if type(value) is not bool:raise ValueError('work flag must be bool')
            else:_integer(value,'counter '+key)
        if state['work_used']!=state['counters']['menu_work']+state['counters']['proposal_work']:
            raise ValueError('work counters disagree')
        if not isinstance(state['history'],list) or any(not isinstance(x,str) for x in state['history']):
            raise ValueError('history must contain context identities')
        if len(state['history'])!=state['counters']['proposal_calls']:
            raise ValueError('history and proposal count disagree')
        if state['anchor']!=self.anchor:raise ValueError('anchor route changed')
        _close(state['anchor_cost'],self.anchor_cost,'anchor cost')
        checked={}
        for name in ('anchor','current','best'):
            route=state[name]
            if not isinstance(route,list) or any(type(n) is not int for n in route):
                raise ValueError(name+' route must contain integer nodes')
            certificate=self.adapter.certify(route,method='g2_load_check')
            if certificate.route!=route:raise ValueError(name+' route not canonical')
            _close(certificate.cost,state[name+'_cost'],name+' route cost')
            checked[name]=certificate
        anchor,current,best=(state[k+'_cost'] for k in ('anchor','current','best'))
        if best>anchor+1e-8 or best>current+1e-8 or current>best+self.config.epsilon*anchor+1e-8:
            raise ValueError('best/current/anchor bounds violated')
        memory=state['attempt_memory']
        if not isinstance(memory,dict):raise ValueError('memory must be mapping')
        from rc3_rl_research.types import Action
        for key,record in memory.items():
            if not isinstance(key,str) or not isinstance(record,dict):raise ValueError('invalid memory identity')
            _integer(record['attempts'],'attempts')
            if not record['attempts'] or type(record['failed']) is not bool or type(record['no_change']) is not bool:
                raise ValueError('invalid memory outcome')
            for optional_flag in ('bound_rejected','certificate_failed'):
                if optional_flag in record and type(record[optional_flag]) is not bool:
                    raise ValueError('invalid rejection flag')
            if record.get('bound_rejected'):
                _number(record['certified_proposal_cost'],'recorded proposal cost',True)
                _number(record['acceptance_bound'],'recorded acceptance bound',True)
                if (not record['failed'] or record['certified_proposal_cost']<=record['acceptance_bound']+1e-8
                    or best+self.config.epsilon*anchor>record['acceptance_bound']+1e-8):
                    raise ValueError('invalid historical acceptance-bound rejection')
            if record.get('problem_id')!=self.adapter.problem_id:raise ValueError('foreign memory problem')
            if any(type(n) is not int for n in record['route']):raise ValueError('invalid memory route')
            action=Action(**record['action'])
            if proposal_key(self.adapter.problem_id,record['route'],action,self.config)!=key:
                raise ValueError('memory proposal identity differs')
        if sum(r['attempts'] for r in memory.values())>state['counters']['proposal_calls']:
            raise ValueError('memory attempts exceed actual calls')
        candidate=self.fork()
        for key in ('seed','anchor','current','best','anchor_cost','current_cost','best_cost','steps','work_used',
                    'history','terminated','stop_reason','finalization','transitions','events','total_reward',
                    'total_quality_reward','total_work_penalty','total_shaping_reward','counters','timings',
                    'attempt_memory','record_transitions'):
            setattr(candidate,key,state[key])
        candidate.anchor_certificate=checked['anchor'];candidate.best_certificate=checked['best']
        # Also verify persisted certificate payloads, not only their cost fields.
        for name in ('anchor','best'):
            saved=state[name+'_certificate']
            _number(saved['cost'],name+' certificate cost',True)
            cert=self.adapter.certify(saved['route'],method=saved['plan'].get('method','g2_load_check'))
            if saved['route']!=state[name] or cert.plan!=saved['plan']:
                raise ValueError('saved '+name+' certificate differs')
            _close(saved['cost'],state[name+'_cost'],name+' certificate cost')
            setattr(candidate,name+'_certificate',Certificate(saved['route'],saved['plan'],saved['cost'],saved.get('diagnostics',{})))
        returned=state['returned_certificate'];final=state['finalization']
        if state['terminated']:
            if returned is None or final is None or not isinstance(state['stop_reason'],str):
                raise ValueError('terminated state lacks final output')
            _number(returned['cost'],'returned cost',True)
            cert=self.adapter.certify(returned['route'],method=returned['plan'].get('method','g2_load_check'))
            if cert.plan!=returned['plan'] or cert.route!=returned['route']:raise ValueError('returned plan differs')
            _close(cert.cost,returned['cost'],'returned certificate')
            _number(final['returned_cost'],'final returned cost',True);_number(final['best_cost'],'final best cost',True)
            _number(final['quality_correction'],'final correction')
            _integer(final['work_overrun'],'work overrun')
            if type(final['fallback_used']) is not bool:raise ValueError('fallback flag not bool')
            if final['reason']!=state['stop_reason']:raise ValueError('termination reason differs')
            expected_route=state['anchor'] if final['fallback_used'] else state['best']
            if cert.route!=expected_route:raise ValueError('fallback/best output identity differs')
            _close(final['returned_cost'],cert.cost,'final returned cost')
            _close(final['best_cost'],best,'final best cost')
            _close(final['quality_correction'],100*(best-cert.cost)/max(anchor,1e-8),'final correction')
            if final['work_overrun']!=max(0,state['work_used']-self.config.work_budget):raise ValueError('work overrun differs')
            if state['menu'] is not None:raise ValueError('terminated state retains active menu')
            candidate.returned_certificate=Certificate(returned['route'],returned['plan'],returned['cost'],returned.get('diagnostics',{}))
        else:
            if returned is not None or final is not None or state['stop_reason'] is not None:
                raise ValueError('active state has final output')
            if state['steps']>=self.config.horizon or state['work_used']>=self.config.work_budget:
                raise ValueError('active state has exhausted budget or horizon')
            candidate.returned_certificate=None
        expected_quality=100*(anchor-(candidate.returned_certificate.cost if candidate.done else best))/max(anchor,1e-8)
        _close(state['total_quality_reward'],expected_quality,'quality total')
        _close(state['total_work_penalty'],self.config.lambda_work*state['work_used']/self.config.work_scale,'work penalty total')
        expected_shape=0. if candidate.done else self.config.potential_weight*100*(best-current)/max(anchor,1e-8)
        _close(state['total_shaping_reward'],expected_shape,'shaping total')
        _close(state['total_reward'],state['total_quality_reward']-state['total_work_penalty']+state['total_shaping_reward'],'reward total')
        candidate._menu=None
        if state['menu'] is not None:
            menu=state['menu'];n=len(menu['actions'])
            _integer(menu['work_units'],'menu work')
            if not n or any(len(menu[k])!=n for k in ('mask','features','action_context_keys')):
                raise ValueError('menu dimensions disagree')
            if any(type(m) is not bool for m in menu['mask']):raise ValueError('menu mask must be bool')
            for row in menu['features']:
                if len(row)!=len(menu['feature_names']):raise ValueError('feature dimensions disagree')
                for value in row:_number(value,'menu feature')
            expected=candidate._build_actions(self.adapter.problem,candidate.current,
                remaining_work=candidate.remaining_work,steps_remaining=self.config.horizon-candidate.steps,
                history=tuple(candidate.history),config=self.config,problem_id=self.adapter.problem_id)
            if _menu_dict(expected)!=menu:raise ValueError('menu does not belong to restored state/config/memory')
            candidate._menu=_menu_from_dict(menu)
        rng=np.random.default_rng();rng.bit_generator.state=state['rng_state'];candidate.rng=rng
        # Commit only now. The target state is untouched on every earlier error.
        self.__dict__.update(candidate.__dict__)
        self.proposer=self._remember_propose;self.action_builder=self._build_actions
        self.certifier=self._record_certificate;self._pending_proposal_key=None
        return self.state_dict()
