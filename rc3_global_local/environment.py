"""Exact constructive MDP: positive gain until all local tasks, then specified exit."""
import numpy as np
from .problem import integer


class Environment:
    def __init__(self,problem,terminal):
        self.problem=problem;self.terminal=terminal;self.reset()

    def reset(self):
        self.remaining=np.ones(len(self.problem.C),dtype=bool)
        self.current=0;self.prefix=[];self.raw_L=0.;self.rewards=[];self.trace=[];self.result=None
        self.done=not self.remaining.any()
        if self.done:
            self.result=self.terminal(self.problem,[0,0])
            self.rewards=[-self.result['J']]
        return self.observe()

    def mask(self):
        if self.done:return np.zeros(len(self.problem.candidate_ids),dtype=bool)
        return self.problem.C[self.remaining].any(0)&np.isfinite(self.problem.D[self.current,1:])&np.isfinite(self.problem.D[1:,0])

    def observe(self):
        return dict(parent_identity=self.problem.parent.identity,problem_identity=self.problem.identity,
            remaining=self.remaining.copy(),current_local_node=self.current,
            entry_global_node=self.problem.entry,exit_global_node=self.problem.exit,
            ordered_prefix_local=list(self.prefix),ordered_prefix_global=[int(self.problem.candidate_ids[n-1]) for n in self.prefix],
            action_mask=self.mask(),done=self.done,raw_prefix_length=self.raw_L)

    def step(self,node):
        node=integer(node,'action node')
        if self.done or not 1<=node<=len(self.problem.candidate_ids) or not self.mask()[node-1]:
            raise ValueError('Illegal action or already terminal')
        served=np.flatnonzero(self.remaining & self.problem.C[:,node-1])
        reward=-float(self.problem.D[self.current,node])-self.problem.penalty
        self.raw_L+=float(self.problem.D[self.current,node]);self.current=node;self.prefix.append(node)
        self.remaining &= ~self.problem.C[:,node-1];self.done=not self.remaining.any()
        correction=0.
        if self.done:
            raw=[0,*self.prefix,0];raw_J=self.problem.metrics(raw)['J']
            self.result=self.terminal(self.problem,raw)
            correction=raw_J-self.result['J']
            reward-=float(self.problem.D[node,0]);reward+=correction
        self.rewards.append(reward)
        event=dict(action_local_node=node,action_global_candidate=int(self.problem.candidate_ids[node-1]),
            served_tasks_global=self.problem.task_ids[served].tolist(),reward=reward,terminal_H_correction=correction,
            remaining_tasks_global=self.problem.task_ids[np.flatnonzero(self.remaining)].tolist(),done=self.done)
        self.trace.append(event)
        if len(self.prefix)>len(self.problem.C):raise AssertionError('Positive-gain horizon exceeded')
        if self.done and abs(sum(self.rewards)+self.result['J'])>1e-7:raise AssertionError('Return is not negative complete terminal cost')
        return event

    def replay(self,actions):
        if self.prefix:raise ValueError('Replay requires a fresh environment')
        for a in actions:self.step(a)
        if not self.done:raise ValueError('Replay ended before covering all tasks')
        return self.result


def rule_action(env):
    ids=np.flatnonzero(env.mask())
    if not len(ids):raise ValueError('No legal action')
    gain=env.problem.C[env.remaining].sum(0)
    return int(min(ids,key=lambda j:(-gain[j]/(env.problem.D[env.current,j+1]+.75),int(env.problem.candidate_ids[j]))))+1
