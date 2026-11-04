"""Bounded fresh-action collector with independent G21 worker persistence."""
import multiprocessing as mp
import sys
import time
sys.dont_write_bytecode=True

import numpy as np

from rc3_rl_g2.collector import VectorCollector
from .worker_pilot import worker_main


class PilotCollector(VectorCollector):
    def __init__(self, cases, config, seed, run_dir, workers=8, actions_max=4096):
        if not cases or type(workers) is not int or workers < 1:
            raise ValueError('Nonempty cases and positive worker count required')
        if type(actions_max) is not int or not 1 <= actions_max <= 4096 or actions_max % workers:
            raise ValueError('Pilot action cap must be <=4096 and divisible by workers')
        self.workers=workers; self.connections=[]; self.processes=[]; self.observations=None
        self.actions_max=actions_max; self.committed=0
        context=mp.get_context('spawn')
        try:
            for i in range(workers):
                parent, child=context.Pipe()
                owned=cases[i::workers] or [cases[i % len(cases)]]
                process=context.Process(target=worker_main,
                    args=(child,owned,config,seed+i*100003,i,str(run_dir)),daemon=True)
                process.start(); child.close(); self.connections.append(parent); self.processes.append(process)
            self.observations=self.command_all('reset',[None]*workers)
        except Exception:
            self.close(); raise

    def collect(self, model, steps):
        import torch
        if type(steps) is not int or steps < 1 or self.committed+steps*self.workers > self.actions_max:
            raise ValueError('Collection request exceeds preregistered committed action cap')
        observations=[]; actions=[]; logps=[]; values=[]; rewards=[]; terminals=[]
        episodes=[]; kinds=[]; changed=[]; proposals=[]; transitions=[]
        started=time.perf_counter(); inference_s=0.; worker_s=0.
        for _ in range(steps):
            tick=time.perf_counter(); a,lp,v=model.act(self.observations)
            inference_s+=time.perf_counter()-tick
            observations.extend(self.observations); actions.extend(a.cpu().tolist()); logps.extend(lp.cpu().tolist())
            values.append(v.cpu().numpy())
            tick=time.perf_counter(); out=self.command_all('step',a.cpu().tolist()); worker_s+=time.perf_counter()-tick
            self.committed+=self.workers
            rewards.append([r['reward'] for r in out]); terminals.append([r['terminated'] for r in out])
            kinds.extend(r['action_kind'] for r in out); changed.extend(r['route_changed'] for r in out)
            proposals.extend(int(r['proposal_calls']) for r in out)
            transitions.extend(r['transition'] for r in out)
            episodes.extend(r['summary'] for r in out if r['summary'] is not None)
            self.observations=[r['observation'] for r in out]
        with torch.no_grad(): last=model(self.observations)[1].cpu().numpy()
        value_array=np.asarray(values,dtype=np.float32)
        next_values=np.concatenate((value_array[1:],last[None]),axis=0)
        next_values[np.asarray(terminals,dtype=bool)]=0.
        return dict(observations=observations,actions=np.asarray(actions,dtype=np.int64),
            old_logp=np.asarray(logps,dtype=np.float32), values=value_array,
            rewards=np.asarray(rewards,dtype=np.float32),terminated=np.asarray(terminals,dtype=bool),
            next_values=next_values,episodes=episodes,action_kinds=kinds,changed=np.asarray(changed),
            actual_proposal_calls=np.asarray(proposals,dtype=np.int64),transitions=transitions,
            elapsed_s=time.perf_counter()-started,inference_s=inference_s,worker_s=worker_s,
            collector_cut=True,external_finalization_used=False)

    def load_state_dict(self, states):
        raise ValueError('Pilot resume is intentionally disabled; audit partial committed work before a new authorization')
