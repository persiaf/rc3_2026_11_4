"""Fresh on-policy interaction; rollout boundaries preserve live environments."""
import multiprocessing as mp
import time
import numpy as np
from .worker import worker_main


class VectorCollector:
    def __init__(self,cases,config,seed=20261103,workers=8,cache_limit=1):
        self.workers=workers;self.connections=[];self.processes=[];self.observations=None
        context=mp.get_context('spawn')
        for i in range(workers):
            parent,child=context.Pipe()
            owned=cases[i::workers] or [cases[i%len(cases)]]
            process=context.Process(target=worker_main,args=(child,owned,config,seed+i*100003,cache_limit),daemon=True)
            process.start();child.close();self.connections.append(parent);self.processes.append(process)
        self.observations=self.command_all('reset',[None]*workers)

    def command_all(self,name,payloads):
        for connection,payload in zip(self.connections,payloads):connection.send((name,payload))
        results=[]
        for i,connection in enumerate(self.connections):
            # Timeout means a failed worker must not silently hang a training run.
            if not connection.poll(180):raise RuntimeError(f'worker {i} did not answer {name}')
            status,result=connection.recv()
            if status!='ok':raise RuntimeError(result)
            results.append(result)
        return results

    def state_dict(self):return self.command_all('state',[None]*self.workers)

    def load_state_dict(self,states):
        if len(states)!=self.workers:raise ValueError('worker count mismatch')
        self.observations=self.command_all('load',states)

    def collect(self,model,steps):
        import torch
        obslist=[];actions=[];logps=[];values=[];rewards=[];terminals=[];episodes=[];kinds=[];changed=[]
        started=time.perf_counter();inference_s=0.;worker_s=0.
        for _ in range(steps):
            tick=time.perf_counter()
            a,lp,v=model.act(self.observations)
            inference_s+=time.perf_counter()-tick
            obslist.extend(self.observations);actions.extend(a.cpu().tolist());logps.extend(lp.cpu().tolist())
            values.append(v.cpu().numpy())
            tick=time.perf_counter();out=self.command_all('step',a.cpu().tolist());worker_s+=time.perf_counter()-tick
            rewards.append([r['reward'] for r in out]);terminals.append([r['terminated'] for r in out])
            kinds.extend(r['action_kind'] for r in out);changed.extend(r['route_changed'] for r in out)
            episodes.extend(r['summary'] for r in out if r['summary'] is not None)
            self.observations=[r['observation'] for r in out]
        with torch.no_grad():last=model(self.observations)[1].cpu().numpy()
        value_array=np.asarray(values,dtype=np.float32)
        next_values=np.concatenate((value_array[1:],last[None]),axis=0)
        # Reset observations belong to a new episode; terminal V is exactly 0.
        next_values[np.asarray(terminals,dtype=bool)]=0.
        return dict(observations=obslist,actions=np.asarray(actions,dtype=np.int64),old_logp=np.asarray(logps,dtype=np.float32),
            values=value_array,rewards=np.asarray(rewards,dtype=np.float32),terminated=np.asarray(terminals,dtype=bool),
            next_values=next_values,episodes=episodes,action_kinds=kinds,changed=np.asarray(changed),
            elapsed_s=time.perf_counter()-started,inference_s=inference_s,worker_s=worker_s,
            collector_cut=True,external_finalization_used=False)

    def close(self):
        for connection,process in zip(self.connections,self.processes):
            if process.is_alive():
                try:connection.send(('close',None))
                except (BrokenPipeError,EOFError):pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():process.terminate();process.join()
        for connection in self.connections:connection.close()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
