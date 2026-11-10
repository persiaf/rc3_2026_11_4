"""Offline source/object/actor integrity and exact action/reward replay, no H."""
import argparse
from .common import *
from .data import Inputs
from rc3_global_local.environment import Environment


class ReadClock:
    def check(self):pass


def verify(replay_actions=False):
    configure();verify_sources();events,tail=read_chain(OUT/'ledger/events.jsonl');head=read(OUT/'ledger/HEAD.json')
    require(head['sequence']==len(events) and head['sha256']==tail,'Unclosed ledger')
    class ReadLog:
        def __init__(self):self.events=events
        def __call__(self,row):raise ValueError('Read-only replay cannot create any new input/event')
    inputs=Inputs(ReadLog(),ReadClock());objects=0;replayed=0
    for e in events:
        if e['event']!='request_done':continue
        r=object_read(e['object']);objects+=1
        if replay_actions and 'result' in r:
            q=r['request'];stage=q['stage'];seed=str(q['seed'])
            if stage in ('main','no_anchor_KL','no_task_scale'):s=inputs.declared['training'][seed][q['round']-1][q['slot']]
            elif stage=='main_warm':s=inputs.declared['warm'][seed][q['round']][q['slot']]
            else:continue
            p,_=inputs.view(s);require(p.identity==r['input_identity'],'Replay input changed');env=Environment(p,lambda p,raw:r['result'])
            env.replay(r['actions']);require(np.allclose(env.rewards,r['rewards'],atol=1e-9,rtol=0),'Exact step reward replay mismatch');replayed+=1
    actors=0
    for p in (OUT/'actors').rglob('*.pt.json'):
        entry=read(p);require(digest(ROOT/entry['path'])==entry['actor_sha'],'Actor changed');d=torch.load(ROOT/entry['path'],map_location='cpu',weights_only=True)
        require(tensor_digest(d['model'])==entry['tensor_sha'],'Actor tensor changed');actors+=1
    print(json.dumps(dict(status='PASS',objects=objects,actors=actors,action_replays=replayed,ledger_tail=tail,
        new_model_forward=0,new_navigation=0,new_H=0,new_optimizer=0),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--actions',action='store_true');args=p.parse_args();verify(args.actions)
