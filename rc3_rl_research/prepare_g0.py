"""Predeclare diagnostic states without generating scenes or examining outcomes."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from rc3_v3.data import historical_seeds
from .types import SearchConfig, SCHEMA


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('rc3_rl_research/outputs/g0'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    target=a.output/'protocol.json'
    if target.exists():raise SystemExit('Protocol already exists; refusing to change the registered sample')
    excluded,scan=historical_seeds(Path.cwd(),exclude=Path('rc3_rl_research'))
    rng=np.random.default_rng(20261103);cases=[];used=set(excluded)
    for i in range(32):
        while True:
            seed=int(rng.integers(100000,2**31-1))
            if seed not in used:break
        used.add(seed)
        cases.append(dict(case_id=i,seed=seed,nx=10,ny=10,mode=('regular','mixed','blocked')[i%3]))
    protocol=dict(schema=SCHEMA,phase='G0 development diagnosis, not confirmatory evaluation',
        sampling_seed=20261103,cases=cases,config=asdict(SearchConfig()),
        diagnostic_depth=3,branch_limit_including_stop=8,
        branch_rule='First seven valid non-STOP entries of the common interleaved detour menu, plus STOP; identical rule at every node.',
        starting_state='independently certified Fixed3; one initial state per new scene',
        no_scene_replacement=True,failures_retained=True,no_training=True,
        metrics=['strictly worse first step then best below anchor',
                 'nonimproving first step then best below anchor',
                 'strictly worse first step then best below every root one-step result',
                 'strict immediate-vs-descendant action ranking reversal',
                 'all queried actions, actual menu/proposal work, failures and rejected proposals'],
        accounting='Each executed tree edge charges its menu share to its counterfactual episode; physical oracle work counts each actually built parent menu once and every proposal once. No oracle is a deployment baseline.',
        decision_rule='G0 positive means at least one strict-worse sequence beats every queried root one-step result; report frequency across all 32, not an RL performance claim. G1 requires all regression/replay/freeze checks. No PPO training in this stage.',
        seed_exclusion_scope=scan['scope'],excluded_seed_count=len(excluded),
        seed_scan_unreadable=scan['unreadable'])
    target.write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    (a.output/'historical_seed_scan.json').write_text(json.dumps(scan,ensure_ascii=False,indent=2),encoding='utf-8')
    digest=hashlib.sha256(target.read_bytes()).hexdigest()
    (a.output/'protocol.sha256').write_text(digest+'  protocol.json\n',encoding='ascii')
    print(json.dumps(dict(protocol=str(target),sha256=digest,cases=32,excluded_seeds=len(excluded),unreadable=len(scan['unreadable']))))


if __name__=='__main__':main()
