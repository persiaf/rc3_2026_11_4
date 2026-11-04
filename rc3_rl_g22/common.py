"""Pure serialization, frozen-feedback semantics, and evaluation-only guards."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parent
METHODS=('fixed3','detour','stop','detour_early','continued_ce','ce_gain')

def jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return value.as_posix()
    if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(v) for v in value]
    return value

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(jsonable(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def digest(value):return hashlib.sha256(json.dumps(jsonable(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def check_contract(p):
    if p.get('schema')!='rc3-g22-frozen-evaluation-v1' or p.get('evaluation_authorized') is not True:
        raise ValueError('Separate frozen-evaluation protocol required')
    if p.get('training_allowed') is not False or p.get('ppo_allowed') is not False:
        raise ValueError('This evaluation cannot authorize any training')
    if p.get('max_plans')!=72 or type(p['max_plans']) is not int or p.get('methods')!=list(METHODS):
        raise ValueError('Only the registered six-method, 72-plan scope is allowed')
    cases=p.get('cases',[])
    if len(cases)!=12 or len({c['case_id'] for c in cases})!=12 or any(c['nx']!=10 or c['ny']!=10 for c in cases):
        raise ValueError('Exactly 12 distinct hundred-table development scenes required')
    if any(sum(c['mode']==mode for c in cases)!=4 for mode in ('regular','mixed','blocked')):
        raise ValueError('Four cases per mode required')
    if p['config']['horizon']!=16 or p['config']['work_budget']!=2_000_000:
        raise ValueError('Frozen shared horizon/work caps changed')
    if p['early_stop']!={'max_proposals':4,'patience':1,'progress_tolerance':1e-12}:
        raise ValueError('Early rule must remain fixed; no development sweep')
    if p['soft_complete_limit_s']!=5.0:raise ValueError('Registered complete soft deadline changed')
    return True

def teacher_index(samples,case):
    scene_id=f"{case['case_id']}:{case['seed']}"
    own=[s for s in samples if s['scene_id']==scene_id]
    if not own:
        # Scene IDs use a frozen descriptive suffix in some G2 exports.
        own=[s for s in samples if int(s['scene_id'].split(':',1)[0])==case['case_id']]
    if len(own)!=8 or sum(not s['path'] for s in own)!=1:raise ValueError('Expected eight existing teacher states')
    return {digest(s['environment_observation']):s for s in own}

def feedback_label(index,observation,action_id):
    sample=index.get(digest(observation))
    if sample is None:
        return dict(teacher_status='state_unseen',known_Q=None,known_delta_vs_stop=None,teacher_depth=None,
                    teacher_state_signature=None)
    if not 0<=action_id<len(sample['evidence_mask']) or not sample['legality_mask'][action_id]:
        raise ValueError('Selected action does not match the saved legal menu')
    known=bool(sample['evidence_mask'][action_id]);stop=int(sample['stop_id'])
    return dict(teacher_status='action_queried' if known else 'action_unqueried',
        known_Q=float(sample['q_values'][action_id]) if known else None,
        known_delta_vs_stop=float(sample['q_values'][action_id]-sample['q_values'][stop]) if known else None,
        teacher_depth=int(sample['teacher_depth']),teacher_state_signature=sample['state_signature'])

def finish_feedback(rows,episode_return,terminal_events):
    """Observed suffix returns under this frozen rollout, never an optimal Q."""
    for row in rows:
        row['realized_policy_suffix_return']=float(episode_return-row['prefix_return'])
        row['realized_after_action_return']=float(episode_return-row['prefix_return']-row['reward'])
        row['feedback_scope']='Executed action and actual frozen-policy continuation; not optimal Q or depth-2 teacher target'
    ledger=sum(r['reward'] for r in rows)+sum(e['reward'] for e in terminal_events)
    if abs(ledger-episode_return)>1e-7:raise ValueError('Transition/event reward ledger mismatch')
    return rows

def schedule(cases):
    return [(case,method) for i,case in enumerate(cases) for method in METHODS[i%6:]+METHODS[:i%6]]
