"""Exactly-once transfer of true external termination into actor samples.

Consumed IDs are per episode, and should be checkpointed with that episode.
A collector length boundary is not an event and must not call this merger.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import numbers
import numpy as np


def _finite(value, name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,numbers.Real) or not math.isfinite(float(value)):
        raise ValueError(name+' must be finite and numeric')
    return float(value)


def _event_key(event):
    explicit=event.get('event_id')
    if explicit is not None:
        if not isinstance(explicit,str) or not explicit:raise ValueError('invalid terminal event identity')
        return explicit
    # G1-compatible synthetic probes have no event_id. This digest is only
    # episode-local; it is not a global duplicate key across reset episodes.
    serialized=json.dumps(event,sort_keys=True,separators=(',',':'),allow_nan=False,
                          default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item())
    return hashlib.sha256(serialized.encode()).hexdigest()


def merge_terminal_events(records, events, final_observation, consumed_event_ids=None):
    """Return a copied record list and copied consumed-event ID set.

    Each unseen runtime_termination event is folded into the last real action:
    reward is added once, termination becomes true, and all bootstrap fields
    are zeroed. A zero-action episode never creates an actor sample; callers
    retain its event reward in episode metrics. Natural step termination with
    no events is unchanged. Merging a new event into an already natural-ended
    action is rejected instead of silently duplicating an episode end.
    """
    copied=copy.deepcopy(records)
    consumed=set() if consumed_event_ids is None else set(consumed_event_ids)
    if any(not isinstance(x,str) or not x for x in consumed):raise ValueError('invalid consumed event IDs')
    for record in copied:
        _finite(record['reward'],'record reward')
        if type(record['terminated']) is not bool:raise ValueError('record terminated flag must be bool')
    for event in events:
        if event.get('event')!='runtime_termination':raise ValueError('not a true external termination event')
        key=_event_key(event)
        if key in consumed:continue
        reward=_finite(event['reward'],'event reward')
        for name in ('quality_reward','work_penalty','shaping_reward'):
            if name in event:_finite(event[name],name)
        if all(name in event for name in ('quality_reward','work_penalty','shaping_reward')):
            expected=event['quality_reward']-event['work_penalty']+event['shaping_reward']
            if not math.isclose(reward,expected,rel_tol=1e-8,abs_tol=1e-8):
                raise ValueError('terminal event reward components disagree')
        if copied:
            last=copied[-1]
            if last['terminated'] and not last.get('external_terminal_event_ids'):
                raise ValueError('new external event follows an already natural terminal sample')
            if not isinstance(final_observation,dict) or final_observation.get('terminated') is not True:
                raise ValueError('external termination requires an actual terminal observation')
            last['reward']=_finite(last['reward']+reward,'merged reward')
            last['terminated']=True;last['truncated']=False
            last['next_observation']=copy.deepcopy(final_observation)
            last['bootstrap_value']=0.;last['next_value']=0.
            last['external_terminal_event_reward']=last.get('external_terminal_event_reward',0.)+reward
            last.setdefault('external_terminal_event_ids',[]).append(key)
        consumed.add(key)
    return copied,consumed
