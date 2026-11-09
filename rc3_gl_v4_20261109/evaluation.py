"""Unchanged greedy decisions, with actual masks/logits recorded for equivalence."""
import numpy as np
import torch
from unittest.mock import patch
from .common import *
from rc3_global_local.environment import Environment
from rc3_global_local.problem import array_hash


def greedy(model,p,meter):
    env=Environment(p,lambda q,r:dict(q.metrics(r),raw_route=list(r),route=list(r)))
    with torch.inference_mode():
        with meter.span('features_and_global_local_encoding'):
            encoded=model.encode(p);meter.add('fresh_encodings')
        with meter.span('greedy_construction'):
            while not env.done:
                meter.check();logits=model.score(env,encoded);mask=env.mask()
                legal=torch.as_tensor(mask,dtype=torch.bool)
                require(bool(legal.any()) and bool(torch.isfinite(logits[legal]).all()) and bool(torch.isneginf(logits[~legal]).all()),'Illegal/nonfinite greedy distribution')
                masks=(np.flatnonzero(mask)+1).tolist();ls=array_hash(logits.detach().cpu().numpy());ms=array_hash(mask)
                node=int(torch.argmax(logits).item())+1
                event=env.step(node);event.update(mask_local_nodes=masks,mask_sha256=ms,logits_sha256=ls)
                meter.add('greedy_actions')
    return list(env.result['raw_route']),env.trace


def evaluate(*args,**kwargs):
    from rc3_s3_evaluator import pipelines
    from rc3_s3_connectivity_20261109.scope import evaluate as scoped_evaluate
    with patch.object(pipelines,'greedy',greedy):result=scoped_evaluate(*args,**kwargs)
    case=args[2] if len(args)>2 else kwargs['case']
    if 'view_identity' in case:require(result['source']['view_identity']==case['view_identity'],'Frozen new DEV view identity changed')
    return result
